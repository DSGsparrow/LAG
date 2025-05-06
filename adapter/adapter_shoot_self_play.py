import copy

import gymnasium as gym
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from gymnasium import spaces
import torch.nn.functional as F
import torch
import logging


class ShootSelfPlayWrapper(gym.Env):
    def __init__(self, base_env_fn, opponent, args):
        super().__init__()
        self.env = base_env_fn()
        self.opponent = opponent

        self.history_len = args.history_len
        self.raw_obs_dim = args.raw_obs_dim
        self.fly_act_dim = args.fly_act_dim
        self.fire_act_dim = args.fire_act_dim
        self.total_act_dim = self.fly_act_dim + self.fire_act_dim

        self.is_eval = getattr(args, "is_eval", False)

        self.fly_model = PPO.load(args.fly_model_path)
        self.dodge_model = PPO.load(args.dodge_model_path)
        self.guide_model = PPO.load(args.guide_model_path)

        obs_act_dim = self.raw_obs_dim + self.total_act_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.history_len * obs_act_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.fire_act_dim,), dtype=np.float32)

        self.obs_history_self = deque(maxlen=self.history_len)
        self.act_history_self = deque(maxlen=self.history_len)
        self.obs_history_enemy = deque(maxlen=self.history_len)
        self.act_history_enemy = deque(maxlen=self.history_len)

        self.episode_data_self = []
        self.episode_data_enemy = []

        self.after_launch = False
        self.opponent_after_launch = False
        self.launch_index = None
        self.opponent_launch_index = None

        self.ammo_self = 1
        self.ammo_enemy = 1

        self.warmup_action = np.array(args.warmup_action, dtype=np.int32)

    def reset(self, **kwargs):
        obs = self.env.reset()
        self._clear_buffers()

        for _ in range(self.history_len):
            # warmup，使用固定动作
            maneuver_action_self = self.warmup_action[:3]
            fire_action_self = np.array([0.0, 0.0], dtype=np.float32)

            maneuver_action_enemy = self.warmup_action[:3]
            fire_action_enemy = np.array([0.0, 0.0], dtype=np.float32)

            act_self = np.concatenate([maneuver_action_self, fire_action_self])
            act_enemy = np.concatenate([maneuver_action_enemy, fire_action_enemy])

            norm_action_self = self.normalize_action(act_self)
            norm_action_enemy = self.normalize_action(act_enemy)
            full_action = np.stack([norm_action_self, norm_action_enemy]).astype(int)

            obs, reward, done, info = self.env.step(full_action)

            truncated = info.get('timeout', False)

            self.obs_history_self.append(obs[0])
            self.act_history_self.append(act_self)
            self.obs_history_enemy.append(obs[1])
            self.act_history_enemy.append(act_enemy)

            self.episode_data_self.append(
                [self._get_observation(self.obs_history_self, self.act_history_self), reward[0].item(), done.item(), truncated, info])
            self.episode_data_enemy.append(
                [self._get_observation(self.obs_history_enemy, self.act_history_enemy), reward[1].item(), done.item(), truncated, info])

        # return {
        #     'self_obs': self._get_observation(self.obs_history_self, self.act_history_self),
        #     'enemy_obs': self._get_observation(self.obs_history_enemy, self.act_history_enemy)
        # }, {}
        return self._get_observation(self.obs_history_self, self.act_history_self), {}

    def _select_maneuver_model(self, is_self):
        obs = self.obs_history_self[-1] if is_self else self.obs_history_enemy[-1]
        # obs = obs[:-5]
        ammo = self.ammo_self if is_self else self.ammo_enemy
        launched = self.after_launch if is_self else self.opponent_after_launch
        other_launched = self.opponent_after_launch if is_self else self.after_launch

        missile_flying = not np.allclose(obs[-6:], 0.0)

        if launched:
            return self.guide_model
        elif other_launched and missile_flying:
            return self.dodge_model
        elif other_launched and not missile_flying:
            return self.guide_model if ammo == 0 else self.fly_model
        else:
            return self.fly_model

    def _predict_maneuver_action(self, model, is_self):
        if model == self.fly_model:
            obs_input = self._get_observation(
                self.obs_history_self, self.act_history_self
            ) if is_self else self._get_observation(
                self.obs_history_enemy, self.act_history_enemy
            )
        else:
            obs_input = self.obs_history_self[-1] if is_self else self.obs_history_enemy[-1]

        return model.predict(obs_input, deterministic=True)

    def step(self, fire_action_self):
        # 机动动作
        maneuver_model_self = self._select_maneuver_model(is_self=True)
        maneuver_model_enemy = self._select_maneuver_model(is_self=False)

        maneuver_action_self, _ = self._predict_maneuver_action(maneuver_model_self, is_self=True)
        maneuver_action_enemy, _ = self._predict_maneuver_action(maneuver_model_enemy, is_self=False)

        # 发射动作
        maneuver_obs_enemy = self._get_observation(self.obs_history_enemy, self.act_history_enemy)
        fire_action_enemy, _ = self.opponent.predict(maneuver_obs_enemy, deterministic=True)

        act_self = np.concatenate([maneuver_action_self, fire_action_self])
        act_enemy = np.concatenate([maneuver_action_enemy, fire_action_enemy])

        norm_action_self = self.normalize_action(act_self)
        norm_action_enemy = self.normalize_action(act_enemy)
        full_action = np.stack([norm_action_self, norm_action_enemy]).astype(int)

        obs, reward, done, info = self.env.step(full_action)
        truncated = info.get('timeout', False)

        self.obs_history_self.append(obs[0])
        self.act_history_self.append(act_self)
        self.obs_history_enemy.append(obs[1])
        self.act_history_enemy.append(act_enemy)

        self.episode_data_self.append([
            self._get_observation(self.obs_history_self, self.act_history_self),
            reward[0].item(), done.item(), truncated, info
        ])
        self.episode_data_enemy.append([
            self._get_observation(self.obs_history_enemy, self.act_history_enemy),
            reward[1].item(), done.item(), truncated, info
        ])

        if info.get("launch", False):
            # todo luanch
            self.after_launch = True
            self.launch_index = len(self.episode_data_self) - 1
            self.ammo_self -= 1

        if info.get("opponent_launch", False):
            self.opponent_after_launch = True
            self.opponent_launch_index = len(self.episode_data_enemy) - 1
            self.ammo_enemy -= 1

        if self.after_launch:
            return self._run_until_done()

        # return {
        #     'self_obs': self._get_observation(self.obs_history_self, self.act_history_self),
        #     'enemy_obs': self._get_observation(self.obs_history_enemy, self.act_history_enemy)
        # }, {
        #     'self_reward': reward,
        #     'enemy_reward': reward,
        #     'done': done,
        #     'truncated': truncated,
        #     'info': info
        # }
        return (self.episode_data_self[-1][0], self.episode_data_self[-1][1], self.episode_data_self[-1][2],
                self.episode_data_self[-1][3], self.episode_data_self[-1][4])

    def _run_until_done(self):
        done = False
        info = {}
        cumulative_reward_self = 0
        cumulative_reward_enemy = 0

        # 保存发射时的状态
        self_obs_return = copy.deepcopy(self._get_observation(self.obs_history_self, self.act_history_self))
        enemy_obs_return = copy.deepcopy(self._get_observation(self.obs_history_enemy, self.act_history_enemy))

        while not done:
            # 产生机动动作
            maneuver_model_self = self._select_maneuver_model(is_self=True)
            maneuver_model_enemy = self._select_maneuver_model(is_self=False)

            maneuver_action_self, _ = self._predict_maneuver_action(maneuver_model_self, is_self=True)
            maneuver_action_enemy, _ = self._predict_maneuver_action(maneuver_model_enemy, is_self=False)

            # 发射动作
            fire_action_self = np.array([0.0, 0.0], dtype=np.float32)

            maneuver_obs_enemy = self._get_observation(self.obs_history_enemy, self.act_history_enemy)
            fire_action_enemy, _ = self.opponent.predict(maneuver_obs_enemy, deterministic=True)

            act_self = np.concatenate([maneuver_action_self, fire_action_self])
            act_enemy = np.concatenate([maneuver_action_enemy, fire_action_enemy])

            norm_action_self = self.normalize_action(act_self)
            norm_action_enemy = self.normalize_action(act_enemy)
            full_action = np.stack([norm_action_self, norm_action_enemy]).astype(int)

            obs, reward, done, info = self.env.step(full_action)
            truncated = info.get('timeout', False)

            self.obs_history_self.append(obs[0])
            self.act_history_self.append(act_self)
            self.obs_history_enemy.append(obs[1])
            self.act_history_enemy.append(act_enemy)

            if self.launch_index is not None:
                cumulative_reward_self += reward[0].item()
            # todo enemy reward calculate wrong
            # after enemy shoots enemy should stop cumulative
            if self.opponent_launch_index is not None:
                cumulative_reward_enemy += reward[1].item()

            if info.get("opponent_launch", False):
                self.opponent_after_launch = True
                self.opponent_launch_index = len(self.episode_data_enemy) - 1
                self.ammo_enemy -= 1

        if self.launch_index is not None:
            self.episode_data_self[self.launch_index][1] += cumulative_reward_self
        if self.opponent_launch_index is not None:
            self.episode_data_enemy[self.opponent_launch_index][1] += cumulative_reward_enemy

        logging.info("cumulative_reward_self: " + str(cumulative_reward_self))
        logging.info("cumulative_reward_enemy: " + str(cumulative_reward_enemy))

        # return {
        #     'self_obs': self_obs_return,
        #     'enemy_obs': enemy_obs_return
        # }, {
        #     'self_reward': self.episode_data_self[self.launch_index][1],
        #     'enemy_reward': self.episode_data_enemy[self.opponent_launch_index][1],
        #     'done': True,
        #     'truncated': True,
        #     'info': info
        # }
        return self_obs_return, self.episode_data_self[self.launch_index][1], True, True, info

    def _get_observation(self, obs_history, act_history):
        seq = [np.concatenate([o, a], axis=0) for o, a in zip(obs_history, act_history)]
        return np.concatenate(seq, axis=0)

    def _clear_buffers(self):
        self.obs_history_self.clear()
        self.act_history_self.clear()
        self.obs_history_enemy.clear()
        self.act_history_enemy.clear()
        self.episode_data_self = []
        self.episode_data_enemy = []
        self.after_launch = False
        self.opponent_after_launch = False
        self.launch_index = None
        self.opponent_launch_index = None
        self.ammo_self = 1
        self.ammo_enemy = 1

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

    def normalize_action(self, action, temperature=0.5, threshold=0.5, mode='train'):
        norm_action = np.zeros(4)
        norm_action[:3] = action[:3]

        logits = torch.tensor([action[3], action[4]])
        probs = F.softmax(logits / temperature, dim=0)
        act_prob = probs[0].item()
        do_act = threshold < act_prob
        norm_action[3] = 1.0 if do_act else 0.0

        if action[3] == action[4] == 0:
            norm_action[3] = 0.0

        return norm_action