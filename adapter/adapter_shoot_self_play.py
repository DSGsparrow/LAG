import copy
import gymnasium
import gymnasium as gym
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from gymnasium import spaces
import torch.nn.functional as F
import torch
import logging
from agent.upper_agent import UpperAgent


class SelfPlayUpperWrapper(gym.Env):
    def __init__(self, base_env, target_state):
        super().__init__()
        self.env = base_env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.num_agent = 2
        self.upper_agent = [UpperAgent() for _ in range(self.num_agent)]
        self.opponent_agent = None  # 先留空，后续用 set_opponent_agent 设置
        self.target_state = target_state

        self.guide_model = PPO.load("trained_model/guide/ppo_air_combat.zip")
        self.dodge_model = PPO.load("trained_model/dodge_missile/ppo_air_combat_dodge4.zip")

        self.agent_id = "A0100"
        self.opponent_id = "B0100"
        self.agent_ids = [self.agent_id, self.opponent_id]

        self.acc_reward = 0
        self.obs = None
        self.info = {}

    def set_opponent_agent(self, model_path):
        self.opponent_agent = PPO.load(model_path)

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.obs = obs
        # self.info = info
        self.acc_reward = 0

        for i in range(self.num_agent):
            self.upper_agent[i].reset()

        while not self._self_in_target_state():
            action_dict = {
                self.agent_id: self._select_fixed_model_action(),
                self.opponent_id: self._select_opponent_action()
            }
            action_packed = self.env.pack(action_dict)
            obs, reward, done, info = self.env.step(action_packed)
            self.acc_reward += reward[0]
            if done:
                # break
                obs = self.env.reset()
                info = {}
            self.obs = obs
            self.info = info

        ego_obs = self.obs[0]
        # 环境返回了双方的状态

        return ego_obs, self.info

    def step(self, action):
        if not self._self_in_target_state():
            self.acc_reward = 0
            terminated = False
            truncated = False

            while not self._self_in_target_state():
                action_dict = {
                    self.agent_id: self._select_fixed_model_action(),
                    self.opponent_id: self._select_opponent_action()
                }
                action_packed = self.env._pack(action_dict)
                obs, reward, done, info = self.env.step(action_packed)

                self.acc_reward += reward[0].item()
                self.obs = obs
                self.info = info

                if done:
                    # ✅ 根据 info 推断 terminated / truncated
                    terminated = done.item()
                    truncated = info.get("timeout", False)
                    break

            ego_obs = self.obs[0]
            return ego_obs, self.acc_reward, terminated, truncated, self.info

        # === 正常训练阶段 ===
        oppo_action = self._select_opponent_action()
        action_dict = {
            self.agent_id: action,
            self.opponent_id: oppo_action
        }
        action_packed = self.env._pack(action_dict)
        obs, reward, done, info = self.env.step(action_packed)
        self.obs = obs
        self.info = info
        ego_obs = self.obs[0]

        # ✅ 判断 terminated 和 truncated
        terminated = done.item()
        truncated = info.get("timeout", False)

        return ego_obs, reward[0].item(), terminated, truncated, info

    def _self_in_target_state(self):
        missile_launching = self.env.agents[self.agent_id].check_missile_launching() is not None
        missile_warning = self.env.agents[self.agent_id].check_missile_warning() is not None
        state = self.upper_agent[0].select_maneuver_model(missile_launching, missile_warning)
        return state == self.target_state

    def _select_fixed_model_action(self):
        # 这个函数就是给我方的
        obs = self.obs[0]
        state = self.upper_agent[0].current_state

        with torch.no_grad():
            if state == 1:
                action = np.concatenate([self.guide_model.predict(obs, deterministic=True)[0], [0]])
            elif state == 2:
                action = np.concatenate([self.dodge_model.predict(obs, deterministic=True)[0], [0]])
            else:
                raise ValueError("Invalid fixed model state for self agent")
        return action

    def _select_opponent_action(self):
        # 这个函数就是给敌方的
        obs = self.obs[1]
        missile_launching = self.env.agents[self.opponent_id].check_missile_launching() is not None
        missile_warning = self.env.agents[self.opponent_id].check_missile_warning() is not None
        state = self.upper_agent[1].select_maneuver_model(missile_launching, missile_warning)

        with torch.no_grad():
            if state == 0:
                action = self.opponent_agent.predict(obs, deterministic=True)[0]
            elif state == 1:
                action = np.concatenate([self.guide_model.predict(obs, deterministic=True)[0], [0]])
            elif state == 2:
                action = np.concatenate([self.dodge_model.predict(obs, deterministic=True)[0], [0]])
            else:
                raise ValueError("Invalid fixed model state for opponent")
        return action


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


class SelfPlayDodgeWrapper(gym.Env):
    def __init__(self, base_env_fn, args):
        super().__init__()
        self.env = base_env_fn()

        # 参数
        self.history_len = args.history_len
        self.raw_obs_dim = args.raw_obs_dim
        self.action_dim = args.action_dim  # 单边动作维度

        self.fly_act_dim = args.fly_act_dim
        self.fire_act_dim = args.fire_act_dim

        # 组合后的 observation 空间，仍为“单个智能体”的输入
        obs_act_dim = self.raw_obs_dim + self.action_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.history_len * obs_act_dim,), dtype=np.float32
        )
        self.action_space = self.env.action_space  # 假设为单边动作空间

        # 历史状态动作缓存（每方一个）
        self.obs_history = [deque(maxlen=self.history_len), deque(maxlen=self.history_len)]
        self.act_history = [deque(maxlen=self.history_len), deque(maxlen=self.history_len)]

        self.warmup_action = np.array(args.warmup_action, dtype=np.int32)

        # 累计奖励
        self.total_rewards = [0.0, 0.0]

    def reset(self, **kwargs):
        obs_pair = self.env.reset()  # obs_pair: (2, obs_dim)
        for i in [0, 1]:
            self.obs_history[i].clear()
            self.act_history[i].clear()
            self.total_rewards[i] = 0.0

        # warmup 执行
        for _ in range(self.history_len):
            norm_action = self.normalize_action(self.warmup_action)
            obs_pair, reward_pair, done, info = self.env.step([norm_action, norm_action])
            for i in [0, 1]:
                self.obs_history[i].append(obs_pair[i])
                self.act_history[i].append(self.warmup_action)

        return self._get_observations(), {}

    def step(self, action_pair):
        """
        :param action_pair: list or tuple of two actions [action0, action1]
        动作是发射的概率，2位
        """
        full_actions = []
        for i in range(len(action_pair)):
            norm_action = self.normalize_action(action_pair[i])
            full_actions.append(norm_action)
        full_actions = np.array(full_actions)

        obs_pair, reward_pair, done, info = self.env.step(full_actions)

        # 判断是否 timeout
        truncated = info.get("timeout", False)

        # 保存数据
        for i in [0, 1]:
            self.obs_history[i].append(obs_pair[i])
            self.act_history[i].append(action_pair[i])
            self.total_rewards[i] += reward_pair[i]

        return self._get_observations(), reward_pair, done, truncated, info

    def _get_observations(self):
        """
        返回双方的 observation 序列
        """
        obs_seq = []
        for i in [0, 1]:
            seq = [np.concatenate([o, a], axis=0) for o, a in zip(self.obs_history[i], self.act_history[i])]
            obs_seq.append(np.concatenate(seq, axis=0))
        return obs_seq  # [obs0_seq, obs1_seq]

    def get_total_rewards(self):
        return self.total_rewards

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

    def normalize_action(self, action, temperature=0.5, threshold=0.3, mode='train'):
        norm_action = np.zeros(4)
        norm_action[:3] = action[:3]

        logits = torch.tensor([action[3], action[4]])
        probs = F.softmax(logits / temperature, dim=0)
        act_prob = probs[0].item()
        do_act = threshold < act_prob
        norm_action[3] = 1.0 if do_act else 0.0

        if action[3] == action[4] == 0:
            norm_action[3] = 0.0

        return norm_action.astype(np.int32)