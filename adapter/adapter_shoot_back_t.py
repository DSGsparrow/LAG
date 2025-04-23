import gymnasium as gym
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from gymnasium import spaces

import torch.nn.functional as F
import torch
import logging


class ShootControlWrapper(gym.Env):
    def __init__(self, base_env_fn, args):
        super().__init__()
        self.env = base_env_fn()

        # 参数配置
        self.history_len = args.history_len
        self.raw_obs_dim = args.raw_obs_dim
        self.fly_act_dim = args.fly_act_dim
        self.fire_act_dim = args.fire_act_dim
        self.total_act_dim = self.fly_act_dim + self.fire_act_dim

        self.is_eval = getattr(args, "is_eval", False)

        # 加载模型
        self.fire_model = PPO.load(args.fire_model_path)  # 发射判断模型
        self.guide_model = PPO.load(args.guide_model_path)  # 发射后导引策略

        # 状态空间：(obs + act) * history_len
        obs_act_dim = self.raw_obs_dim + self.total_act_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.history_len * obs_act_dim,), dtype=np.float32
        )

        # 训练动作空间来自环境（我们现在训练的是发射前的飞行动作）
        self.action_space = self.env.action_space

        # 缓存
        self.obs_history = deque(maxlen=self.history_len)
        self.act_history = deque(maxlen=self.history_len)

        # 状态标志
        self.in_warmup = False
        self.after_launch = False
        self.launch_index = None
        self.episode_data = []

        self.warmup_action = np.array(args.warmup_action, dtype=np.float32)

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        self.obs_history.clear()
        self.act_history.clear()
        self.episode_data = []

        self.in_warmup = False
        self.after_launch = False
        self.launch_index = None

        for _ in range(self.history_len):
            norm_action = self.normalize_action(self.warmup_action, mode='eval' if self.is_eval else 'train')
            obs, reward, done, truncated, info = self.env.step(norm_action)
            self.obs_history.append(obs)
            self.act_history.append(self.warmup_action)
        self.episode_data.append([self._get_observation(), 0.0, False, False, {}])

        return self._get_observation(), {}

    def step(self, fly_action):
        # 使用 fire_model 判断是否发射（输入为历史轨迹）
        fire_input = self._get_observation()
        fire_logits, _ = self.fire_model.predict(fire_input, deterministic=True)
        is_fire = self._fire_decision_from_logits(fire_logits)

        # 当前整体动作 = [飞行动作 + fire决策（0或1）]
        full_action = np.concatenate([fly_action, [is_fire]])

        # 执行动作
        obs, reward, done, truncated, info = self.env.step(full_action)

        self.obs_history.append(obs)

        his_action = np.concatenate([fly_action, fire_logits])
        self.act_history.append(his_action)
        observation = self._get_observation()
        self.episode_data.append([observation, reward, done, truncated, info])

        # 如果 fire_model 判断需要发射导弹
        if info.get("launch", False):
            self.after_launch = True
            self.launch_index = len(self.episode_data) - 1

            # 发射后使用 guide_model 控制，直到 episode 结束
            done = False
            while not done:
                guide_action, _ = self.guide_model.predict(obs, deterministic=True)
                guide_full_action = np.concatenate([guide_action, [0.0]])  # 不再发射
                obs, reward, done, truncated, info = self.env.step(guide_full_action)

                # 奖励加到发射动作那一帧
                if self.launch_index is not None:
                    self.episode_data[self.launch_index][1] += reward

            # 记录奖励
            cumulative_reward = sum(x[1] for x in self.episode_data)
            logging.info("cumulative_reward: " + str(cumulative_reward))

            # 训练应只训练 launch 之前的动作（即 self.launch_index 对应的 fly_action）
            return observation, self.episode_data[self.launch_index][1], True, True, info

        # 正常回合结束
        if done:
            cumulative_reward = sum(x[1] for x in self.episode_data)
            logging.info("cumulative_reward: " + str(cumulative_reward))

        return observation, reward, done, truncated, info

    def _get_observation(self):
        seq = [np.concatenate([o, a], axis=0) for o, a in zip(self.obs_history, self.act_history)]
        return np.concatenate(seq, axis=0)

    def _fire_decision_from_logits(self, logits, temperature=0.5, threshold=0.5):
        if isinstance(logits, np.ndarray):
            logits = torch.tensor(logits)
        probs = F.softmax(logits / temperature, dim=0)
        act_prob = probs[0].item()
        return float(act_prob > threshold)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

    def normalize_action(self, action, temperature=0.5, threshold=0.5, mode='train'):
        """
        用于 reset 阶段的 warmup 动作处理（不用于训练阶段）
        """
        norm_action = np.zeros(4)
        norm_action[0] = action[0]
        norm_action[1] = action[1]
        norm_action[2] = action[2]

        # logits = torch.tensor([action[3], action[4]])
        # probs = F.softmax(logits / temperature, dim=0)
        # act_prob = probs[0].item()
        #
        # if mode == 'train':
        #     do_act = threshold < act_prob
        # else:
        #     do_act = threshold < act_prob
        #
        # norm_action[3] = 1.0 if do_act else 0.0

        # if action[3] == action[4] == 0:
        norm_action[3] = 0.0

        return norm_action
