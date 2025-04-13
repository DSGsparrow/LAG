import gymnasium as gym
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from gymnasium import spaces


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

        # 模型加载
        self.fly_model = PPO.load(args.fly_model_path)
        self.guide_model = PPO.load(args.guide_model_path)

        # 状态 = (obs + act) * history_len
        obs_act_dim = self.raw_obs_dim + self.total_act_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.history_len * obs_act_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.fire_act_dim,), dtype=np.float32)

        # 缓存
        self.obs_history = deque(maxlen=self.history_len)
        self.act_history = deque(maxlen=self.history_len)

        # 控制逻辑
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
            obs, reward, done, truncated, info = self.env.step(self.warmup_action)
            self.obs_history.append(obs)
            self.act_history.append(self.warmup_action)
            self.episode_data.append([self._get_observation(), reward, done, truncated, info])

        return self._get_observation(), {}

    def step(self, fire_action):
        # 正常阶段
        fly_action, _ = self.fly_model.predict(self.obs_history[-1], deterministic=True)
        action = np.concatenate([fly_action, fire_action])

        obs, reward, done, truncated, info = self.env.step(action)
        self.obs_history.append(obs)
        self.act_history.append(action)
        observation = self._get_observation()
        self.episode_data.append([observation, reward, done, truncated, info])

        if info.get("launch", False):
            self.after_launch = True
            self.launch_index = len(self.episode_data) - 1

            # 立刻进入制导阶段并走到底
            done = False
            while not done:
                guide_action, _ = self.guide_model.predict(obs, deterministic=True)
                full_action = np.concatenate([guide_action, np.zeros(self.fire_act_dim)])
                obs, reward, done, truncated, info = self.env.step(full_action)

            # 奖励加到打弹那一帧
            if self.launch_index is not None:
                self.episode_data[self.launch_index][1] += reward

            return observation, self.episode_data[self.launch_index][1], True, True, info

        return observation, reward, done, truncated, info

    def _get_observation(self):
        seq = [np.concatenate([o, a], axis=0) for o, a in zip(self.obs_history, self.act_history)]
        return np.concatenate(seq, axis=0)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()
