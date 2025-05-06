import gymnasium
from gymnasium import spaces
import numpy as np
import logging
from collections import deque

from LAGmaster.envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, SingleCombatEnvShoot, SingleCombatEnvDodge


# ========== 1. 适配 SB3 的自定义环境 ==========
class SB3SingleCombatEnv(gymnasium.Env):
    """将 SingleCombatEnvTest 适配为 SB3 兼容的 Gym 环境"""

    def __init__(self, env_id, config_name):
        super(SB3SingleCombatEnv, self).__init__()
        self.env = SingleCombatEnvDodge(config_name, env_id)  # 你的原始环境
        # obs_shape = self.env.get_obs().shape[0]  # 获取观测空间维度
        # act_shape = self.env.get_action_space().shape[0]  # 获取动作空间维度
        # 继承原始环境的动作空间和观察空间
        # self.action_space = self.env.action_space
        self.episode_num = 0
        self.win_num = 0

        # 提取原始环境的 action_space
        if isinstance(self.env.action_space, spaces.Tuple):
            # 获取所有离散动作空间的维度
            action_dims = []
            for space in self.env.action_space.spaces:
                if isinstance(space, spaces.MultiDiscrete):
                    action_dims.extend(space.nvec)  # 展开 MultiDiscrete
                elif isinstance(space, spaces.Discrete):
                    action_dims.append(space.n)  # Discrete 直接添加
                else:
                    raise ValueError("Unsupported action space type: {}".format(type(space)))

            # 转换为 MultiDiscrete
            self.action_space = spaces.MultiDiscrete(action_dims)
        else:
            if isinstance(self.env.action_space, spaces.MultiDiscrete):
                self.action_space = self.env.action_space
            else:
                raise ValueError("Unexpected action space type: {}".format(type(self.env.action_space)))

        self.observation_space = self.env.observation_space

        # # 定义 Gym 兼容的观测和动作空间
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(act_shape,), dtype=np.float32)

    def step(self, action):
        # 将长度为 4 的动作转换为长度为 (1,4) 的动作
        # actual_action = action.reshape(-1, 3)  # 取第一个值
        # 因为内部的环境，假设可能有多架飞机在控制，所有第一位都加了个序号
        # reward 和dones什么的直接取值

        action = np.expand_dims(action, axis=0) if action.ndim == 1 else action

        obs, rewards, dones, info = self.env.step(action)

        timeout = info.get('timeout', False)

        observation, reward, terminated, truncated, info = obs[0], rewards.item(), dones.item(), timeout, info

        # logging.info('test')
        if terminated or truncated:
            self.episode_num += 1
            shoot_success = info.get("dodge success", False)
            if shoot_success:
                self.win_num += 1
            logging.info(
                f'test {self.episode_num} episode, dodge {self.win_num} times, win rate is{self.win_num / self.episode_num * 100}%')

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """重置环境，支持 `seed` 以适配 SB3"""
        super().reset(seed=seed)  # 让 Gym 兼容 SB3 的 `seed`
        obs = self.env.reset()
        observation = obs[0]
        return observation, None

    def close(self):
        return self.env.close()

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi', tacview=None):
        self.env.render(mode=mode, filepath=filepath, tacview=tacview)



class DodgeControlWrapper(gymnasium.Env):
    def __init__(self, base_env_fn, args):
        super().__init__()
        self.env = base_env_fn()

        # 参数配置
        self.history_len = args.history_len
        self.raw_obs_dim = args.raw_obs_dim
        self.action_dim = args.action_dim  # 躲弹动作维度（例如飞行动作维度）

        # 状态空间：(obs + act) * history_len
        obs_act_dim = self.raw_obs_dim + self.action_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.history_len * obs_act_dim,), dtype=np.float32
        )

        # 动作空间直接来自底层环境
        self.action_space = self.env.action_space

        # 历史轨迹缓存
        self.obs_history = deque(maxlen=self.history_len)
        self.act_history = deque(maxlen=self.history_len)

        # 初始 warmup 动作
        self.warmup_action = np.array(args.warmup_action, dtype=np.int32)

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        self.obs_history.clear()
        self.act_history.clear()

        # Warmup 动作执行，填充历史状态动作
        for _ in range(self.history_len):
            obs, reward, done, truncated, info = self.env.step(self.warmup_action)
            self.obs_history.append(obs)
            self.act_history.append(self.warmup_action)

        return self._get_observation(), {}

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        self.obs_history.append(obs)
        self.act_history.append(action)

        observation = self._get_observation()
        return observation, reward, done, truncated, info

    def _get_observation(self):
        seq = [np.concatenate([o, a], axis=0) for o, a in zip(self.obs_history, self.act_history)]
        return np.concatenate(seq, axis=0)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

















