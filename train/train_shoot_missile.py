import gym
import gymnasium
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import argparse
import os
import logging

from LAGmaster.envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, SingleCombatEnvShoot

# ========== 1. 适配 SB3 的自定义环境 ==========
class SB3SingleCombatEnv(gymnasium.Env):
    """将 SingleCombatEnvTest 适配为 SB3 兼容的 Gym 环境"""

    def __init__(self, env_id, config_name):
        super(SB3SingleCombatEnv, self).__init__()
        self.env = SingleCombatEnvShoot(config_name, env_id)  # 你的原始环境
        # obs_shape = self.env.get_obs().shape[0]  # 获取观测空间维度
        # act_shape = self.env.get_action_space().shape[0]  # 获取动作空间维度
        # 继承原始环境的动作空间和观察空间
        # self.action_space = self.env.action_space

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


# ========== 2. MLPBase（特征提取） ==========
class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPBase, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.network(x)


# ========== 3. GRULayer（时间序列建模） ==========
class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_dim = hidden_dim

    def forward(self, x):
        if x.dim() == 2:  # (batch, features)
            x = x.unsqueeze(1)  # 变成 (batch, 1, features)，保证 GRU 兼容
        elif x.dim() == 4:  # (batch, 1, seq_len, features)
            x = x.squeeze(1)  # 去掉多余的 batch 维度
        x, _ = self.gru(x)  # GRU 处理
        return x.squeeze(1)  # (batch, features)


# ========== 4. ACTLayer（动作决策层） ==========
class ACTLayer(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ACTLayer, self).__init__()

        action_dims = list(action_dim.nvec)  # 转换成 Python list 以确保兼容性

        # 创建独立的输出层，每个维度一个 Linear 层
        self.action_heads = nn.ModuleList([nn.Linear(input_dim, dim) for dim in action_dims])

    def forward(self, x):
        return [head(x) for head in self.action_heads]


# ========== 5. 自定义策略网络（Feature + GRU + Action） ==========
class CustomPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, action_dim):
        super(CustomPolicy, self).__init__(observation_space, features_dim=128)

        input_dim = observation_space.shape[0]
        hidden_dim = 128  # 隐藏层大小
        self.feature_extractor = MLPBase(input_dim, hidden_dim)
        self.gru_layer = GRULayer(hidden_dim, hidden_dim)  # GRU 处理时间序列
        self.act_layer = ACTLayer(hidden_dim, action_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.gru_layer(features)
        return features  # 只返回 Tensor，不要返回 Tuple


def get_config():
    parser = argparse.ArgumentParser(description="PPO Training for Single Combat Dodge Missile Scenario")

    # 环境相关参数
    parser.add_argument("--env-name", type=str, default="SingleCombat", help="环境名称")
    parser.add_argument("--algorithm-name", type=str, default="ppo", help="算法名称")
    parser.add_argument("--scenario-name", type=str, default="1v1/DodgeMissile/HierarchyVsBaseline", help="场景名称")
    parser.add_argument("--experiment-name", type=str, default="1v1", help="实验名称")

    # 训练设置
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--n-training-threads", type=int, default=1, help="训练时的线程数")
    parser.add_argument("--n-rollout-threads", type=int, default=1, help="采样线程数")
    parser.add_argument("--cuda", action="store_true", help="是否使用 CUDA 加速")

    # 记录与保存
    parser.add_argument("--log-interval", type=int, default=1, help="日志记录间隔（单位：回合）")
    parser.add_argument("--save-interval", type=int, default=1, help="模型保存间隔（单位：回合）")

    # 评估设置
    parser.add_argument("--n-choose-opponents", type=int, default=1, help="选择的对手数量")
    parser.add_argument("--use-eval", action="store_true", help="是否使用评估模式")
    parser.add_argument("--n-eval-rollout-threads", type=int, default=1, help="评估时的 rollout 线程数")
    parser.add_argument("--eval-interval", type=int, default=1, help="评估间隔")
    parser.add_argument("--eval-episodes", type=int, default=1, help="每次评估的 episode 数")

    # PPO 训练超参数
    parser.add_argument("--num-mini-batch", type=int, default=5, help="PPO 的 mini-batch 数量")
    parser.add_argument("--buffer-size", type=int, default=200, help="经验缓冲区大小")
    parser.add_argument("--num-env-steps", type=float, default=1e8, help="训练环境步数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--ppo-epoch", type=int, default=4, help="PPO 训练的 epoch 数")
    parser.add_argument("--clip-params", type=float, default=0.2, help="PPO 裁剪参数")
    parser.add_argument("--max-grad-norm", type=float, default=2, help="梯度裁剪最大范数")
    parser.add_argument("--entropy-coef", type=float, default=1e-3, help="熵正则系数")

    # 神经网络结构
    parser.add_argument("--hidden-size", type=int, nargs="+", default=[128, 128], help="Actor-Critic 网络的隐藏层大小")
    parser.add_argument("--act-hidden-size", type=int, nargs="+", default=[128, 128], help="Actor 网络的隐藏层大小")
    parser.add_argument("--recurrent-hidden-size", type=int, default=128, help="RNN 隐藏层大小")
    parser.add_argument("--recurrent-hidden-layers", type=int, default=1, help="RNN 隐藏层数")
    parser.add_argument("--data-chunk-length", type=int, default=8, help="RNN 训练时的数据块长度")

    return parser


def setup_logging(log_file = None):
    """配置 logging，让日志既输出到终端，又写入 run.log 文件"""
    # 获取全局 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设定最低日志级别

    # 清除已有的 handlers，防止重复添加
    logger.handlers.clear()

    # 终端 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 文件 Handler
    file_handler = logging.FileHandler(log_file, mode="a")  # "a" 追加模式
    file_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s [PID %(process)d]- %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加 handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info("init complete, log path: " + log_file)


# ========== 6. 训练 PPO ==========
if __name__ == "__main__":
    num_envs = 16  # 设定 8 个并行环境（根据 GPU 性能调整）

    log_file = "./train/result/train_shoot.log"

    # setup_logging(log_file)
    # 创建并行环境
    def make_env(env_id):
        setup_logging(log_file)
        return SB3SingleCombatEnv(env_id, config_name='1v1/ShootMissile/HierarchyVsBaseline_self')

    # env = SubprocVecEnv([lambda: make_env(env_id) for env_id in range(num_envs)])
    env = SubprocVecEnv([lambda env_id=env_id: make_env(env_id) for env_id in range(num_envs)])

    # 定义 PPO 模型（自定义 MLP 作为特征提取器）
    policy_kwargs = dict(
        features_extractor_class=CustomPolicy,
        features_extractor_kwargs=dict(action_dim=env.action_space)
    )

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                learning_rate=3e-4,  # 默认 PPO 学习率
                n_steps=2048,  # 训练步数（较大值提高样本利用率）
                batch_size=64,  # 每次更新的批量大小
                n_epochs=10,  # 每个 batch 训练 10 次
                gamma=0.99,  # 折扣因子
                gae_lambda=0.95,  # GAE
                clip_range=0.2,  # PPO 剪辑范围
                ent_coef=0.01,  # 策略熵正则化
                verbose=1,  # 显示训练进度
                tensorboard_log="./ppo_air_combat_tb/",  # 记录 TensorBoard 日志
                device="cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU 加速

    # 训练模型
    model.learn(total_timesteps=1_000_000)  # 训练 100 万步

    # 保存训练好的 PPO 模型
    model.save("ppo_air_combat")

    # 关闭环境
    env.close()
