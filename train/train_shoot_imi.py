import gym
import gymnasium
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces
import argparse
import os
import logging

from LAGmaster.envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, SingleCombatEnvShoot
from net import CustomImitationPolicy

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
            # raise ValueError("Unexpected action space type: {}".format(type(self.env.action_space)))
            self.action_space = self.env.action_space

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


class EnvIDFilter(logging.Filter):
    def __init__(self, env_id):
        super().__init__()
        self.env_id = env_id

    def filter(self, record):
        record.env_id = f"{self.env_id}"
        return True


def setup_logging(env_id=0, log_file=None):
    """配置 logging，让日志既输出到终端，又写入文件，标明 ENV ID"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 创建 Filter，用于注入 env_id
    env_filter = EnvIDFilter(env_id)

    # 日志格式带 env_id
    formatter = logging.Formatter("%(asctime)s - %(levelname)s [ENV %(env_id)s] - %(message)s")

    # 终端 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(env_filter)

    # 文件 handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(env_filter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info(f"Logger for ENV {env_id} initialized, log path: {log_file}")


# ========== 6. 训练 PPO ==========
# =================== 训练主程序 ===================
if __name__ == "__main__":
    # 参数
    num_envs = 16
    log_file = "./train/result/train_shoot_imi2.log"
    model_path = ''  # "./trained_model/imitation_shoot/imitation_pretrained.zip"
    pretrained_pt_path = "./trained_model/imitation_shoot/imitation_pretrained_pytorch.pt"

    # 多进程环境创建
    def make_env(env_id):
        setup_logging(env_id, log_file)
        return SB3SingleCombatEnv(env_id, config_name='1v1/ShootMissile/HierarchyVsBaselineImitation')

    env = SubprocVecEnv([lambda env_id=i: make_env(env_id) for i in range(num_envs)])

    # 定义 PPO 模型（指定自定义特征提取器）
    policy_kwargs = dict(
        features_extractor_class=CustomImitationPolicy,
        features_extractor_kwargs={}
    )

    # 如果已有模型，加载继续训练；否则新建
    if os.path.exists(model_path):
        print("✅ 加载已有模型继续训练...")
        model = PPO.load(
            model_path,
            env=env,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./ppo_air_combat_tb/",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        print("🆕 没有旧模型，重新训练一个新的 PPO 模型")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            verbose=1,
            tensorboard_log="./ppo_air_combat_tb/",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 如果存在预训练 PyTorch 模型，加载其中部分参数
        if os.path.exists(pretrained_pt_path):
            print("🔄 正在加载模仿学习预训练参数...")
            pretrained_dict = torch.load(pretrained_pt_path, map_location="cpu")
            extractor = model.policy.features_extractor
            current_dict = extractor.state_dict()
            matched_dict = {k: v for k, v in pretrained_dict.items() if k in current_dict}
            current_dict.update(matched_dict)
            extractor.load_state_dict(current_dict)
            print("✅ 成功加载预训练特征提取器参数！")
            print("匹配的参数：", [k for k in pretrained_dict.keys() if k in current_dict])
            print("未匹配的参数：", [k for k in pretrained_dict.keys() if k not in current_dict])

    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./trained_model/shoot_missile_checkpoints_imi/",
        name_prefix="ppo_air_combat_shoot"
    )

    # 冻结特征提取器参数
    for param in model.policy.features_extractor.parameters():
        param.requires_grad = False

    # 重新设置优化器（只训练 requires_grad=True 的参数）
    model.policy.optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.policy.parameters()),
        lr=3e-4
    )

    # 训练阶段1：只训练线性层
    model.learn(
        total_timesteps=1_000_000,  # 可调整冻结期
        tb_log_name="frozen_feature",
        callback=checkpoint_callback
    )

    # 解冻所有参数
    for param in model.policy.features_extractor.parameters():
        param.requires_grad = True

    # 重新设置优化器（训练所有参数）
    model.policy.optimizer = torch.optim.Adam(
        model.policy.parameters(),
        lr=3e-4
    )

    # 训练阶段2：正常训练
    model.learn(
        total_timesteps=2_000_000,  # 剩余训练步数
        tb_log_name="full_train",
        callback=checkpoint_callback
    )

    # 保存最终模型
    model.save("./trained_model/shoot_imitation/ppo_air_combat_imi")
    env.close()
