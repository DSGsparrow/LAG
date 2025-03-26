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

from net.net_shoot_missile import MLPBase, GRULayer, ACTLayer, CustomPolicy
from adapter.adapter_shoot_missile import SB3SingleCombatEnv

from LAGmaster.envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, SingleCombatEnvTest


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
    num_envs = 1  # 设定 8 个并行环境（根据 GPU 性能调整）

    log_file = "./train/result/train_dodge.log"

    # 创建并行环境
    def make_env(env_id):
        setup_logging(log_file)
        return SB3SingleCombatEnv(env_id, config_name='1v1/DodgeMissile/HierarchyVsBaselineSelf')


    # env = SubprocVecEnv([lambda: make_env(env_id) for env_id in range(num_envs)])
    env = SubprocVecEnv([lambda env_id=env_id: make_env(env_id) for env_id in range(num_envs)])

    # 定义 PPO 模型（自定义 MLP 作为特征提取器）
    policy_kwargs = dict(
        features_extractor_class=CustomPolicy,
        features_extractor_kwargs=dict(action_dim=env.action_space)
    )

    # 模型路径
    model_path = "./trained_model/dodge_missile/ppo_air_combat_dodge.zip"

    if os.path.exists(model_path):
        print("✅ 加载已有模型继续训练...")
        model = PPO.load(
            model_path,
            env=env,
            tensorboard_log="./ppo_air_combat_tb/",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        print("🆕 没有旧模型，重新训练一个新的 PPO 模型")
        # 创建 PPO 模型
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

    # 创建 checkpoint 回调，每 10 万步保存一次
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,  # 每 1*num_env 万步保存一次
        save_path="./trained_model/dodge_missile_checkpoints/",  # 保存文件夹
        name_prefix="ppo_air_combat_dodge"  # 文件名前缀
    )

    # 开始训练，同时记录 TensorBoard 和保存中间模型
    model.learn(
        total_timesteps=3_000_000,
        tb_log_name="test",
        callback=checkpoint_callback
    )

    # 最终训练完成后保存一次完整模型
    model.save("./trained_model/dodge_missile/ppo_air_combat_dodge")

    # 关闭环境
    env.close()
