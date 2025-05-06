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

from adapter.adapter_dodge_missile import SB3SingleCombatEnv, DodgeControlWrapper
from net import (CustomImitationShootBackPolicy, CustomActorCriticShootBackPolicy,
                 CustomImitationPolicy, CustomTransformerExtractor)

from adapter.adapter_shoot_back_t import ShootControlWrapper


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config1", type=str, default="1v1/DodgeMissile/HierarchyKnown")
    parser.add_argument("--config2", type=str, default="1v1/DodgeMissile/HierarchyUnknown")

    # 基本路径
    # first step
    parser.add_argument("--log_file", type=str, default="./train/result/train_dodge_2step_1.log")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--pretrained_pt_path", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="./trained_model/dodge_2step_1/checkpoints/")
    parser.add_argument("--tb_log", type=str, default="./ppo_air_combat_tb/")
    parser.add_argument("--save_model_path", type=str, default="./trained_model/dodge_2step_1/ppo_air_combat")

    # second step
    parser.add_argument("--log_file2", type=str, default="./train/result/train_dodge_2step_2.log")
    parser.add_argument("--model_path2", type=str, default="./trained_model/dodge_2step_1/ppo_air_combat.zip")
    parser.add_argument("--pretrained_pt_path2", type=str, default="")
    parser.add_argument("--checkpoint_path2", type=str, default="./trained_model/dodge_2step_2/checkpoints/")
    parser.add_argument("--tb_log2", type=str, default="./ppo_air_combat_tb/")
    parser.add_argument("--save_model_path2", type=str, default="./trained_model/dodge_2step_2/ppo_air_combat")

    # 模型路径
    parser.add_argument("--fly_model_path", type=str, default="trained_model/shoot_back/ppo_air_combat.zip")
    parser.add_argument("--fire_model_path", type=str, default="./trained_model/shoot_solo5/ppo_air_combat.zip")
    parser.add_argument("--guide_model_path", type=str, default="trained_model/guide/ppo_air_combat.zip")

    # 环境参数
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--raw_obs_dim", type=int, default=22)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--warmup_action", nargs='+', type=float, default=[1, 2, 1])

    # 多线程
    parser.add_argument("--num_envs", type=int, default=16)

    # 训练参数
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--save_freq", type=int, default=4_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.02)

    # Transformer 网络参数
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser.parse_args()


def make_env(env_id, args, num):
    if num == 1:
        setup_logging(env_id, args.log_file)
        return DodgeControlWrapper(lambda: SB3SingleCombatEnv(env_id, config_name=args.config1), args)
    else:
        setup_logging(env_id, args.log_file2)
        return DodgeControlWrapper(lambda: SB3SingleCombatEnv(env_id, config_name=args.config2), args)


def main():
    args = parse_args()

    # first step missile known
    env_fns = [lambda env_id=i: make_env(env_id, args, 1) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=CustomTransformerExtractor,
        features_extractor_kwargs=dict(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            seq_len=args.history_len,
            input_dim=args.raw_obs_dim + args.action_dim,
        ),
        net_arch=[],
    )

    print("🆕 没有旧模型，创建新 PPO 模型")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=args.tb_log,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.checkpoint_path,
        name_prefix="ppo_air_combat_shoot"
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name="dodge_2step_1",
        callback=checkpoint_callback
    )

    # 最终训练完成后保存一次完整模型
    model.save(args.save_model_path)

    # second step missile unknown
    env_fns = [lambda env_id=i: make_env(env_id, args, 2) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=CustomTransformerExtractor,
        features_extractor_kwargs=dict(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            seq_len=args.history_len,
            input_dim=args.raw_obs_dim + args.action_dim,
        ),
        net_arch=[],
    )

    print("✅ 加载已有模型继续训练...")
    model = PPO.load(
        args.model_path2,
        env=env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.tb_log,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.checkpoint_path2,
        name_prefix="ppo_air_combat_shoot"
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name="dodge_2step_2",
        callback=checkpoint_callback
    )

    # 最终训练完成后保存一次完整模型
    model.save(args.save_model_path2)




if __name__ == "__main__":
    main()
