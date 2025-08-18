import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from typing import List, Dict, Optional
import logging
import argparse

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from env_factory.env_factory_selfplay import make_env, make_normal_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="1v1/ShootMissile/HierarchySelfPlayShoot")
    parser.add_argument("--target_state", type=int, default=0)

    # 基本路径
    parser.add_argument("--log_file", type=str, default="./train/result/train_shoot_static3.log")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--pretrained_pt_path", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="./trained_model/shoot_static3/checkpoints/")
    parser.add_argument("--tb_log", type=str, default="./ppo_air_combat_sp_tb/")
    parser.add_argument("--save_model_path", type=str, default="./trained_model/shoot_static3")
    parser.add_argument("--model_dir", type=str, default="./model_pool/shoot_static")

    # 模型路径
    parser.add_argument("--fly_model_path", type=str, default="trained_model/shoot_back_t2/ppo_air_combat.zip")
    parser.add_argument("--fire_model_path", type=str, default="./trained_model/shoot_solo5/ppo_air_combat.zip")
    parser.add_argument("--guide_model_path", type=str, default="trained_model/guide/ppo_air_combat.zip")
    parser.add_argument("--dodge_model_path", type=str, default="trained_model/dodge_missile/ppo_air_combat_dodge4.zip")

    # 环境参数
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--raw_obs_dim", type=int, default=21)
    parser.add_argument("--fly_act_dim", type=int, default=3)
    parser.add_argument("--fire_act_dim", type=int, default=2)
    parser.add_argument("--warmup_action", nargs='+', type=float, default=[1, 2, 1, 0.0, 0.0])

    # 多线程
    parser.add_argument("--num_envs", type=int, default=16)

    # 训练参数
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--save_interval", type=int, default=10_000)
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


def main_self_play():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # 初始化单个环境（封装成 DummyVecEnv，防止SB3复制问题）
    env_fns = [lambda env_id=i: make_env(env_id, args) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)

    # 初始模型保存路径
    initial_model_path = os.path.join(args.model_dir, "model_step_0.zip")

    # 初始化 ego agent
    if os.path.exists(args.model_path):
        print("✅ 加载已有模型继续训练...")
        ego_agent = PPO.load(
            args.model_path,
            env=env,
            # policy_kwargs=policy_kwargs,
            tensorboard_log=args.tb_log,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        print("🆕 没有旧模型，创建新 PPO 模型")
        ego_agent = PPO(
            "MlpPolicy",
            env,
            # policy_kwargs=policy_kwargs,
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

    ego_agent.save(initial_model_path)

    # 加载静态 opponent agent
    # opponent_agent = PPO.load(initial_model_path)

    env.env_method("set_opponent_agent", initial_model_path)
    # for i in range(args.num_envs):
    #     agent = load_opponent_for_env(i)
    #     env.env_method("set_opponent_agent", agent, indices=i)

    # 训练与保存
    total_timesteps = args.total_timesteps
    save_interval = args.save_interval

    # 保存历史模型路径列表
    saved_models = [initial_model_path]

    for step in range(0, total_timesteps, save_interval):
        # 学习
        ego_agent.learn(total_timesteps=save_interval, reset_num_timesteps=False)

        # 保存当前模型
        model_path = os.path.join(args.model_dir, f"model_step_{step + save_interval}.zip")
        ego_agent.save(model_path)
        saved_models.append(model_path)
        logging.critical(f"[INFO] Saved ego model to {model_path}")
        print(f"[INFO] Saved ego model to {model_path}")

        # === 自博弈更新：从已有模型中随机选择一个作为对手 ===
        opponent_path = random.choice(saved_models)
        # opponent_agent = PPO.load(opponent_path)
        env.env_method("set_opponent_agent", opponent_path)
        logging.critical(f"[INFO] Loaded opponent model from {opponent_path}")
        print(f"[INFO] Loaded opponent model from {opponent_path}")


def main_shoot_static():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # 创建多环境
    env_fns = [lambda env_id=i: make_normal_env(env_id, args) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)

    # 创建 Checkpoint 回调：每隔 save_interval 步保存一次模型
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_interval,
        save_path=args.checkpoint_path,
        name_prefix="ppo_model"
    )

    # ✅ 自定义网络结构：更深更宽的 MLP
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
        activation_fn=nn.ReLU
    )

    # 加载或新建模型
    if os.path.exists(args.model_path):
        print("✅ 加载已有模型继续训练...")
        model = PPO.load(
            args.model_path,
            env=env,
            tensorboard_log=args.tb_log,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        print("🆕 没有旧模型，创建新 PPO 模型")
        model = PPO(
            "MlpPolicy",
            env,
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
            device="cuda" if torch.cuda.is_available() else "cpu",
            policy_kwargs=policy_kwargs  # ✅ 设置更大网络
        )

    # 训练一次性完成 + 自动保存
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="ppo_run"
    )

    # 最后再保存一次模型
    final_model_path = os.path.join(args.save_model_path, "final_model.zip")
    model.save(final_model_path)
    print(f"✅ 最终模型已保存到 {final_model_path}")



if __name__ == "__main__":
    main_shoot_static()








