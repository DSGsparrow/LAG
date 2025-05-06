import os
import json
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from typing import List, Dict, Optional
import logging
import argparse

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from self_play_utils.strategy_pool_manager import StrategyPoolManager
from adapter.adapter_shoot_self_play import ShootSelfPlayWrapper

from LAGmaster.envs.JSBSim.envs import SingleCombatEnvShootSelfPlay

from net import CustomTransformerExtractor


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
    parser.add_argument("--config", type=str, default="1v1/ShootMissile/HierarchySelfPlayShoot")

    # 基本路径
    parser.add_argument("--log_file", type=str, default="./train/result/train_shoot_selfplay.log")
    parser.add_argument("--model_path", type=str, default="trained_model/shoot_solo5/ppo_air_combat.zip")
    parser.add_argument("--pretrained_pt_path", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="./trained_model/shoot_selfplay/shoot_solo_checkpoints/")
    parser.add_argument("--tb_log", type=str, default="./ppo_air_combat_tb/")
    parser.add_argument("--save_model_path", type=str, default="./trained_model/shoot_selfplay/ppo_air_combat")

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
    parser.add_argument("--num_envs", type=int, default=1)

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


def make_env(env_id, opponent, args):
    setup_logging(env_id, args.log_file)
    return ShootSelfPlayWrapper(lambda: SingleCombatEnvShootSelfPlay(config_name=args.config, env_id=env_id), opponent, args)


class BestResponseTrainer:
    def __init__(self, args, total_timesteps=100_000):
        self.args = args
        self.total_timesteps = total_timesteps

    def train_best_response(self, opponent_policy: PPO, initial_model_path: Optional[str] = None):
        self.env_fns = [lambda env_id=i: make_env(env_id, opponent_policy, self.args) for i in range(self.args.num_envs)]
        self.env = SubprocVecEnv(self.env_fns)

        if initial_model_path and os.path.exists(initial_model_path):
            model = PPO.load(initial_model_path, env=self.env)
        else:
            model = PPO("MlpPolicy", self.env, verbose=1)
        model.learn(total_timesteps=self.total_timesteps)
        return model


def main():
    args = parse_args()

    opponent = PPO.load(args.fire_model_path)

    env_fns = [lambda env_id=i: make_env(env_id, opponent, args) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=CustomTransformerExtractor,
        features_extractor_kwargs=dict(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            seq_len=args.history_len,
            input_dim=args.raw_obs_dim + args.fly_act_dim + args.fire_act_dim,
        ),
        net_arch=[],
    )

    if os.path.exists(args.model_path):
        print("✅ 加载已有模型继续训练...")
        model = PPO.load(
            args.model_path,
            env=env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.tb_log,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
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

        if os.path.exists(args.pretrained_pt_path):
            print("🔄 加载模仿学习预训练参数...")
            policy = model.policy
            pretrained = torch.load(args.pretrained_pt_path, map_location="cpu")

            mlp_state = {k.replace("feature_extractor.", ""): v for k, v in pretrained.items() if k.startswith("feature_extractor.")}
            policy.features_extractor.mlp.load_state_dict(mlp_state)

            gru_state = {k.replace("gru.", ""): v for k, v in pretrained.items() if k.startswith("gru.")}
            policy.features_extractor.gru.gru.load_state_dict(gru_state)

            act_state = {k.replace("action_head.", ""): v for k, v in pretrained.items() if k.startswith("action_head.")}
            policy.action_net.load_state_dict(act_state)

            print("✅ 模仿学习参数加载成功！")

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.checkpoint_path,
        name_prefix="ppo_air_combat_shoot"
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name="shoot_back_t2",
        callback=checkpoint_callback
    )

    # 最终训练完成后保存一次完整模型
    model.save(args.save_model_path)

if __name__ == "__main__":
    main()
    # pool = StrategyPoolManager(pool_dir="policy_pool")
    #
    # args = parse_args()
    # trainer = BestResponseTrainer(args=args, total_timesteps=100_000)
    #
    # save_interval = 5  # 每隔多少轮进行一次全矩阵评估
    # for round_id in range(20):
    #     print(f"[PSRO] Round {round_id}")
    #     opponent, opp_name = pool.sample_opponent(mode="softmax")
    #     initial_model_path = args.model_path if round_id == 0 else os.path.join("policy_pool", f"policy_{round_id-1}.zip")
    #     best_response = trainer.train_best_response(opponent_policy=opponent, initial_model_path=initial_model_path)
    #     pool.add_policy(best_response, step=round_id)
    #     new_name = f"policy_{round_id}.zip"
    #     pool.evaluate_all_vs_all(env_fn=make_env, n_episodes=3, full_matrix=(round_id % save_interval == 0))
    #     matrix_data = np.load("policy_pool/winrate_matrix.npy", allow_pickle=True).item()
    #     matrix = matrix_data["matrix"]
    #     names = matrix_data["names"]
    #     i = names.index(new_name)
    #     j = names.index(opp_name)
    #     winrate = matrix[i, j]
    #     pool.update_elo(winner_name=new_name, loser_name=opp_name, score_winner=winrate)
    #     print(f"[PSRO] Added new strategy for round {round_id} and updated Elo\n")
