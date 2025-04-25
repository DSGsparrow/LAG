import os
import json
import random
import numpy as np
from stable_baselines3 import PPO
from typing import List, Dict, Optional
import logging
import argparse

from stable_baselines3.common.vec_env import SubprocVecEnv

from self_play_utils.strategy_pool_manager import StrategyPoolManager
from adapter.adapter_shoot_back_t import ShootControlWrapper

from LAGmaster.envs.JSBSim.envs import SingleCombatEnvShootBack


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
    parser.add_argument("--config", type=str, default="1v1/ShootMissile/HierarchyVsBaselineShootBack")

    # 基本路径
    parser.add_argument("--log_file", type=str, default="./train/result/train_shoot_back_t2.log")
    parser.add_argument("--model_path", type=str, default="trained_model/shoot_back_t/ppo_air_combat.zip")
    parser.add_argument("--pretrained_pt_path", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="./trained_model/shoot_back_t2/shoot_solo_checkpoints/")
    parser.add_argument("--tb_log", type=str, default="./ppo_air_combat_tb/")
    parser.add_argument("--save_model_path", type=str, default="./trained_model/shoot_back_t2/ppo_air_combat")

    # 模型路径
    parser.add_argument("--fly_model_path", type=str, default="trained_model/shoot_back/ppo_air_combat.zip")
    parser.add_argument("--fire_model_path", type=str, default="./trained_model/shoot_solo5/ppo_air_combat.zip")
    parser.add_argument("--guide_model_path", type=str, default="trained_model/guide/ppo_air_combat.zip")

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


def make_env(env_id, args):
    setup_logging(env_id, args.log_file)
    return ShootControlWrapper(lambda: SingleCombatEnvShootBack(config_name=args.config, env_id=env_id), args)


class BestResponseTrainer:
    def __init__(self, args, total_timesteps=100_000):
        self.env_fns = [lambda env_id=i: make_env(env_id, args) for i in range(args.num_envs)]
        self.env = SubprocVecEnv(self.env_fns)
        self.total_timesteps = total_timesteps

    def train_best_response(self, opponent_policy: PPO, initial_model_path: Optional[str] = None):
        import gymnasium as gym
        from stable_baselines3.common.env_util import make_vec_env

        class SelfPlayEnv(gym.Env):
            def __init__(self, base_env_fn, opponent_policy):
                super().__init__()
                self.env = base_env_fn()
                self.opponent = opponent_policy
                self.observation_space = self.env.observation_space
                self.action_space = self.env.action_space

            def reset(self, seed=None, options=None):
                obs, info = self.env.reset()
                return obs[0], info

            def step(self, action):
                obs = self.env._get_obs()
                opp_action, _ = self.opponent.predict(obs[1], deterministic=True)
                joint_action = [action, opp_action]
                obs, reward, terminated, truncated, info = self.env.step(joint_action)
                return obs[0], reward[0], terminated[0], truncated[0], info

        env = SelfPlayEnv(self.env_fn, opponent_policy)
        if initial_model_path and os.path.exists(initial_model_path):
            model = PPO.load(initial_model_path, env=env)
        else:
            model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=self.total_timesteps)
        return model


if __name__ == "__main__":
    pool = StrategyPoolManager(pool_dir="policy_pool")

    args = parse_args()
    trainer = BestResponseTrainer(args=args, total_timesteps=100_000)

    save_interval = 5  # 每隔多少轮进行一次全矩阵评估
    for round_id in range(20):
        print(f"[PSRO] Round {round_id}")
        opponent, opp_name = pool.sample_opponent(mode="softmax")
        initial_model_path = args.model_path if round_id == 0 else os.path.join("policy_pool", f"policy_{round_id-1}.zip")
        best_response = trainer.train_best_response(opponent_policy=opponent, initial_model_path=initial_model_path)
        pool.add_policy(best_response, step=round_id)
        new_name = f"policy_{round_id}.zip"
        pool.evaluate_all_vs_all(env_fn=make_env, n_episodes=3, full_matrix=(round_id % save_interval == 0))
        matrix_data = np.load("policy_pool/winrate_matrix.npy", allow_pickle=True).item()
        matrix = matrix_data["matrix"]
        names = matrix_data["names"]
        i = names.index(new_name)
        j = names.index(opp_name)
        winrate = matrix[i, j]
        pool.update_elo(winner_name=new_name, loser_name=opp_name, score_winner=winrate)
        print(f"[PSRO] Added new strategy for round {round_id} and updated Elo\n")
