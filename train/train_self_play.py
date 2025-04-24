import os
import json
import random
import numpy as np
from stable_baselines3 import PPO
from typing import List, Dict, Optional

class StrategyPoolManager:
    def __init__(self, pool_dir="policy_pool", max_size=20):
        self.pool_dir = pool_dir
        self.meta_path = os.path.join(pool_dir, "meta.json")
        self.matrix_path = os.path.join(pool_dir, "winrate_matrix.npy")
        os.makedirs(pool_dir, exist_ok=True)
        self.max_size = max_size
        self.policies: Dict[str, Dict] = self._load_meta()
        self.k_factor = 32

    def _load_meta(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                return json.load(f)
        return {}

    def _save_meta(self):
        with open(self.meta_path, "w") as f:
            json.dump(self.policies, f, indent=2)

    def _save_matrix(self, matrix, names):
        np.save(self.matrix_path, {"matrix": matrix, "names": names})

    def _get_policy_list(self):
        return sorted(self.policies.items(), key=lambda x: x[1]["step"])

    def add_policy(self, policy: PPO, step: int, meta: Optional[Dict] = None):
        name = f"policy_{step}.zip"
        path = os.path.join(self.pool_dir, name)
        policy.save(path)
        self.policies[name] = meta or {"elo": 1000, "step": step}
        self._save_meta()
        self._cleanup()

    def _cleanup(self):
        if len(self.policies) > self.max_size:
            sorted_by_step = sorted(self.policies.items(), key=lambda x: x[1]["step"])
            to_remove, *_ = sorted_by_step
            os.remove(os.path.join(self.pool_dir, to_remove[0]))
            del self.policies[to_remove[0]]
            self._save_meta()

    def sample_opponent(self, mode="uniform", current_elo=1000, best_response_target=None):
        names = list(self.policies.keys())
        elos = np.array([self.policies[name]["elo"] for name in names])

        if mode == "uniform":
            probs = np.ones(len(names)) / len(names)
        elif mode == "elo_diff":
            diffs = np.abs(elos - current_elo)
            probs = 1.0 / (diffs + 1e-5)
            probs /= probs.sum()
        elif mode == "softmax":
            logits = elos / 100.0
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
        elif mode == "psro_best_response":
            if not os.path.exists(self.matrix_path):
                raise ValueError("Winrate matrix not found. Run evaluate_all_vs_all first.")
            matrix_data = np.load(self.matrix_path, allow_pickle=True).item()
            matrix = matrix_data["matrix"]
            matrix_names = matrix_data["names"]
            if best_response_target is None or best_response_target not in matrix_names:
                raise ValueError("Must provide valid best_response_target for psro_best_response.")
            col_index = matrix_names.index(best_response_target)
            probs = matrix[:, col_index]
            probs[col_index] = 0  # 不选自己
            probs = probs / probs.sum()
        else:
            raise ValueError("Unknown sampling mode")

        idx = np.random.choice(len(names), p=probs)
        return PPO.load(os.path.join(self.pool_dir, names[idx])), names[idx]

    def update_elo(self, winner_name: str, loser_name: str, score_winner: float = 1.0):
                ra = self.policies[winner_name]["elo"]
        rb = self.policies[loser_name]["elo"]
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 - ea
        sa = score_winner
        sb = 1 - sa
        self.policies[winner_name]["elo"] = ra + self.k_factor * (sa - ea)
        self.policies[loser_name]["elo"] = rb + self.k_factor * (sb - eb)
        self._save_meta()

    def evaluate_all_vs_all(self, env_fn, n_episodes=5, full_matrix=True):
        policies = self._get_policy_list()
        n = len(policies)
        matrix = np.zeros((n, n))
        names = [name for name, _ in policies]

        def play_match(p1: PPO, p2: PPO):
            from gymnasium import Env
            class MatchEnv(Env):
                def __init__(self):
                    self.env = env_fn()
                    self.observation_space = self.env.observation_space
                    self.action_space = self.env.action_space

                def reset(self, seed=None, options=None):
                    obs, info = self.env.reset()
                    return obs, info

                def step(self, action_pair):
                    obs = self.env._get_obs()
                    a1, _ = p1.predict(obs[0], deterministic=True)
                    a2, _ = p2.predict(obs[1], deterministic=True)
                    obs, reward, terminated, truncated, info = self.env.step([a1, a2])
                    return obs, reward, terminated, truncated, info

            win_count = 0
            for _ in range(n_episodes):
                match_env = MatchEnv()
                obs, _ = match_env.reset()
                done = False
                while not done:
                    obs, reward, term, trunc, _ = match_env.step(None)
                    done = term[0] or trunc[0]
                if reward[0] > reward[1]:
                    win_count += 1
            return win_count / n_episodes

        for i in range(n):
            for j in range(n):
                if i != j and (full_matrix or i == n - 1 or j == n - 1):
                    p1 = PPO.load(os.path.join(self.pool_dir, names[i]))
                    p2 = PPO.load(os.path.join(self.pool_dir, names[j]))
                    winrate = play_match(p1, p2)
                    matrix[i, j] = winrate

        self._save_matrix(matrix, names)


class BestResponseTrainer:
    def __init__(self, env_fn, total_timesteps=100_000):
        self.env_fn = env_fn
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
    def make_env():
    import argparse
    from your_env_module import ShootControlWrapper, SB3SingleCombatEnv  # 替换为你真实的模块路径

    def setup_logging(env_id, log_file):
        import logging
        logging.basicConfig(filename=log_file, level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="1v1/ShootMissile/HierarchyVsBaselineShootBack")
    parser.add_argument("--log_file", type=str, default="./train/result/train_shoot_back_t2.log")
    parser.add_argument("--model_path", type=str, default="trained_model/shoot_back_t/ppo_air_combat.zip")
    parser.add_argument("--pretrained_pt_path", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="./trained_model/shoot_back_t2/shoot_solo_checkpoints/")
    parser.add_argument("--tb_log", type=str, default="./ppo_air_combat_tb/")
    parser.add_argument("--save_model_path", type=str, default="./trained_model/shoot_back_t2/ppo_air_combat")
    parser.add_argument("--fly_model_path", type=str, default="trained_model/shoot_back/ppo_air_combat.zip")
    parser.add_argument("--fire_model_path", type=str, default="./trained_model/shoot_solo5/ppo_air_combat.zip")
    parser.add_argument("--guide_model_path", type=str, default="trained_model/guide/ppo_air_combat.zip")
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--raw_obs_dim", type=int, default=21)
    parser.add_argument("--fly_act_dim", type=int, default=3)
    parser.add_argument("--fire_act_dim", type=int, default=2)
    parser.add_argument("--warmup_action", nargs='+', type=float, default=[1, 2, 1, 0.0, 0.0])
    parser.add_argument("--num_envs", type=int, default=1)
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
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    args, _ = parser.parse_known_args()
    setup_logging(env_id=0, log_file=args.log_file)
    return ShootControlWrapper(lambda: SB3SingleCombatEnv(env_id=0, config_name=args.config), args)

    pool = StrategyPoolManager(pool_dir="policy_pool")
    trainer = BestResponseTrainer(env_fn=make_env, total_timesteps=100_000)

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
