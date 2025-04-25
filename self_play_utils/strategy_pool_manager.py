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

                # 胜负判断还要修改？
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