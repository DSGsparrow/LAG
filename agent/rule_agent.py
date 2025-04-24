import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymObs
from stable_baselines3.common.utils import obs_as_tensor


class RuleBasedCombatAgent(BasePolicy):
    def __init__(self, observation_space, action_space, args, **kwargs):
        super().__init__(observation_space, action_space)

        from stable_baselines3 import PPO
        self.shoot_agent = PPO.load(args.shoot_model_path)
        self.guide_agent = PPO.load(args.guide_model_path)
        self.dodge_agent = PPO.load(args.dodge_model_path)
        self.counter_agent = PPO.load(args.counter_model_path)
        self.fire_decision_agent = PPO.load(args.fire_decision_model_path)

        self.history_len = args.history_len
        self.obs_dim = args.obs_dim
        self.act_dim = args.act_dim
        self.fly_act_dim = args.fly_act_dim
        self.fire_act_dim = args.fire_act_dim
        self.agent_id = args.agent_id
        self.debug = args.debug
        self.initial_missile_num = args.missile_num

        # 并行环境支持（每个环境一个状态）
        self.env_state = {}  # key: env index, value: dict with histories & flags

    def reset(self, env_idx):
        self.env_state[env_idx] = {
            "ego_has_fired": False,
            "first_strike_done": False,
            "remaining_missiles": self.initial_missile_num,
            "obs_history": deque(maxlen=self.history_len),
            "act_history": deque(maxlen=self.history_len),
        }

    def _normalize_action(self, action_3d):
        return np.clip(action_3d / 5.0, -1.0, 1.0)

    def _enemy_has_fired(self, obs):
        return not np.allclose(obs[15:21], 0.0)

    def _construct_fire_decision_obs(self, state):
        if len(state["obs_history"]) < self.history_len:
            return None
        seq = [np.concatenate([o, a], axis=0) for o, a in zip(state["obs_history"], state["act_history"])]
        return np.concatenate(seq, axis=0).astype(np.float32)

    def _determine_stage(self, obs, state):
        enemy_fired = self._enemy_has_fired(obs)
        if enemy_fired:
            return "DODGE"
        if state["ego_has_fired"] and state["remaining_missiles"] > 0:
            return "COUNTER"
        if state["ego_has_fired"]:
            return "GUIDE"
        if state["first_strike_done"] and state["remaining_missiles"] > 0:
            return "COUNTER"
        if state["remaining_missiles"] == 0:
            return "DODGE"
        return "ENGAGE"

    def predict(self, observation: GymObs, state=None, episode_start=None, deterministic=True):
        if observation.ndim == 1:
            observation = observation[None, :]

        actions = []
        for env_idx, obs in enumerate(observation):
            if env_idx not in self.env_state:
                self.reset(env_idx)
            s = self.env_state[env_idx]

            stage = self._determine_stage(obs, s)
            fire_decision_obs = self._construct_fire_decision_obs(s)
            fire_decision = False
            fire_prob = 0.0

            if fire_decision_obs is not None:
                score = self.fire_decision_agent.predict(fire_decision_obs, deterministic=True)[0]
                logits = torch.tensor([score[3], score[4]])
                probs = F.softmax(logits / 0.5, dim=0)
                fire_prob = probs[0].item()
                fire_decision = fire_prob > 0.5

            if self.debug:
                print(f"[Agent {self.agent_id} | Env {env_idx}] Stage: {stage} | FireProb: {fire_prob:.3f} | Missiles Left: {s['remaining_missiles']}")

            if stage == "ENGAGE":
                shoot_action = self.shoot_agent.predict(obs, deterministic=True)[0]
                action = shoot_action.copy()
                should_fire = (
                    (shoot_action[4] > 0.5 or fire_decision)
                    and not s["first_strike_done"] and s["remaining_missiles"] > 0
                )
                if should_fire:
                    s["first_strike_done"] = True
                    s["ego_has_fired"] = True
                    s["remaining_missiles"] -= 1
                action[4] = 1.0 if should_fire else 0.0

            elif stage == "GUIDE":
                raw = self.guide_agent.predict(obs, deterministic=True)[0]
                action = np.concatenate([self._normalize_action(raw), [0.0]])

            elif stage == "DODGE":
                raw = self.dodge_agent.predict(obs, deterministic=True)[0]
                action = np.concatenate([self._normalize_action(raw), [0.0]])

            elif stage == "COUNTER":
                raw = self.counter_agent.predict(obs, deterministic=True)[0]
                action = np.concatenate([self._normalize_action(raw), [0.0]])

            else:
                action = np.zeros(5)

            if stage != "ENGAGE" and fire_decision and s["remaining_missiles"] > 0:
                action[4] = 1.0
                s["ego_has_fired"] = True
                s["remaining_missiles"] -= 1

            s["obs_history"].append(obs)
            s["act_history"].append(action)

            actions.append(action)

        return np.array(actions), None


from argparse import Namespace

args = Namespace(
    shoot_model_path="trained_model/shoot_imitation/ppo_air_combat_imi.zip",
    guide_model_path="trained_model/guide/ppo_air_combat.zip",
    dodge_model_path="trained_model/dodge_missile/ppo_air_combat_dodge4.zip",
    counter_model_path="trained_model/shoot_back/ppo_air_combat.zip",
    fire_decision_model_path="trained_model/shoot_solo5/ppo_air_combat.zip",
    history_len=5,
    obs_dim=21,
    act_dim=5,
    fly_act_dim=4,
    fire_act_dim=1,
    missile_num=2,
    agent_id=0,
    debug=True
)

agent = RuleBasedCombatAgent(observation_space=..., action_space=..., args=args)








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

    def sample_opponent(self, mode="uniform", current_elo=1000):
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
        else:
            raise ValueError("Unknown sampling mode")

        idx = np.random.choice(len(names), p=probs)
        return PPO.load(os.path.join(self.pool_dir, names[idx])), names[idx]

    def update_elo(self, winner_name: str, loser_name: str):
        ra = self.policies[winner_name]["elo"]
        rb = self.policies[loser_name]["elo"]
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 - ea
        self.policies[winner_name]["elo"] = ra + self.k_factor * (1 - ea)
        self.policies[loser_name]["elo"] = rb + self.k_factor * (0 - eb)
        self._save_meta()


class BestResponseTrainer:
    def __init__(self, env_fn, total_timesteps=100_000):
        self.env_fn = env_fn
        self.total_timesteps = total_timesteps

    def train_best_response(self, opponent_policy: PPO):
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
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=self.total_timesteps)
        return model


if __name__ == "__main__":
    def make_env():
        import your_custom_gym_env  # 替换为你的环境导入
        return your_custom_gym_env.create_env()  # 替换为你的环境构造函数

    pool = StrategyPoolManager(pool_dir="policy_pool")
    trainer = BestResponseTrainer(env_fn=make_env, total_timesteps=100_000)

    for round_id in range(20):
        print(f"[PSRO] Round {round_id}")
        opponent, opp_name = pool.sample_opponent(mode="softmax")
        best_response = trainer.train_best_response(opponent_policy=opponent)
        pool.add_policy(best_response, step=round_id)
        new_name = f"policy_{round_id}.zip"
        pool.update_elo(winner_name=new_name, loser_name=opp_name)
        print(f"[PSRO] Added new strategy for round {round_id} and updated Elo\n")

