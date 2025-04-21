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
