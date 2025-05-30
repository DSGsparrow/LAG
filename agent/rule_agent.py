import numpy as np
import logging
import torch
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from argparse import Namespace

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymObs
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv

from adapter.adapter_shoot_self_play import SelfPlayDodgeWrapper
from LAGmaster.envs.JSBSim.envs.singlecombat_env_shoot_selfplay import SingleCombatEnvShootSelfPlay


class RuleBasedCombatAgent(BasePolicy):
    def __init__(self, observation_space, action_space, args: Namespace, **kwargs):
        super().__init__(observation_space, action_space)

        self.counter_agent = PPO.load(args.counter_model_path)
        self.dodge_agent = PPO.load(args.dodge_model_path)
        self.guide_agent = PPO.load(args.guide_model_path)
        self.fire_decision_agent = PPO.load(args.fire_decision_model_path)

        self.history_len = args.history_len
        self.obs_dim = args.raw_obs_dim
        self.act_dim = args.action_dim
        self.fly_act_dim = args.fly_act_dim
        self.fire_act_dim = args.fire_act_dim
        self.debug = args.debug
        self.initial_missile_num = args.missile_num

        self.env_state = {}

    def reset(self, env_idx):
        self.env_state[env_idx] = {
            "remaining_missiles": self.initial_missile_num,
            "opponent_missile_in_air": False,
            "ego_has_fired": False
        }

    def normalize_fire_action(self, action, temperature=0.5, threshold=0.3):
        logits = torch.tensor([action[0], action[1]])
        probs = F.softmax(logits / temperature, dim=0)
        act_prob = probs[0].item()
        do_act = act_prob > threshold
        return 1.0 if do_act else 0.0

    def _enemy_has_fired(self, obs):
        return not np.allclose(obs[-5:], 0.0)

    def _update_enemy_missile_flag(self, state, obs):
        state["opponent_missile_in_air"] = self._enemy_has_fired(obs)

    def _predict(self, observation, state=None, episode_start=None, deterministic=True):
        if observation.ndim == 1:
            observation = observation[None, :]

        actions = []

        for env_idx, obs_seq in enumerate(observation):
            # todo 还是有问题，这个现在用法和多环境冲突了，还是得再看一下
            if state is not None and isinstance(state, (list, tuple)) and state[env_idx] == True:
                self.reset(env_idx)
            elif env_idx not in self.env_state:
                self.reset(env_idx)

            s = self.env_state[env_idx]

            obs_latest = obs_seq[:self.obs_dim]
            self._update_enemy_missile_flag(s, obs_latest)

            # fire decision uses sequence directly
            fire_decision = 0.0
            if len(obs_seq) == self.history_len * (self.obs_dim + self.act_dim):
                fire_logits, _ = self.fire_decision_agent.predict(obs_seq, deterministic=True)
                fire_decision = self.normalize_fire_action(fire_logits)

            # determine stage
            if s["ego_has_fired"]:
                stage = "GUIDE"
            elif s["opponent_missile_in_air"]:
                stage = "DODGE"
            elif s["remaining_missiles"] == 0:
                stage = "GUIDE"
            else:
                stage = "COUNTER"

            # select model input
            if stage == "COUNTER":
                act, _ = self.counter_agent.predict(obs_seq, deterministic=True)
            elif stage == "DODGE":
                act, _ = self.dodge_agent.predict(obs_latest, deterministic=True)
            elif stage == "GUIDE":
                act, _ = self.guide_agent.predict(obs_latest, deterministic=True)
            else:
                act = np.zeros(self.fly_act_dim)

            # combine action
            full_action = np.zeros(self.fly_act_dim + self.fire_act_dim)
            full_action[:self.fly_act_dim] = act[:self.fly_act_dim]
            full_action[self.fly_act_dim:] = fire_logits

            # update missile state
            if fire_decision > 0.3 and s["remaining_missiles"] > 0:
                s["ego_has_fired"] = True
                s["remaining_missiles"] -= 1

            actions.append(full_action)

        return np.array([actions]), None  # 再包一层是环境的



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


def make_env(env_id, args):
    setup_logging(env_id, args.log_file)
    return SelfPlayDodgeWrapper(lambda: SingleCombatEnvShootSelfPlay(config_name=args.config, env_id=env_id), args)


if __name__ == "__main__":
    args = Namespace(
        shoot_model_path="trained_model/shoot_imitation/ppo_air_combat_imi.zip",
        guide_model_path="trained_model/guide/ppo_air_combat.zip",
        dodge_model_path="trained_model/dodge_missile/ppo_air_combat_dodge4.zip",
        counter_model_path="trained_model/shoot_back_t2/ppo_air_combat.zip",
        fire_decision_model_path="trained_model/shoot_solo5/ppo_air_combat.zip",
        log_file="selfplay.log",
        history_len=10,
        raw_obs_dim=21,
        action_dim=5,
        fly_act_dim=3,
        fire_act_dim=2,
        missile_num=1,
        agent_id=0,
        debug=True,
        num_envs=1,
        config="1v1/ShootMissile/HierarchySelfplayShoot",
        max_steps=1000000,
        warmup_action=[1, 2, 1, 0, 0]
    )

    env_fns = [lambda env_id=i: make_env(env_id, args) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)

    agent = RuleBasedCombatAgent(observation_space=env.observation_space, action_space=env.action_space, args=args)
    # agent1 = RuleBasedCombatAgent(observation_space=env.observation_space, action_space=env.action_space, args=args)

    obs = env.reset()
    dones = [False, False]
    for step in range(args.max_steps):
        action, _ = agent._predict(obs[0], state=dones, deterministic=True)
        # action1, _ = agent1._predict(obs[1], deterministic=True)

        obs, rewards, dones, infos = env.step(action)

        if dones[0]:
            print('agent0 rewards', env.get_attr('total_rewards')[0][0],
                  'agent1 rewards', env.get_attr('total_rewards')[0][1],)

    env.close()
    print("🎯 推理完成")






