import numpy as np
from gymnasium import spaces
from collections import deque
import logging
from typing import List, Tuple

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, PostureShootReward
from ..reward_functions import ShootPenaltyReward, ShootGapPenaltyReward, RelativeAltitudeReward, ShootEventDrivenReward
from ..reward_functions import AltitudeGuideReward, SpeedGuideReward
from ..reward_functions import (SelfPlayShootPenaltyReward, SelfPlayPostureReward, SelfPlayShootEventDrivenReward,
                                SelfPlayShootGapPenalty, SelfPlayShootPosturePenalty,
                                SelfPlayShootWaitReward, SelfPlayEnemyPostureReward, SelfPlayShootMissileRewardWithDistance)
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, DodgeMissileSafeReturn
from ..termination_conditions import SafeReturn, ShootSafeReturn
from ..reward_functions import EndAltitudeReward
from ..core.simulatior import MissileSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R
from .singlecombat_with_missle_task import SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, SingleCombatShootMissileTask
from utils.shoot_rule import fuzzy_should_attack

class SingleCombatShootMissileBackTask(SingleCombatDodgeMissileTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            PostureShootReward(self.config),
            AltitudeReward(self.config),
            ShootEventDrivenReward(self.config),
            ShootPenaltyReward(self.config),
            ShootGapPenaltyReward(self.config),
            RelativeAltitudeReward(self.config),
        ]

        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            ShootSafeReturn(self.config),
            Timeout(self.config),
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, shoot control
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.4, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 0.9, 1.0], dtype=np.float32),
            dtype=np.float32
        )
    
    def get_obs(self, env, agent_id):
        return super().get_obs(env, agent_id)
    
    def normalize_action(self, env, agent_id, action):
        if self.use_baseline and agent_id in env.enm_ids:
            # 敌方智能体：DodgeMissileAgent 调用的自己SB3上训练的PPO
            norm_action = self.baseline_agent.get_action(env, agent_id)
            # np.ndarray(4,)
            # norm_action = self.baseline_agent.normalize_action(env, agent_id, action)
            # norm_action = action[:-1]
            # 4位杆量，可以直接传
            # self._shoot_action[agent_id] = action[-1]
            # 不能起效

            return norm_action

        else:
            norm_action = np.zeros(4)
            norm_action[0] = action[0]
            norm_action[1] = action[1]
            norm_action[2] = action[2]
            norm_action[3] = action[3]
            self._shoot_action[agent_id] = 1 if action[-1] > 0.5 else 0

            return norm_action
    
    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        super().reset(env)
    
    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0

            obs = self.get_obs(env, agent_id)
            obs_list = obs.tolist()
            state = self.get_states(env, agent_id)
            state_list = state.tolist()

            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                logging.info(f'{agent_id} launch mission! '
                             f'Total Steps={env.current_step}, obs={obs_list}, state={state_list}, '
                             f'current_reward={env.cumulative_reward}')


class HierarchicalSingleCombatShootMissileBackTask(HierarchicalSingleCombatTask, SingleCombatDodgeMissileTask):

    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)

        self.reward_functions = [
            # SelfPlayShootPenaltyReward(self.config),  # 打弹就有惩罚，且引诱对方打弹有奖励，零和
            # SelfPlayPostureReward(self.config),
            # SelfPlayShootEventDrivenReward(self.config),
            # SelfPlayShootGapPenalty(self.config),  # 打弹间隔
            # SelfPlayShootPosturePenalty(self.config),  # 打弹时姿势
            # SelfPlayShootWaitReward(self.config),  # 等待奖励
            # SelfPlayEnemyPostureReward(self.config),  # 敌方躲弹
            # AltitudeReward(self.config),  # 防坠地
            SelfPlayShootMissileRewardWithDistance(self.config),
        ]

        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            ShootSafeReturn(self.config),
            Timeout(self.config),
        ]

    def load_observation_space(self):
        return SingleCombatDodgeMissileTask.load_observation_space(self)

    def load_action_space(self):
        if self.shoot_decide_method == "bool":
            self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])
        else:
            self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])  # not decided yet

        # return HierarchicalSingleCombatTask.load_action_space(self)
        return self.action_space

    def get_obs(self, env, agent_id):
        return SingleCombatDodgeMissileTask.get_obs(self, env, agent_id)

    def get_states(self, env, agent_id):
        return SingleCombatDodgeMissileTask.get_states(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
                """
        if self.use_baseline and agent_id in env.enm_ids:
            # 敌方智能体：DodgeMissileAgent 调用的自己SB3上训练的PPO
            norm_action = self.baseline_agent.get_action(env, agent_id)

            return norm_action

        else:
            # Hierarchical的norm 从变化量到杆量
            # generate low-level input_obs
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)
            # (1) delta altitude/heading/velocity
            input_obs[0] = self.norm_delta_altitude[action[0]]
            input_obs[1] = self.norm_delta_heading[action[1]]
            input_obs[2] = self.norm_delta_velocity[action[2]]
            # (2) ego info
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.expand_dims(input_obs, axis=0)
            # output low-level action
            _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
            action_low = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            # normalize low-level action
            norm_act = np.zeros(4)
            norm_act[0] = action_low[0] / 20 - 1.
            norm_act[1] = action_low[1] / 20 - 1.
            norm_act[2] = action_low[2] / 20 - 1.
            norm_act[3] = action_low[3] / 58 + 0.4

            # distance = raw_obs[13] * 10000
            # angle = raw_obs[11] * 180 / np.pi
            # speed = np.linalg.norm([raw_obs[9], raw_obs[10], raw_obs[11]]) * 340
            # alt_diff = -raw_obs[10] * 1000
            # shoot_action = fuzzy_should_attack(distance, angle, speed, alt_diff)
            # self._shoot_action[agent_id] = shoot_action
            self._shoot_action[agent_id] = action[3]

            return norm_act

    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self.launch = {agent_id: False for agent_id in env.agents.keys()}
        self.cumulative_rewards = {agent_id: [0 for _ in range(len(self.reward_functions))] for agent_id in env.agents.keys()}
        return SingleCombatDodgeMissileTask.reset(self, env)

    def get_reward(self, env, agent_id, info={}) -> Tuple[float, dict]:
        """
        Aggregate reward functions

        Args:
            env: environment instance
            agent_id: current agent id
            info: additional info

        Returns:
            (tuple):
                reward(float): total reward of the current timestep
                info(dict): additional info
        """
        total_reward = 0.0

        # 确保初始化过（防止外部提前调用未初始化）
        if not hasattr(self, 'cumulative_rewards'):
            self.cumulative_rewards = {
                aid: [0.0 for _ in range(len(self.reward_functions))]
                for aid in env.agents.keys()
            }

        for i, reward_function in enumerate(self.reward_functions):
            r = reward_function.get_reward(self, env, agent_id)
            total_reward += r
            self.cumulative_rewards[agent_id][i] += r

        return total_reward, info

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0

            obs = self.get_obs(env, agent_id)
            state = self.get_states(env, agent_id)
            self.launch[agent_id] = False

            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                self.launch[agent_id] = True
                logging.info(f'{agent_id} launch mission! '
                             f'Total Steps={env.current_step}, obs={obs}, state={state}, current_reward={env.cumulative_reward}')
                             # f'Total Steps={env.current_step}, current_reward={env.cumulative_reward}')