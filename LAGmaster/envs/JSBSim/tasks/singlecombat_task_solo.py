import numpy as np
from gymnasium import spaces
from collections import deque
import logging
from typing import List, Tuple

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, PostureShootReward
from ..reward_functions import ShootPenaltyReward, ShootGapPenaltyReward, RelativeAltitudeReward, ShootEventDrivenReward
from ..reward_functions import ShootEnemyPostureReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, DodgeMissileSafeReturn, ShootWrong
from ..termination_conditions import SafeReturn, ShootSafeReturn
from ..reward_functions import EndAltitudeReward, ShootWaitReward
from ..reward_functions import (SelfPlayShootPenaltyReward, SelfPlayPostureReward, SelfPlayShootEventDrivenReward,
                                SelfPlayShootGapPenalty, SelfPlayShootPosturePenalty,
                                SelfPlayShootWaitReward, SelfPlayEnemyPostureReward, SelfPlayShootMissileRewardWithDistance)
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
            norm_action = self.baseline_agent.get_action(env, agent_id)
            # norm_action = np.zeros(4)
            # norm_action[0] = action[0]
            # norm_action[1] = action[1]
            # norm_action[2] = action[2]
            # norm_action[3] = action[3]
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


class HierarchicalSingleCombatShootMissileSoloTask(HierarchicalSingleCombatTask, SingleCombatDodgeMissileTask):

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
        self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])

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
            norm_action = self.baseline_agent.get_action(env.agents[agent_id])

            return norm_action

        else:
            norm_action = self.baseline_agent.get_action(env.agents[agent_id])
            # norm_action = np.zeros(4)
            # norm_action[0] = action[0]
            # norm_action[1] = action[1]
            # norm_action[2] = action[2]
            # norm_action[3] = action[3]
            self._shoot_action[agent_id] = 1 if action[-1] > 0.5 else 0

            return norm_action

    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}

        self.lock_duration_num = {agent_id: 0 for agent_id in env.agents.keys()}
        self.lose_lock_duration_num = {agent_id: 0 for agent_id in env.agents.keys()}

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

    # def step(self, env):
    #     SingleCombatTask.step(self, env)
    #     for agent_id, agent in env.agents.items():
    #         # [RL-based missile launch with limited condition]
    #         shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0
    #
    #         obs = self.get_obs(env, agent_id)
    #         state = self.get_states(env, agent_id)
    #
    #         target = agent.enemies[0].get_position() - agent.get_position()
    #         heading = agent.get_velocity()
    #         distance = np.linalg.norm(target)
    #         attack_angle = np.rad2deg(
    #             np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
    #
    #         if attack_angle < self.max_attack_angle:
    #             # 超出视线角开始计时
    #             self.lock_duration_num[agent_id] += 1
    #         else:
    #             self.lock_duration_num[agent_id] = 0
    #             if self.remaining_missiles['A0100'] == 0 and agent_id == 'A0100':
    #                 # A已经发弹
    #                 self.lose_lock_duration_num['B0100'] += 1
    #
    #         if shoot_flag:
    #             new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
    #             env.add_temp_simulator(
    #                 MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
    #             self.remaining_missiles[agent_id] -= 1
    #             logging.info(f'{agent_id} launch mission! '
    #                          f'Total Steps={env.current_step}, obs={obs}, state={state}, current_reward={env.cumulative_reward}')