import numpy as np
from gymnasium import spaces
from collections import deque
import logging

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, PostureShootReward
from ..reward_functions import ShootPenaltyReward, ShootGapPenaltyReward, RelativeAltitudeReward, ShootEventDrivenReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, DodgeMissileSafeReturn
from ..termination_conditions import SafeReturn, ShootSafeReturn
from ..reward_functions import EndAltitudeReward
from ..core.simulatior import MissileSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R
from .singlecombat_with_missle_task import SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, SingleCombatShootMissileTask


class SingleCombatShootMissileImitationTask(SingleCombatDodgeMissileTask):
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


