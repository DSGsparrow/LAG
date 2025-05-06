import numpy as np
from gymnasium import spaces
from collections import deque
import logging

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward
from ..reward_functions import ShootPenaltyReward, ShootGapPenaltyReward, RelativeAltitudeReward, ShootEventDrivenReward
from ..reward_functions import AltitudeGuideReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, DodgeMissileSafeReturn
from ..termination_conditions import SafeReturn, ShootSafeReturn
from ..reward_functions import EndAltitudeReward
from ..core.simulatior import MissileSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R


class SingleCombatDodgeMissileTask(SingleCombatTask):
    """This task aims at training agent to dodge missile attacking
    """
    def __init__(self, config):
        super().__init__(config)

        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)
        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]
        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            DodgeMissileSafeReturn(self.config),
            Timeout(self.config),
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def get_states(self, env, agent_id):
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))

        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_obs_list[:3], *ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_obs_list[:3], *enm_cur_ned, *enm_obs_list[6:9]])

        state = np.concatenate([ego_feature, enm_feature], axis=0)

        # state = np.zeros(len(ego_feature) + len(enm_feature))
        #
        # for j in range(len(ego_feature)):
        #     state[j] = ego_feature[j]
        # for j in range(len(enm_feature)):
        #     state[len(ego_feature) + j] = enm_feature[j]

        # missile_sim = env.agents[agent_id].check_missile_warning()
        # if missile_sim is not None:
        #     missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
        #     for j in range(6):
        #         state[12+j] = missile_feature[j]

        return state

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        - relative missile info
            - [15] delta_v_body_x
            - [16] delta altitude
            - [17] ego_AO
            - [18] ego_TA
            - [19] relative distance
            - [20] side flag
        """
        norm_obs = np.zeros(21)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        # (2) relative enm info
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        else:
            pass
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.lock_duration = {agent_id: deque(maxlen=int(1 / env.time_interval)) for agent_id in env.agents.keys()}
        return super().reset(env)

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [Rule-based missile launch]
            target = agent.enemies[0].get_position() - agent.get_position()
            heading = agent.get_velocity()
            distance = np.linalg.norm(target)
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            self.lock_duration[agent_id].append(attack_angle < self.max_attack_angle)
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]

            shoot_flag = agent.is_alive and np.sum(self.lock_duration[agent_id]) >= self.lock_duration[agent_id].maxlen \
                and distance <= self.max_attack_distance and self.remaining_missiles[agent_id] > 0 and shoot_interval >= self.min_attack_interval

            shoot_flag = agent.is_alive and self.remaining_missiles[agent_id] > 0
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step


class HierarchicalSingleCombatDodgeUnknownMissileTask(HierarchicalSingleCombatTask, SingleCombatDodgeMissileTask):

    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)

        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeGuideReward(self.config),
            EventDrivenReward(self.config),
            EndAltitudeReward(self.config),
        ]

        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            DodgeMissileSafeReturn(self.config),
            Timeout(self.config),
        ]

    def load_observation_space(self):
        return SingleCombatDodgeMissileTask.load_observation_space(self)

    def load_action_space(self):
        return HierarchicalSingleCombatTask.load_action_space(self)

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        - relative missile info
            - [15] delta_v_body_x
            - [16] delta altitude
            - [17] ego_AO
            - [18] ego_TA
            - [19] relative distance
            - [20] side flag
            - [21] flag of if missile exits
        """
        norm_obs = np.zeros(22)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        # (2) relative enm info
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_warning()
        # if there is a missile, get warning
        if missile_sim is not None:
            norm_obs[15] = 0
            norm_obs[16] = 0
            norm_obs[17] = 0
            norm_obs[18] = 0
            norm_obs[19] = 0
            norm_obs[20] = 0
            norm_obs[21] = 1
        else:
            pass
        return norm_obs

    def get_states(self, env, agent_id):
        return SingleCombatDodgeMissileTask.get_states(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
                """
        if self.use_baseline and agent_id in env.enm_ids:
            # 敌方智能体：ShootAgent 调用的自己SB3上训练的PPO
            action = self.baseline_agent.get_action(env.agents[agent_id])
            # np.ndarray(4,)
            # norm_action = self.baseline_agent.normalize_action(env, agent_id, action)
            norm_action = action[:-1]
            # 4位杆量，可以直接传
            self._shoot_action[agent_id] = 1 if env.current_step >= 10 else 0  # action[-1]  # shoot immediately
            # 最后的发射标志位
            # prior
            # obs = self.get_obs(env, agent_id)
            # ego_AO = obs[11] / np.pi * 180
            # ego_TA = obs[12] / np.pi * 180
            # distance = obs[13]
            #
            # if distance > 1:  # 距离超过10公里
            #     self._shoot_action[agent_id] = 0
            #
            # elif ego_AO > 50.:  # 视线角过大不可以打弹
            #     self._shoot_action[agent_id] = 0

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
            action = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            # normalize low-level action
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20 - 1.
            norm_act[1] = action[1] / 20 - 1.
            norm_act[2] = action[2] / 20 - 1.
            norm_act[3] = action[3] / 58 + 0.4

            return norm_act

    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return SingleCombatDodgeMissileTask.reset(self, env)

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0

            obs = self.get_obs(env, agent_id)
            state = self.get_states(env, agent_id)

            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                logging.info(f'{agent_id} launch mission! '
                             f'Total Steps={env.current_step}, obs={obs}, state={state}, current_reward={env.cumulative_reward}')


class HierarchicalSingleCombatDodgeKnownMissileTask(HierarchicalSingleCombatTask, SingleCombatDodgeMissileTask):

    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)

        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeGuideReward(self.config),
            EventDrivenReward(self.config),
            EndAltitudeReward(self.config),
        ]

        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            DodgeMissileSafeReturn(self.config),
            Timeout(self.config),
        ]

    def load_observation_space(self):
        return SingleCombatDodgeMissileTask.load_observation_space(self)

    def load_action_space(self):
        return HierarchicalSingleCombatTask.load_action_space(self)

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        - relative missile info
            - [15] delta_v_body_x
            - [16] delta altitude
            - [17] ego_AO
            - [18] ego_TA
            - [19] relative distance
            - [20] side flag
            - [21] flag of if missile exits
        """
        norm_obs = np.zeros(22)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        # (2) relative enm info
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_warning()
        # if there is a missile, get warning
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
            norm_obs[21] = 1
        else:
            pass
        return norm_obs

    def get_states(self, env, agent_id):
        return SingleCombatDodgeMissileTask.get_states(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
                """
        if self.use_baseline and agent_id in env.enm_ids:
            # 敌方智能体：ShootAgent 调用的自己SB3上训练的PPO
            action = self.baseline_agent.get_action(env.agents[agent_id])
            # np.ndarray(4,)
            # norm_action = self.baseline_agent.normalize_action(env, agent_id, action)
            norm_action = action[:-1]
            # 4位杆量，可以直接传
            self._shoot_action[agent_id] = 1 if env.current_step >= 10 else 0  # action[-1]  # shoot immediately
            # 最后的发射标志位
            # prior
            # obs = self.get_obs(env, agent_id)
            # ego_AO = obs[11] / np.pi * 180
            # ego_TA = obs[12] / np.pi * 180
            # distance = obs[13]
            #
            # if distance > 1:  # 距离超过10公里
            #     self._shoot_action[agent_id] = 0
            #
            # elif ego_AO > 50.:  # 视线角过大不可以打弹
            #     self._shoot_action[agent_id] = 0

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
            action = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            # normalize low-level action
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20 - 1.
            norm_act[1] = action[1] / 20 - 1.
            norm_act[2] = action[2] / 20 - 1.
            norm_act[3] = action[3] / 58 + 0.4

            return norm_act

    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return SingleCombatDodgeMissileTask.reset(self, env)

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0

            obs = self.get_obs(env, agent_id)
            state = self.get_states(env, agent_id)

            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                logging.info(f'{agent_id} launch mission! '
                             f'Total Steps={env.current_step}, obs={obs}, state={state}, current_reward={env.cumulative_reward}')

