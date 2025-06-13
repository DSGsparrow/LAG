import numpy as np

from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class PostureShootReward(BaseRewardFunction):
    """
    PostureReward = Orientation * Range
    - Orientation: Encourage pointing at enemy fighter, punish when is pointed at.
    - Range: Encourage getting closer to enemy fighter, punish if too far away.

    NOTE:
    - Only support one-to-one environments.
    """
    def __init__(self, config):
        super().__init__(config)
        self.orientation_version = getattr(self.config, f'{self.__class__.__name__}_orientation_version', 'v2')
        self.range_version = getattr(self.config, f'{self.__class__.__name__}_range_version', 'v3')
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 3.0)

        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is a complex function of AO, TA and R in the last timestep.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        new_reward = 0
        missile_num = task.remaining_missiles[agent_id]
        if missile_num > 0 or agent_id == "B0100":
            # feature: (north, east, down, vn, ve, vd)
            ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                     env.agents[agent_id].get_velocity()])
            x = env.agents[agent_id].enemies[0].get_position()
            v = env.agents[agent_id].enemies[0].get_velocity()
            ego_speed = np.linalg.norm([ego_feature[3], ego_feature[4], ego_feature[5]])

            for enm in env.agents[agent_id].enemies:
                enm_feature = np.hstack([enm.get_position(),
                                        enm.get_velocity()])
                AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
                orientation_reward = self.orientation_fn(AO, TA)
                range_reward = self.range_fn(R / 1000)
                speed_reward = self.speed_reward_function(ego_speed)
                new_reward += orientation_reward * range_reward * speed_reward
        else:
            new_reward = 0
            orientation_reward = 0
            range_reward = 0
        return self._process(new_reward, agent_id, (orientation_reward, range_reward))

    def tactical_angle_reward(self, AO, TA):
        my_angle = AO / np.pi * 180
        enemy_angle = 180 - TA / np.pi * 180

        # 归一化角度到 [0, 1]，0 表示完全对准，1 表示完全偏离
        my_norm = np.clip(my_angle / 30.0, 0, 1)
        enemy_norm = np.clip(enemy_angle / 30.0, 0, 1)

        if my_angle < 30 <= enemy_angle:
            # ✅ 优势态势
            # 奖励范围：1.2 ~ 1.5（越准越高，敌人越偏越好）
            reward = 1.5 - 0.3 * my_norm + 0.1 * enemy_norm
            return reward

        elif my_angle < 30 and enemy_angle < 30:
            # ⚖️ 均势态势
            # 奖励范围：0.8 ~ 1.1
            reward = 1.1 - 0.3 * my_norm - 0.3 * (1 - enemy_norm)
            return reward

        elif my_angle >= 30 and enemy_angle >= 30:
            # 💤 双方都没对准
            # 奖励范围：0.4 ~ 0.7（轻微鼓励我朝敌人方向转头）
            reward = 0.4 + 0.3 * (1 - my_norm)
            return reward

        else:
            # ❌ 劣势态势（我没看敌人，敌人对着我）
            # 奖励范围：≤ 0.4
            # 惩罚敌人越准惩罚越重，鼓励我开始对准
            penalty = 0.6 * (1 - enemy_norm)
            bonus = 0.2 * (1 - my_norm)
            reward = 0.4 - penalty + bonus
            return reward


    def get_orientation_function(self, version):
        if version == 'v0':
            return lambda AO, TA: (1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3. \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        elif version == 'v1':
            return lambda AO, TA: (1. - np.tanh(2 * (AO - np.pi / 2))) / 2. \
                * (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5
        elif version == 'v2':
            return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        elif version == 'v3':
            return lambda AO, TA: self.tactical_angle_reward(AO, TA)
        else:
            raise NotImplementedError(f"Unknown orientation function version: {version}")

    def distance_reward(self, R, missile_num):
        # r = distance / 1000
        r6 = 0.6  # 奖励在6公里处
        k = 1.5  # 控制远距离惩罚曲线陡度（>1确保梯度递增）
        a = 0.07  # 远距离惩罚幅度
        if missile_num > 0:
            return 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)
        else:
            if R < 2:
                return 1.0
            elif R <= 6:
                return ((1.0 - r6) / (6 - 2)) * (6 - R) + r6
            else:
                return max(-a * (R - 6) ** k + r6, -1.5)

    def get_range_funtion(self, version):
        if version == 'v0':
            return lambda R: np.exp(-(R - self.target_dist) ** 2 * 0.004) / (1. + np.exp(-(R - self.target_dist + 2) * 2))
        elif version == 'v1':
            return lambda R: np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                     (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1)
        elif version == 'v2':
            return lambda R: max(np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                         (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1), np.sign(7 - R))
        elif version == 'v3':
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)
        else:
            raise NotImplementedError(f"Unknown range function version: {version}")

    def speed_reward_function(self, v):
        if v <= 250:
            # 从 (200, 0.2) 到 (250, 0.4)
            return 0.2 + (v - 200) * (0.3 - 0.2) / (250 - 200)
        elif v <= 340:
            # 从 (250, 0.4) 到 (340, 0.8)
            return 0.3 + (v - 250) * (0.9 - 0.3) / (340 - 250)
        elif 340 < v < 400:
            # 从 (340, 0.8) 到 (400, 0.9)
            return 0.9 + (v - 340) * (1 - 0.9) / (400 - 340)
        else:
            return 1

