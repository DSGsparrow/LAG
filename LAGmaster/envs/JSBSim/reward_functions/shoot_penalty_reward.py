from .reward_function_base import BaseRewardFunction
import numpy as np


class ShootPenaltyReward(BaseRewardFunction):
    """
    ShootPenaltyReward
    when launching a missile, give -10 reward for penalty, 
    to avoid launching all missiles at once 
    """
    def __init__(self, config):
        super().__init__(config)
        self.shoot_distance_center = getattr(config, 'shoot_distance_center', 7000)
        self.shoot_distance_sigma = getattr(config, 'shoot_distance_sigma', 3000)

        self.shoot_angle_center = getattr(config, 'shoot_angle_center', 0)
        self.shoot_angle_sigma = getattr(config, 'shoot_angle_sigma', 30)

    def reset(self, task, env):
        self.pre_remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward

        [-35, 20]
        """

        obss = env.get_obs()
        obs = obss[agent_id]
        ego_AO = obs[11] / np.pi * 180
        ego_TA = obs[12] / np.pi * 180
        distance = obs[13] * 10000
        relative_height = obs[10] #  * 1000

        ego_v = np.linalg.norm([obs[5], obs[6], obs[7]]) * 340

        # if distance > 1:  # 距离超过10公里
        #     self._shoot_action[agent_id] = 0
        #
        # elif ego_AO > 50.:  # 视线角过大不可以打弹
        #     self._shoot_action[agent_id] = 0

        reward = 0
        if task.remaining_missiles[agent_id] == self.pre_remaining_missiles[agent_id] - 1:
            # 打弹惩罚，防止乱打
            reward -= 1
            # 距离
            if distance > 10000:
                # 太远
                # reward -= 10
                reward -= 1
            elif distance <= 10000:

                if distance <= self.shoot_distance_center:
                    # 高斯函数下降慢一点：sigma 调大
                    bonus = 1  # np.exp(-((distance - self.shoot_distance_center) ** 2) / (2 * self.shoot_distance_sigma ** 2))
                else:  # if distance <= 10000:
                    # 线性递减：从 1 到 -1
                    ratio = (distance - self.shoot_distance_center) / (10000 - self.shoot_distance_center)  # 0 到 1
                    bonus =  1 - 2 * ratio
                    # bonus =  1 - 1 * ratio

                reward += 1 * bonus

            # 角度
            if ego_AO > 50:
                # 太远
                reward -= 1
            elif ego_AO <= 50:

                ratio = (ego_AO - self.shoot_angle_center) / (50 - self.shoot_angle_center)  # 0 到 1
                bonus = 1 - 2 * ratio # -1到1

                # bonus = np.exp(-((ego_AO - self.shoot_angle_center) ** 2) / (2 * self.shoot_angle_sigma ** 2))
                reward += 1 * bonus

            if ego_TA > 150:
                reward += 0.5

            # 速度
            # 0-0.5
            reward_v = (ego_v - 170)/(340 - 170)
            reward_v = np.clip(reward_v, 0, 1) / 2  # 限制最大奖励

            reward += reward_v

            # 高度
            # if relative_height > 0:
            bonus = np.clip(relative_height - 0.5, -2, 2) / 2
            reward += 0.5 * bonus

            # 只要出现严重偏离就严重惩罚
            obj = [0, 0, 0, 0]
            obj[0] = ego_AO > 50
            obj[1] = distance > 10000
            obj[2] = relative_height < -0.5
            obj[3] = ego_v < 0.7 * 340

            if any(obj):
                reward = -7.5

        self.pre_remaining_missiles[agent_id] = task.remaining_missiles[agent_id]
        return self._process(reward, agent_id)
