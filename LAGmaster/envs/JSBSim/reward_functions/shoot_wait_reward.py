from .reward_function_base import BaseRewardFunction
import numpy as np


class ShootWaitReward(BaseRewardFunction):
    """
    ShootWaitReward
    if didn't shoot a missile choose to wait, give rewards depends on the states
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
        if task.remaining_missiles[agent_id] == self.pre_remaining_missiles[agent_id]:
            # 没有打弹的话

            # 1 distance
            # self.shoot_distance_center米内就小于1了
            delta = self.shoot_distance_center - distance
            alpha = 0.0005
            reward_d = np.exp(-alpha * delta)
            reward_d = min(reward_d, 1.5)  # 限制最大奖励

            # 2 self angle
            # 50度内小于1
            delta = ego_AO - 50.0
            beta = 0.05
            reward_a = np.exp(beta * delta)
            reward_a = min(reward_a, 1.5)  # 限制最大奖励

            # 3 height diff
            # 高500米的时候发射最好
            gamma = 1
            reward_hd = 2 * (1 - np.exp(-gamma * abs(relative_height - 0.5)))

            # 4 speed
            # 0.8mach以内小于1
            lam = 0.01
            reward_v = np.exp(-lam * (ego_v - 272))
            reward_v = min(reward_v, 1.5)  # 限制最大奖励

            reward = reward_hd * reward_a * reward_d * reward_v

        self.pre_remaining_missiles[agent_id] = task.remaining_missiles[agent_id]
        return self._process(reward, agent_id)
