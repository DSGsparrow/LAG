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

        lock_duration = task.lock_duration_num[agent_id]

        # if distance > 1:  # 距离超过10公里
        #     self._shoot_action[agent_id] = 0
        #
        # elif ego_AO > 50.:  # 视线角过大不可以打弹
        #     self._shoot_action[agent_id] = 0

        w = [0.35, 0.3, 0.2, 0.15]

        reward = 0
        if task.remaining_missiles[agent_id] == self.pre_remaining_missiles[agent_id] == 1:
            # 没有打弹的话
            #各个奖励范围都在-1 - 1，最终奖励也是-1 - 1

            # 1 distance
            # self.shoot_distance_center米内就小于0了
            delta = distance - self.shoot_distance_center
            alpha = 0.0005
            reward_d = 1 - np.exp(-alpha * delta)
            reward_d = np.clip(reward_d, -1, 1)  # 限制最大奖励

            # 2 self angle
            # 20度内小于0
            delta = ego_AO - 20.0
            beta = 0.05
            reward_a = 1 - np.exp(-beta * delta)
            reward_a = np.clip(reward_a, -1, 1)  # 限制最大奖励

            # 3 height diff
            # 高500米的时候发射最好 0
            #
            gamma = 1
            reward_hd = (np.exp(gamma * abs(relative_height - 0.5)) - 0.5) * 2
            reward_hd = np.clip(reward_hd, -1, 1)

            # 4 speed
            # 0.8mach以内大于0
            lam = 0.01
            reward_v = 1 - np.exp(lam * (ego_v - 272))
            reward_v = np.clip(reward_v, -1, 1)  # 限制最大奖励

            reward = w[0] * reward_d + w[1] * reward_a + w[2] * reward_hd + w[3] * reward_v

        # if lock_duration > 50:
        #     reward = 0.5 * reward / (lock_duration - 50)

        # print(reward)

        self.pre_remaining_missiles[agent_id] = task.remaining_missiles[agent_id]
        return self._process(reward, agent_id)
