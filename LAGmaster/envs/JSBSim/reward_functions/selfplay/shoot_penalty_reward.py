from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import numpy as np


class SelfPlayShootPenaltyReward(BaseRewardFunction):
    """
    ShootPenaltyReward
    when launching a missile, give -10 reward for penalty, 
    to avoid launching all missiles at once 
    """
    def __init__(self, config):
        super().__init__(config)

    def reset(self, task, env):
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

        reward = 0
        if task.launch[agent_id]:
            # 打弹惩罚，防止乱打
            reward -= 1

        for enemy in env.agents[agent_id].enemies:
            # 骗弹成功
            enemy_id = enemy.uid
            if task.launch[enemy_id]:
                reward += 1

        return self._process(reward, agent_id)
