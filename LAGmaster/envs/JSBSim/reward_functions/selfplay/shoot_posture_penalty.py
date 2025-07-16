from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import numpy as np


class SelfPlayShootPosturePenalty(BaseRewardFunction):
    """
    ShootPenaltyReward
    when launching a missile, give -10 reward for penalty, 
    to avoid launching all missiles at once 
    """
    def __init__(self, config):
        super().__init__(config)

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

        reward = 0
        if task.launch[agent_id]:
            # 只要出现严重偏离就严重惩罚
            obj = [0, 0, 0, 0]
            obj[0] = ego_AO > 50
            obj[1] = distance > 11000
            obj[2] = relative_height < -0.5
            obj[3] = ego_v < 0.7 * 340

            if any(obj):
                reward = np.sum(np.array(obj)) / 4

        # self.pre_remaining_missiles[agent_id] = task.remaining_missiles[agent_id]
        return self._process(reward, agent_id)
