import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R


class ShootEnemyPostureReward(BaseRewardFunction):
    """
    PostureReward = Orientation * Range
    - Orientation: Encourage pointing at enemy fighter, punish when is pointed at.
    - Range: Encourage getting closer to enemy fighter, punish if too far away.

    NOTE:
    - Only support one-to-one environments.
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        """
        只算敌方，最大奖励时：敌方背对自己（视线角180），高度下降

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """

        reward = 0

        if agent_id == 'A0100' and env.agents[agent_id].num_left_missiles == 0:
            # 只算自己，且已经发射
            obss = env.get_obs()
            obs = obss['B0100']
            ego_AO = obs[11] / np.pi * 180
            ego_TA = obs[12] / np.pi * 180
            distance = obs[13] * 10000
            relative_height = obs[10]  # * 1000  # A - B
            ego_alt = obs[0] * 5000

            # 角度
            reward_a = ego_AO / 180

            # 相对高度
            reward_dh = np.clip(relative_height, -2, 2) / 2

            # 高度 越靠近2500越好越靠近1  5000以上是0
            reward_h = 1 - (ego_alt - 2500) / 2500
            reward_h = np.clip(reward_h, 0, 1)

            reward = 0.5 * reward_a + 0.1 * reward_dh + 0.4 * reward_h

        return self._process(reward, agent_id)