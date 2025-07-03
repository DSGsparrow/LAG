import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R


class SelfPlayEnemyPostureReward(BaseRewardFunction):
    """
    PostureReward = Orientation * Range
    - Orientation: Encourage pointing at enemy fighter, punish when is pointed at.
    - Range: Encourage getting closer to enemy fighter, punish if too far away.

    NOTE:
    - Only support one-to-one environments.
    """
    def __init__(self, config):
        super().__init__(config)

    def reset(self, task, env):
        self.pre_height = {agent_id: 0 for agent_id, agent in env.agents.items()}
        self.enemy_pre_height = {agent_id: 0 for agent_id, agent in env.agents.items()}
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        只算敌方，最大奖励时：敌方背对自己（视线角180），高度下降

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        obss = env.get_obs()
        obs = obss[agent_id]
        ego_AO = obs[11] / np.pi * 180
        ego_TA = 180 - obs[12] / np.pi * 180
        distance = obs[13] * 10000
        relative_height = obs[10]  # * 1000
        ego_height = obs[0] * 5000
        enm_height = 1000 * relative_height * 1000 + ego_height

        ego_rpy = env.agents[agent_id].get_rpy()
        ego_pitch = np.rad2deg(ego_rpy[1])

        enm_rpy = env.agents[agent_id].enemies[0].get_rpy()
        enm_pitch = np.rad2deg(enm_rpy[1])

        obj = [0, 0, 0]

        reward = 0
        launch_missile = env.agents[agent_id].check_missile_launching()

        if launch_missile:
            # 自己的弹的奖励
            # 角度
            if ego_TA > 50:
                obj[0] = 1

            # 高度（用俯仰角代替）
            if enm_pitch < -10:
                obj[1] = 1

            # 距离
            if distance > 10000:
                obj[2] = 1

            reward += 0.3 * obj[0] + 0.6 * obj[1] + 0.1 * obj[2]

        obj = [0, 0, 0]
        under_missile = env.agents[agent_id].check_missile_warning()
        if under_missile:
            if ego_AO > 50:
                obj[0] = 1

            if ego_pitch < -10:
                obj[1] = 1

            if distance > 10000:
                obj[2] = 1

            reward -= 0.3 * obj[0] + 0.6 * obj[1] + 0.1 * obj[2]

        return self._process(reward, agent_id)