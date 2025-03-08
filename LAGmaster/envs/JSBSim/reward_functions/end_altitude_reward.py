import numpy as np
from sympy.physics.units import velocity

from .reward_function_base import BaseRewardFunction


class EndAltitudeReward(BaseRewardFunction):
    """
    EndAltitudeReward
    Reward when enemy's missile's energy is running out, the fighter chooses to rise up
    and is higher than the enemy. Choose to use positive reward.
    - reward of relative altitude when larger than -1000

    NOTE:
    - Only support one-to-one environments.
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the clip of relative height.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        velocity_threthold = 1

        ego_z = env.agents[agent_id].get_position()[-1] / 1000    # unit: km
        enm_z = env.agents[agent_id].enemies[0].get_position()[-1] / 1000    # unit: km

        height_diff = ego_z - enm_z

        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_v = missile_sim.get_velocity()

            if np.linalg.norm(missile_v) / 340 <= velocity_threthold:
                endgame_altitude_reward = np.clip(height_diff, -1, 3) + 1  # 高度差越高越好（-1~3）
            else:
                endgame_altitude_reward = 0

        else:
            endgame_altitude_reward = 0

        # self.reward_trajectory[agent_id].append([endgame_altitude_reward])  # 包含在_process里

        return self._process(endgame_altitude_reward, agent_id)
