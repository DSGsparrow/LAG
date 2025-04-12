import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class SpeedGuideReward(BaseRewardFunction):
    """
    AltitudeReward
    Punish if current fighter doesn't satisfy some constraints. Typically negative.
    - Punishment of velocity when lower than safe altitude   (range: [-1, 0])
    - Punishment of altitude when lower than danger altitude (range: [-1, 0])
    """
    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)         # km
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)     # km
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)     # mh

        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        ego_z = env.agents[agent_id].get_position()[-1] / 1000    # unit: km
        enm_z = env.agents[agent_id].enemies[0].get_position()[-1] / 1000  # km
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 340    # unit: mh
        ego_vx = env.agents[agent_id].get_velocity()[0] / 340  # unit: mh
        ego_vy = env.agents[agent_id].get_velocity()[1] / 340  # unit: mh

        ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])

        if ego_v >= 0.8:
            reward = 1
        else:
            reward = (ego_v - 0.4) / (0.8 - 0.4)

        return self._process(reward, agent_id)
