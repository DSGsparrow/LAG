from .termination_condition_base import BaseTerminationCondition
import numpy as np


class ShootWrong(BaseTerminationCondition):
    """
    SafeReturn.
    End up the simulation if:
        - the current aircraft has been shot down.
        - all the enemy-aircrafts has been destroyed while current aircraft is not under attack.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.

        End up the simulation if:
            - the current aircraft has been shot down.
            - all the enemy-aircrafts has been destroyed while current aircraft is not under attack.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        # the current aircraft has crashed

        obss = env.get_obs()
        obs = obss[agent_id]
        ego_AO = obs[11] / np.pi * 180
        ego_TA = obs[12] / np.pi * 180
        distance = obs[13] * 10000
        relative_height = obs[10]  # * 1000

        ego_v = np.linalg.norm([obs[5], obs[6], obs[7]]) * 340

        pre_remaining_missiles = task.reward_functions[1].pre_remaining_missiles

        if task.remaining_missiles[agent_id] == pre_remaining_missiles[agent_id] - 1:
            # 只考虑发射时
            obj = [0, 0, 0, 0]
            obj[0] = ego_AO > 50
            obj[1] = distance > 10000
            obj[2] = relative_height < -0.5
            obj[3] = ego_v < 0.7 * 340

            if any(obj):
                self.log(f'{agent_id} Shoot Wrong! Total Steps={env.current_step}')
                return True, False, info
            else:
                self.log(f'{agent_id} Shoot Right at least! Total Steps={env.current_step}')
                return False, False, info

        else:
            return False, False, info
