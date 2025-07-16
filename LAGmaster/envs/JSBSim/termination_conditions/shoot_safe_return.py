from msilib import make_id

from .termination_condition_base import BaseTerminationCondition


class ShootSafeReturn(BaseTerminationCondition):
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

        if env.agents[agent_id].is_shotdown:
            self.log(f'{agent_id} has been shot down! Total Steps={env.current_step}')
            info[f'{agent_id} success'] = False
            return True, False, info

        elif env.agents[agent_id].is_crash:
            self.log(f'{agent_id} has crashed! Total Steps={env.current_step}')
            info[f'{agent_id} success'] = False
            return True, False, info

        # all the enemy-aircrafts has been destroyed while current aircraft is not under attack
        # win hit enemy
        elif all([not enemy.is_alive for enemy in env.agents[agent_id].enemies]) \
                and all([not missile.is_alive for missile in env.agents[agent_id].under_missiles]):
            if all([enemy.is_shotdown for enemy in env.agents[agent_id].enemies]) :
                self.log(f'{agent_id} mission completed! Hit the enemy! Total Steps={env.current_step}')
                info[f'{agent_id} success'] = True
                info[f'{agent_id} shoot success'] = True
            else:
                self.log(f'{agent_id} enemy killed himself! Total Steps={env.current_step}')
                info[f'{agent_id} success'] = True
                info[f'{agent_id} shoot success'] = False
            return True, True, info

        # enemy is alive but current aircraft's missiles all failed
        # lose didn't hit enemy
        elif all([enemy.is_alive for enemy in env.agents[agent_id].enemies]) \
                and env.agents[agent_id].num_left_missiles == 0 \
                and all([not missile.is_alive for missile in env.agents[agent_id].launch_missiles]):
            # self.log(f'{agent_id} mission failed! Did not hit enemy! Total Steps={env.current_step}')
            info[f'{agent_id} shoot success'] = False
            info['draw and both live'] = True
            return True, False, info

        # # not crushed or shot down and not under attack
        # elif all([not missile.is_alive for missile in env.agents[agent_id].under_missiles]):
        #     self.log(f'{agent_id} dodge succeeded! Total Steps={env.current_step}')
        #     info['dodge success'] = True
        #     return False, False, info

        else:
            return False, False, info
