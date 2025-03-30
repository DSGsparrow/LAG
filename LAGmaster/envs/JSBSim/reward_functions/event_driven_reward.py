from .reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """
    def __init__(self, config):
        super().__init__(config)

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
        if env.agents[agent_id].is_shotdown:
            reward -= 200
        elif env.agents[agent_id].is_crash:
            reward -= 200
        for missile in env.agents[agent_id].launch_missiles:
            if missile.is_success:
                reward += 200

        if all([not missile.is_alive for missile in env.agents[agent_id].under_missiles]) \
                and agent_id == 'A0100' \
                and all([enemy.num_left_missiles == 0 for enemy in env.agents[agent_id].enemies]):
            # dodge success
            reward += 200

        return self._process(reward, agent_id)
