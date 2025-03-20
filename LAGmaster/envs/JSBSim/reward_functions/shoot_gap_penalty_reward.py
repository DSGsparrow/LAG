from .reward_function_base import BaseRewardFunction


class ShootGapPenaltyReward(BaseRewardFunction):
    """
    ShootPenaltyReward
    when launching a missile, give -10 reward for penalty,
    to avoid launching all missiles at once
    """
    def __init__(self, config):
        super().__init__(config)
        self.last_shoot_time = 0

    def reset(self, task, env):
        self.pre_remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.last_shoot_time = 0
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.
        penalty if time between this shoot and last is too short

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        if task.remaining_missiles[agent_id] == self.pre_remaining_missiles[agent_id] - 1:
            # reward -= 10
            current_step = env.current_step
            if current_step - self.last_shoot_time <= 10 and self.last_shoot_time > 0:
                reward = -25
            else:
                reward = 0
            self.last_shoot_time = env.current_step
        self.pre_remaining_missiles[agent_id] = task.remaining_missiles[agent_id]
        return self._process(reward, agent_id)
