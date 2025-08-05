from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import numpy as np
import logging


class SelfPlayShootMissileRewardWithDistance(BaseRewardFunction):
    """
    导弹命中或未命中的奖励函数。
    - 命中目标：+1.0
    - 未命中目标：-0.75 + 距离相关奖励（最多 +0.75，最终 reward ∈ [-0.75, +0.75]）
    """

    def __init__(self, config):
        super().__init__(config)

    def reset(self, task, env):
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        reward = 0.0
        agent = env.agents[agent_id]

        for missile in agent.launch_missiles:
            if not missile.is_done:
                continue  # 忽略未结束的导弹

            if missile.is_success:
                # 命中
                reward += 1.0
            elif missile.is_miss:  # _MissileSimulator__status == 2:  # MissileSimulator.MISS:
                # 未命中：先加基础惩罚
                base_penalty = -0.75

                # 获取未命中时的距离
                d = missile.target_distance  # 单位：米

                additional_reward = 2 * np.exp(-np.log(2) / 2000 * d)
                # 2000是1，4000是0.5, 6000是0.25， 8800 0.1

                additional_reward = np.clip(additional_reward, 0, 1)

                additional_reward = additional_reward * 1.5

                reward += base_penalty + additional_reward
                logging.critical(f'missile final distance: {d}, shoot reward: {reward * 150}')

        return self._process(reward, agent_id)

