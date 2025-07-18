from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import numpy as np


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

                additional_reward = 2 * np.exp(-np.log(2) / 300 * d)
                # 300是1，600是0.5

                additional_reward = additional_reward * 1.5

                # if d >= 600:
                #     additional_reward = - (d - 600) * k + 0.75
                #     additional_reward = np.clip(additional_reward, 0, 0.75)
                #
                #
                # # 根据距离给额外补偿奖励（最多补偿 0.75）
                # if d > 500:
                #     additional_reward = 0.0
                # elif 300 < d <= 500:
                #     # 在 [300, 500] 范围线性增长，靠近 300 米时补偿大
                #     # 300 → 0.75，500 → 0
                #     additional_reward = 0.75 * (500 - d) / 200
                # else:
                #     # d ≤ 300，本应是命中，不应该走到这里
                #     additional_reward = 0.0

                reward += base_penalty + additional_reward

        return self._process(reward, agent_id)

