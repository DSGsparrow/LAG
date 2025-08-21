from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import numpy as np


class SelfPlayShootWaitReward(BaseRewardFunction):
    """
    若未发射且无来弹预警，仅按距离给“等待奖励”：
    - 距离 >= max_reward_distance → 奖励 1.0
    - 距离 <= min_reward_distance → 奖励 0.0
    - 其间线性插值，随距离增大线性升高到 1.0
    """

    def _cfg(self, key: str, default):
        prefixed = getattr(self.config, f'{self.__class__.__name__}_{key}', None)
        if prefixed is not None:
            return prefixed
        plain = getattr(self.config, key, None)
        return default if plain is None else plain

    def __init__(self, config):
        super().__init__(config)
        # 距离阈值（单位：米）
        self.min_reward_distance = self._cfg("min_reward_distance", 20000)  # 低于该值不给等待奖励
        self.max_reward_distance = self._cfg("max_reward_distance", 25000)  # 高于该值满奖励
        # 观测中距离的缩放（obs[13]*scale → 米）
        self.obs13_to_meter = self._cfg("obs13_to_meter", 10000.0)
        # 限制只在前 N 步生效，避免回合末尾弹失效“蹭奖励”
        self.wait_reward_steps_cutoff = self._cfg("wait_reward_steps_cutoff", 80)

        # 防御：确保阈值顺序正确
        if self.max_reward_distance <= self.min_reward_distance:
            self.max_reward_distance = self.min_reward_distance + 1.0

    def reset(self, task, env):
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        reward = 0.0
        have_warning = env.agents[agent_id].check_missile_warning()   # 来弹预警
        is_launching = env.agents[agent_id].check_missile_launching() # 自己在打弹
        step = env.current_step

        # 仅在“未发射 & 无预警 & 前N步”时给等待奖励
        if (not have_warning) and (not is_launching) and (step <= self.wait_reward_steps_cutoff):
            obs = env.get_obs()[agent_id]
            distance_m = float(obs[13]) * self.obs13_to_meter  # 米

            # 线性插值并夹紧到 [0, 1]
            t = (distance_m - self.min_reward_distance) / (self.max_reward_distance - self.min_reward_distance)
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            reward = t  # 远→1，近→0

        # 若你希望一旦发射就不给等待奖励/或给惩罚，可打开下面一行：
        # if is_launching:
        #     reward = 0.0  # 或者设为负值

        return self._process(reward, agent_id)






# class SelfPlayShootWaitReward(BaseRewardFunction):
#     """
#     ShootWaitReward
#     if didn't shoot a missile choose to wait, give rewards depends on the states
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         self.shoot_distance_center = getattr(config, 'shoot_distance_center', 7000)
#         self.shoot_distance_sigma = getattr(config, 'shoot_distance_sigma', 3000)
#
#         self.shoot_angle_center = getattr(config, 'shoot_angle_center', 0)
#         self.shoot_angle_sigma = getattr(config, 'shoot_angle_sigma', 30)
#
#     def reset(self, task, env):
#         self.pre_remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
#         return super().reset(task, env)
#
#     def get_reward(self, task, env, agent_id):
#         """
#         Reward is the sum of all the events.
#
#         Args:
#             task: task instance
#             env: environment instance
#
#         Returns:
#             (float): reward
#         """
#
#         obss = env.get_obs()
#         obs = obss[agent_id]
#         ego_AO = obs[11] / np.pi * 180
#         ego_TA = obs[12] / np.pi * 180
#         distance = obs[13] * 10000
#         relative_height = obs[10] #  * 1000
#
#         ego_v = np.linalg.norm([obs[5], obs[6], obs[7]]) * 340
#
#         w = [0.35, 0.3, 0.2, 0.15]
#
#         reward = 0
#         # if task.remaining_missiles[agent_id] == self.pre_remaining_missiles[agent_id] == 1:
#         if not (env.agents[agent_id].check_missile_warning or env.agents[agent_id].check_missile_launching):
#             # 没有打弹的话
#             #各个奖励范围都在-1 - 1，最终奖励也是-1 - 1
#
#             # 1 distance
#             # self.shoot_distance_center米内就小于0了
#             delta = distance - self.shoot_distance_center
#             alpha = 0.0005
#             reward_d = 1 - np.exp(-alpha * delta)
#             reward_d = np.clip(reward_d, -1, 1)  # 限制最大奖励
#
#             # 2 self angle
#             # 20度内小于0
#             delta = ego_AO - 20.0
#             beta = 0.05
#             reward_a = 1 - np.exp(-beta * delta)
#             reward_a = np.clip(reward_a, -1, 1)  # 限制最大奖励
#
#             # 3 height diff
#             # 高500米的时候发射最好 0
#             #
#             gamma = 1
#             reward_hd = (np.exp(gamma * abs(relative_height - 0.5)) - 0.5) * 2
#             reward_hd = np.clip(reward_hd, -1, 1)
#
#             # 4 speed
#             # 0.8mach以内大于0
#             lam = 0.01
#             reward_v = 1 - np.exp(lam * (ego_v - 272))
#             reward_v = np.clip(reward_v, -1, 1)  # 限制最大奖励
#
#             reward = w[0] * reward_d + w[1] * reward_a + w[2] * reward_hd + w[3] * reward_v
#
#         # self.pre_remaining_missiles[agent_id] = task.remaining_missiles[agent_id]
#         return self._process(reward, agent_id)
