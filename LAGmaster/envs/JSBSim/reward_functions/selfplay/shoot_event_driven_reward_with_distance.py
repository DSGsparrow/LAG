from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import numpy as np
import logging


class SelfPlayShootMissileRewardWithDistance(BaseRewardFunction):
    """
    导弹命中或未命中的奖励函数（平滑且抗抖动版）
    - 命中目标：+1.0
    - 未命中目标：-0.75 + 距离相关奖励（最大 +0.75，最终 reward ∈ [-0.75, +0.75]）
    - 距离奖励采用 logistic 形状，并对距离做分箱以减小抖动
    """

    def __init__(self, config):
        super().__init__(config)
        # ---- 可调参数（都给默认值，拿不到 config 时也能跑）----
        # 训练早期距离常在 6~8 km，mid 设在 7 km，steep 适当放缓以减少对小抖动的敏感度
        self.hit_reward = config.get("hit_reward", 1.0)
        self.miss_base_penalty = config.get("miss_base_penalty", -0.75)
        self.miss_bonus_max = config.get("miss_bonus_max", 0.75)   # 未命中最多补到 +0.75

        # logistic 距离 shaping 参数
        self.distance_mid_km = config.get("distance_mid_km", 7.0)   # R=0.5 的中点（建议 6.5~8 之间调）
        self.distance_steep_km = config.get("distance_steep_km", 1.2)  # 越大越平滑（建议 0.8~2.0）

        # 距离分箱（量化）步长，抑制小幅波动（例如雷达/仿真噪声）
        self.distance_bin_km = config.get("distance_bin_km", 0.3)   # 200 m 一档（可调 0.1~0.5）

    def reset(self, task, env):
        return super().reset(task, env)

    @staticmethod
    def _logistic01(x, mid, steep):
        """
        标准 [0,1] logistic： 1 / (1 + exp((x - mid)/steep))
        x, mid, steep 都是标量或 ndarray，单位需一致
        """
        return 1.0 / (1.0 + np.exp((x - mid) / max(1e-6, steep)))

    def get_reward(self, task, env, agent_id):
        reward = 0.0
        agent = env.agents[agent_id]

        for missile in agent.launch_missiles:
            if not missile.is_done:
                continue  # 忽略未结束的导弹

            if missile.is_success:
                # 命中：给定高奖励
                r = self.hit_reward
                reward += r
                logging.debug(f"[HIT] +{r:.3f}")
                continue

            if missile.is_miss:
                # 未命中：基础惩罚 + 距离形状奖励（平滑+分箱）
                base_penalty = self.miss_base_penalty

                # 终端弹目距离（米 -> 公里）
                d_m = float(getattr(missile, "target_distance", np.nan))
                if not np.isfinite(d_m) or d_m < 0:
                    # 防御性处理：距离异常时不给额外奖励
                    d_km = np.inf
                else:
                    d_km = d_m / 1000.0

                # 距离分箱（抑制小波动）：例如 0.2km 为一档
                if np.isfinite(d_km):
                    d_km_q = np.round(d_km / self.distance_bin_km) * self.distance_bin_km
                else:
                    d_km_q = d_km  # inf 保持 inf

                # logistic 形状：距离越小越接近 1，越大越接近 0
                shape = 0.0 if not np.isfinite(d_km_q) else self._logistic01(
                    d_km_q, self.distance_mid_km, self.distance_steep_km
                )

                # 补偿奖励幅度控制（最大不超过 miss_bonus_max）
                add = self.miss_bonus_max * shape

                r = base_penalty + add
                reward += r

                logging.debug(
                    f"[MISS] d={d_m:.1f} m (~{d_km_q:.2f} km binned), "
                    f"shape={shape:.3f}, add={add:.3f}, miss_r={r:.3f}"
                )

        return self._process(reward, agent_id)
