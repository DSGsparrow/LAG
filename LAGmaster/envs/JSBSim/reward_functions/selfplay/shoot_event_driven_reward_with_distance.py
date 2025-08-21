from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import numpy as np
import logging


class SelfPlayShootMissileRewardWithDistance(BaseRewardFunction):
    """
    导弹命中/未命中奖励（平滑、全负的未命中版）
    - 命中：+hit_reward
    - 未命中：r_miss(d) ∈ [miss_min_reward, 0)，且
        · 距离 d 越近 → 奖励趋近 0（惩罚小），梯度更大
        · 距离 d 越远 → 奖励趋近 miss_min_reward（更负）
      提供两种形状可选：logistic / exponential
      统一以 half_km（默认 5km）为“半幅点”（奖励等于最小值的一半）
    """

    # ---------- 配置读取 ----------
    def _cfg(self, key: str, default):
        prefixed = getattr(self.config, f'{self.__class__.__name__}_{key}', None)
        if prefixed is not None:
            return prefixed
        plain = getattr(self.config, key, None)
        return default if plain is None else plain

    def __init__(self, config):
        super().__init__(config)

        # 命中奖励
        self.hit_reward = self._cfg("hit_reward", 1.0)

        # 未命中奖励下限（负数，越负表示惩罚越大）
        # 未命中奖励将位于 [miss_min_reward, 0)
        self.miss_min_reward = float(self._cfg("miss_min_reward", -1.0))

        # 形状选择：'logistic' 或 'exp'
        self.miss_shape_type = str(self._cfg("miss_shape_type", "exp")).lower()

        # 半幅点：在 d=half_km 时，未命中奖励 = miss_min_reward 的一半
        self.half_km = float(self._cfg("half_km", 5.0))

        # logistic 的斜率控制（越大越平滑）
        self.distance_steep_km = float(self._cfg("distance_steep_km", 1.0))

        # 距离分箱（量化）步长，抑制小幅波动（km）
        self.distance_bin_km = float(self._cfg("distance_bin_km", 0.3))

        # 判定命中的距离阈值（km），多保留一份备用
        self.hit_distance_km = float(self._cfg("hit_distance_km", 0.3))

        # 防御：half_km、steep 正数
        self.half_km = max(1e-6, self.half_km)
        self.distance_steep_km = max(1e-6, self.distance_steep_km)

    # ---------- 形状函数 ----------
    @staticmethod
    def _logistic01(x, mid, steep):
        """
        标准 [0,1] logistic，x 越小越接近 1：
        s(x) = 1 / (1 + exp((x - mid)/steep))
        """
        return 1.0 / (1.0 + np.exp((x - mid) / max(1e-6, steep)))

    @staticmethod
    def _exp_close01(x, tau):
        """
        指数型接近度：s(x) = exp(-x / tau)，x 越小越接近 1
        其中 tau = half_km / ln(2) 使得在 x=half_km 时 s=0.5
        """
        tau = max(1e-6, tau)
        return np.exp(-x / tau)

    def _miss_reward_from_distance(self, d_km):
        """
        基于距离的未命中奖励（< 0），距离单位 km
        统一公式：r_miss(d) = miss_min_reward * (1 - s_close(d))
        其中 s_close(d)∈(0,1] 为“接近度”：
          - logistic:  s_close(d) = logistic01(d; mid=half_km, steep=distance_steep_km)
          - exp:       s_close(d) = exp(-d / tau), tau = half_km / ln2
        性质：
          d→0   => s_close→1，r→0^-（惩罚最小，梯度最大）
          d→∞   => s_close→0，r→miss_min_reward（趋近最负）
          d=half_km 时 r = miss_min_reward * 0.5（半幅点）
        """
        if not np.isfinite(d_km) or d_km < 0:
            # 距离异常：按“最远”处理 → 最负
            return self.miss_min_reward

        if self.miss_shape_type == "exp":
            # tau 由 half_km 推出：exp(-half_km / tau) = 0.5
            tau = self.half_km / np.log(2.0)
            s_close = self._exp_close01(d_km, tau)
        else:
            # 默认 logistic
            s_close = self._logistic01(d_km, self.half_km, self.distance_steep_km)

        # 将“接近度”映射为全负奖励（0 附近 -> 接近 0^-；远处 -> miss_min_reward）
        r = self.miss_min_reward * (1.0 - s_close)
        return float(r)

    # ---------- 主流程 ----------
    def reset(self, task, env):
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        reward = 0.0
        agent = env.agents[agent_id]

        for missile in getattr(agent, "launch_missiles", []):
            if not missile.is_done:
                continue  # 忽略飞行中的导弹

            if getattr(missile, "is_success", False):
                r = float(self.hit_reward)
                reward += r
                logging.info(f"[HIT] +{r:.3f}")
                continue

            if getattr(missile, "is_miss", False):
                # 终端弹目距离（米 -> 公里）
                d_m = float(getattr(missile, "target_distance", np.nan))
                d_km = d_m / 1000.0 if np.isfinite(d_m) and d_m >= 0 else np.inf

                # 距离分箱（抑制抖动）
                if np.isfinite(d_km):
                    d_km_q = np.round(d_km / self.distance_bin_km) * self.distance_bin_km
                else:
                    d_km_q = d_km

                r = self._miss_reward_from_distance(d_km_q)
                reward += r

                logging.info(
                    f"[MISS-{self.miss_shape_type}] d={d_m:.1f} m (~{d_km_q:.2f} km binned), miss_r={r:.3f}"
                )

        return self._process(reward, agent_id)
