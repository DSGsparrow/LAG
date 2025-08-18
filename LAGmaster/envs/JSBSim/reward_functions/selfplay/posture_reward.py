import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R


class SelfPlayPostureReward(BaseRewardFunction):
    """
    Φ(s) = w_R * f_R(R_km) + w_AO * g_AO(AO)
      - g_AO(AO) = (1 + cos(AO)) / 2 ∈ [0, 1]（AO 越小越好）
      - f_R(R)   = 1 / (1 + exp(beta * (R_km - R_thr_km)))
                   其中 beta 由 f_R(30 km) = p30 反解：beta = ln((1-p30)/p30) / (30 - R_thr_km)

    注意：
      - 本函数返回“当前势能值 Φ(s)”，不是即时奖励。
      - 即时奖励可在外部用差分：r_t = Φ(s_t) - Φ(s_{t-1})
      - 多敌机：取势能最大的那个（避免因敌机数量放大势能）
    """

    def __init__(self, config):
        super().__init__(config)
        # 角度/距离权重
        self.w_R  = float(getattr(config, 'SelfPlayPosturePotential_w_R', 0.7))
        self.w_AO = float(getattr(config, 'SelfPlayPosturePotential_w_AO', 0.3))

        # logistic 参数：20km 为中点；指定 30km 处的目标值 p30（越小→远距斜率越大→更强靠近驱动力）
        self.R_thr_km = float(getattr(config, 'SelfPlayPosturePotential_R_thr_km', 20.0))
        self.p30      = float(getattr(config, 'SelfPlayPosturePotential_p30', 0.1))

        # 记录项（用于你原框架里的可视化/日志）
        self.reward_item_names = [
            self.__class__.__name__,
            f'{self.__class__.__name__}_angle',
            f'{self.__class__.__name__}_range'
        ]

    # ---- 势能项：角度 ----
    @staticmethod
    def g_AO(AO: float) -> float:
        """AO ∈ [0, π]，AO 越小越好；返回 [0,1]."""
        return 0.5 * (1.0 + np.cos(AO))

    # ---- 势能项：距离（logistic）----
    def _beta_from_p30(self) -> float:
        """由 f_R(30 km)=p30 反解 logistic 斜率 beta；中点在 R_thr_km。"""
        p = float(np.clip(self.p30, 1e-6, 1 - 1e-6))
        denom = max(30.0 - self.R_thr_km, 1e-6)
        return np.log((1 - p) / p) / denom

    def f_R(self, R_km: float) -> float:
        """logistic 距离项：R 越小越接近 1；20–30 km 区间有较大斜率以提供靠近驱动力。"""
        beta = self._beta_from_p30()
        return 1.0 / (1.0 + np.exp(beta * (R_km - self.R_thr_km)))

    # ---- 主接口：返回当前势能 Φ(s) ----
    def get_reward(self, task, env, agent_id):
        """
        返回当前势能 Φ(s)；外部若做 shaping，请自己缓存上一步 Φ 做差分。
        """
        ego = env.agents[agent_id]
        ego_feature = np.hstack([ego.get_position(), ego.get_velocity()])

        best_phi = 0.0
        best_angle_term = 0.0
        best_range_term = 0.0

        enemies = getattr(ego, 'enemies', [])
        if not enemies:
            # 无目标时，返回 0（也可返回上一次的 Φ 或一个基线）
            return self._process(0.0, agent_id, (0.0, 0.0))

        for enm in enemies:
            enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
            AO, _, R_m = get_AO_TA_R(ego_feature, enm_feature)  # TA 不用
            R_km = float(R_m) / 1000.0

            angle_term = self.g_AO(AO)    # [0,1]
            range_term = self.f_R(R_km)   # (0,1)

            phi = self.w_AO * angle_term + self.w_R * range_term

            if phi > best_phi:
                best_phi = phi
                best_angle_term = angle_term
                best_range_term = range_term

        # 返回势能与两个分量（便于你在日志里看分解）
        return self._process(best_phi, agent_id, (best_angle_term, best_range_term))
