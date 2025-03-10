import numpy as np
import random
import math
import copy

from .env_base import BaseEnv
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask
from ..human_task.HumanSingleCombatTask import  HumanSingleCombatTask


def random_init_state(radius_inner = 9000, radius_outer = 14000):
    # 地球半径，单位为米
    EARTH_RADIUS = 6371000

    # 中心点经纬度
    center_lat = 60.0  # 北纬60度
    center_lon = 120.0  # 东经120度

    # 圆环半径范围 单位是米
    # radius_inner = 9000  # 内环半径
    # radius_outer = 14000  # 外环半径

    # 随机生成在圆环内的距离
    rand_distance = random.uniform(radius_inner, radius_outer)

    # 随机生成方位角（0-360度）
    rand_bearing = random.uniform(0, 360)

    # 将角度转换为弧度
    bearing_rad = math.radians(rand_bearing)
    lat_rad = math.radians(center_lat)
    lon_rad = math.radians(center_lon)

    # 根据大地测量公式计算随机点的经纬度
    rand_lat_rad = math.asin(math.sin(lat_rad) * math.cos(rand_distance / EARTH_RADIUS) +
                             math.cos(lat_rad) * math.sin(rand_distance / EARTH_RADIUS) * math.cos(bearing_rad))

    rand_lon_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(rand_distance / EARTH_RADIUS) * math.cos(lat_rad),
        math.cos(rand_distance / EARTH_RADIUS) - math.sin(lat_rad) * math.sin(rand_lat_rad))

    # 弧度转换回角度
    rand_lat = math.degrees(rand_lat_rad)
    rand_lon = math.degrees(rand_lon_rad)

    # 简单平面假设下计算航向角（从随机点指向圆心，正北为0°）
    delta_lat = center_lat - rand_lat
    delta_lon = (center_lon - rand_lon) * math.cos(math.radians(center_lat))
    heading_rad = math.atan2(delta_lon, delta_lat)
    heading_deg = (math.degrees(heading_rad) + 360) % 360

    # 输出结果
    # print(f"随机点纬度: {rand_lat:.6f}°")
    # print(f"随机点经度: {rand_lon:.6f}°")
    # print(f"随机点与圆心的距离: {rand_distance:.2f}米")
    # print(f"飞机航向角（正北为0°）: {heading_deg:.2f}°")

    return rand_lat, rand_lon, heading_deg, rand_distance


class SingleCombatEnvTest(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str, enemy_positions):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None
        self.current_enemy = None
        self.enemy_positions = copy.deepcopy(enemy_positions)

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'hierarchical_singlecombat':
            self.task = HierarchicalSingleCombatTask(self.config)
        elif taskname == 'singlecombat_dodge_missile':
            self.task = SingleCombatDodgeMissileTask(self.config)
        elif taskname == 'singlecombat_shoot':
            self.task = SingleCombatShootMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_dodge_missile':
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_shoot':
            self.task = HierarchicalSingleCombatShootTask(self.config)
        elif taskname == 'HumanSingleCombat':
            self.task = HumanSingleCombatTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def set_enemy(self, enemy):
        """设置当前敌机信息"""
        self.current_enemy = enemy

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.current_enemy = self.enemy_positions[0]
        self.reset_simulators(self.current_enemy)
        self.enemy_positions.pop(0)
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self, enemy):
        # # switch side
        # if self.init_states is None:
        #     self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        # # self.init_states[0].update({
        # #     'ic_psi_true_deg': (self.np_random.uniform(270, 540))%360,
        # #     'ic_h_sl_ft': self.np_random.uniform(17000, 23000),
        # # })
        # init_states = self.init_states.copy()
        # self.np_random.shuffle(init_states)
        # for idx, sim in enumerate(self.agents.values()):
        #     sim.reload(init_states[idx])

        # random initial state
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        # init_heading = self.np_random.uniform(0., 180.)
        # init_altitude = self.np_random.uniform(14000., 30000.)
        # init_velocities_u = self.np_random.uniform(400., 1000.)

        rand_lat, rand_lon, heading_deg, rand_distance = random_init_state()

        self.init_states[0].update({
            'ic_long_gc_deg': 120.0,  # 经度
            'ic_lat_geod_deg': 60.0,  # 纬度
            'ic_h_sl_ft': 20000,  # 高度 英尺
            'ic_psi_true_deg': 0,  # 朝向
            'ic_u_fps': 800.0,  # 速度 英尺每秒 243m/s
        })

        # enemy_positions.append({
        #     "lat": lat,
        #     "lon": lon,
        #     "distance": distance,
        #     "angle": angle,
        #     "alt": altitude,
        #     "speed": speed,
        #     "heading": heading  # 敌机朝向我机的角度
        # })

        self.init_states[1].update({
            'ic_long_gc_deg': enemy['lon'],  # 经度
            'ic_lat_geod_deg': enemy['lat'],  # 纬度
            'ic_h_sl_ft': enemy['alt'],  # 高度 英尺
            'ic_psi_true_deg': enemy['heading'],  # 朝向
            'ic_u_fps': enemy['speed'],  # 速度 英尺每秒 243m/s
        })

        # for init_state in self.init_states:
        #     init_state.update({
        #         'ic_psi_true_deg': init_heading,
        #         'ic_h_sl_ft': init_altitude,
        #         'ic_u_fps': init_velocities_u,
        #         'target_heading_deg': init_heading,
        #         'target_altitude_ft': init_altitude,
        #         'target_velocities_u_mps': init_velocities_u * 0.3048,
        #     })

        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self.init_states[idx])

        self._tempsims.clear()
