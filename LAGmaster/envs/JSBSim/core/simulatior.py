import os
import logging
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from typing import Literal, Union, List

import jsbsim
from .catalog import Property, Catalog
from ..utils.utils import get_root_dir, LLA2NEU, NEU2LLA

TeamColors = Literal["Red", "Blue", "Green", "Violet", "Orange"]


class BaseSimulator(ABC):

    def __init__(self, uid: str, color: TeamColors, dt: float):
        """Constructor. Creates an instance of simulator, initialize all the available properties.

        Args:
            uid (str): 5-digits hexadecimal numbers for unique identification.
            color (TeamColors): use different color strings to represent diferent teams
            dt (float): simulation timestep. Default = `1 / 60`.
        """
        self.__uid = uid
        self.__color = color
        self.__dt = dt
        self.model = ""
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)
        logging.debug(f"{self.__class__.__name__}:{self.__uid} is created!")

    @property
    def uid(self) -> str:
        return self.__uid

    @property
    def color(self) -> str:
        return self.__color

    @property
    def dt(self) -> float:
        return self.__dt

    def get_geodetic(self):
        """(lontitude, latitude, altitude), unit: °, m"""
        return self._geodetic

    def get_position(self):
        """(north, east, up), unit: m"""
        return self._position

    def get_rpy(self):
        """(roll, pitch, yaw), unit: rad"""
        return self._posture

    def get_velocity(self):
        """(v_north, v_east, v_up), unit: m/s"""
        return self._velocity

    def reload(self):
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)

    @abstractmethod
    def run(self, **kwargs):
        pass

    def log(self):
        lon, lat, alt = self.get_geodetic()
        roll, pitch, yaw = self.get_rpy() * 180 / np.pi
        log_msg = f"{self.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
        log_msg += f"Name={self.model.upper()},"
        log_msg += f"Color={self.color}"
        return log_msg

    @abstractmethod
    def close(self):
        pass

    def __del__(self):
        logging.debug(f"{self.__class__.__name__}:{self.uid} is deleted!")


class AircraftSimulator(BaseSimulator):
    """A class which wraps an instance of JSBSim and manages communication with it.
    """

    ALIVE = 0
    CRASH = 1       # low altitude / extreme state / overload
    SHOTDOWN = 2    # missile attack

    def __init__(self,
                 uid: str = "A0100",
                 color: TeamColors = "Red",
                 model: str = 'f16',
                 init_state: dict = {},
                 origin: tuple = (120.0, 60.0, 0.0),
                 sim_freq: int = 60, **kwargs):
        """Constructor. Creates an instance of JSBSim, loads an aircraft and sets initial conditions.

        Args:
            uid (str): 5-digits hexadecimal numbers for unique identification. Default = `"A0100"`.
            color (TeamColors): use different color strings to represent diferent teams
            model (str): name of aircraft to be loaded. Default = `"f16"`.
                model path: './data/aircraft_name/aircraft_name.xml'
            init_state (dict): dict mapping properties to their initial values. Input empty dict to use a default set of initial props.
            origin (tuple): origin point (longitude, latitude, altitude) of the Global Combat Field. Default = `(120.0, 60.0, 0.0)`
            sim_freq (int): JSBSim integration frequency. Default = `60`.
        """
        super().__init__(uid, color, 1 / sim_freq)
        self.model = model
        self.init_state = init_state
        self.lon0, self.lat0, self.alt0 = origin
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        for key, value in kwargs.items():
            if key == 'num_missiles':
                self.num_missiles = value  # type: int
                self.num_left_missiles = self.num_missiles  # type: int
            if key == 'static_mode':
                self.static_mode = value  # type: bool
        # fixed simulator links
        self.partners = []  # type: List[AircraftSimulator]
        self.enemies = []   # type: List[AircraftSimulator]
        # temp simulator links
        self.launch_missiles = []   # type: List[MissileSimulator]
        self.under_missiles = []    # type: List[MissileSimulator]
        # initialize simulator
        self.reload()

    @property
    def is_alive(self):
        return self.__status == AircraftSimulator.ALIVE

    @property
    def is_crash(self):
        return self.__status == AircraftSimulator.CRASH

    @property
    def is_shotdown(self):
        return self.__status == AircraftSimulator.SHOTDOWN

    def crash(self):
        self.__status = AircraftSimulator.CRASH

    def shotdown(self):
        self.__status = AircraftSimulator.SHOTDOWN

    def reload(self, new_state: Union[dict, None] = None, new_origin: Union[tuple, None] = None):
        """Reload aircraft simulator
        """
        super().reload()

        # reset temp simulator links
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        self.launch_missiles.clear()
        self.under_missiles.clear()
        self.num_left_missiles = self.num_missiles

        # load JSBSim FDM
        self.jsbsim_exec = jsbsim.FGFDMExec(os.path.join(get_root_dir(), 'data'))
        self.jsbsim_exec.set_debug_level(0)
        self.jsbsim_exec.load_model(self.model)
        Catalog.add_jsbsim_props(self.jsbsim_exec.query_property_catalog(""))
        self.jsbsim_exec.set_dt(self.dt)
        self.clear_defalut_condition()

        # assign new properties
        if new_state is not None:
            self.init_state = new_state
        if new_origin is not None:
            self.lon0, self.lat0, self.alt0 = new_origin
        for key, value in self.init_state.items():
            self.set_property_value(Catalog[key], value)
        success = self.jsbsim_exec.run_ic()
        if not success:
            raise RuntimeError("JSBSim failed to init simulation conditions.")

        # propulsion init running
        propulsion = self.jsbsim_exec.get_propulsion()
        n = propulsion.get_num_engines()
        for j in range(n):
            propulsion.get_engine(j).init_running()
        propulsion.get_steady_state()
        # update inner property
        self._update_properties()

    def clear_defalut_condition(self):
        default_condition = {
            Catalog.ic_long_gc_deg: 120.0,  # geodesic longitude [deg]
            Catalog.ic_lat_geod_deg: 60.0,  # geodesic latitude  [deg]
            Catalog.ic_h_sl_ft: 20000,      # altitude above mean sea level [ft]
            Catalog.ic_psi_true_deg: 0.0,   # initial (true) heading [deg] (0, 360)
            Catalog.ic_u_fps: 800.0,        # body frame x-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_v_fps: 0.0,          # body frame y-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_w_fps: 0.0,          # body frame z-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_p_rad_sec: 0.0,      # roll rate  [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_q_rad_sec: 0.0,      # pitch rate [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_r_rad_sec: 0.0,      # yaw rate   [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_roc_fpm: 0.0,        # initial rate of climb [ft/min]
            Catalog.ic_terrain_elevation_ft: 0,
        }
        for prop, value in default_condition.items():
            self.set_property_value(prop, value)

    def run(self):
        """Runs JSBSim simulation until the agent interacts and update custom properties.

        JSBSim monitors the simulation and detects whether it thinks it should
        end, e.g. because a simulation time was specified. False is returned
        if JSBSim termination criteria are met.

        Returns:
            (bool): False if sim has met JSBSim termination criteria else True.
        """
        if self.is_alive:
            if self.bloods <= 0:
                self.shotdown()
            if self.static_mode:
                return True
            result = self.jsbsim_exec.run()
            if not result:
                raise RuntimeError("JSBSim failed.")
            self._update_properties()
            return result
        else:
            return True

    def close(self):
        """ Closes the simulation and any plots. """
        if self.jsbsim_exec:
            self.jsbsim_exec = None
        self.partners = []
        self.enemies = []

    def _update_properties(self):
        # update position
        self._geodetic[:] = self.get_property_values([
            Catalog.position_long_gc_deg,
            Catalog.position_lat_geod_deg,
            Catalog.position_h_sl_m
        ])
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)
        # update posture
        self._posture[:] = self.get_property_values([
            Catalog.attitude_roll_rad,
            Catalog.attitude_pitch_rad,
            Catalog.attitude_heading_true_rad,
        ])
        # update velocity
        self._velocity[:] = self.get_property_values([
            Catalog.velocities_v_north_mps,
            Catalog.velocities_v_east_mps,
            Catalog.velocities_v_down_mps,
        ])
        # v_down -> v_up
        self._velocity[2] = -self._velocity[2]

    def get_sim_time(self):
        """ Gets the simulation time from JSBSim, a float. """
        return self.jsbsim_exec.get_sim_time()

    def get_property_values(self, props):
        """Get the values of the specified properties

        :param props: list of Properties

        : return: NamedTupl e with properties name and their values
        """
        return [self.get_property_value(prop) for prop in props]

    def set_property_values(self, props, values):
        """Set the values of the specified properties

        :param props: list of Properties

        :param values: list of float
        """
        if not len(props) == len(values):
            raise ValueError("mismatch between properties and values size")
        for prop, value in zip(props, values):
            self.set_property_value(prop, value)

    def get_property_value(self, prop):
        """Get the value of the specified property from the JSBSim simulation

        :param prop: Property

        :return : float
        """
        if isinstance(prop, Property):
            if prop.access == "R":
                if prop.update:
                    prop.update(self)
            return self.jsbsim_exec.get_property_value(prop.name_jsbsim)
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def set_property_value(self, prop, value):
        """Set the values of the specified property

        :param prop: Property

        :param value: float
        """
        # set value in property bounds
        if isinstance(prop, Property):
            if value < prop.min:
                value = prop.min
            elif value > prop.max:
                value = prop.max

            self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)

            if "W" in prop.access:
                if prop.update:
                    prop.update(self)
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def check_missile_warning(self):
        for missile in self.under_missiles:
            if missile.is_alive:
                return missile
        return None

    def check_missile_launching(self):
        for missile in self.launch_missiles:
            if missile.is_alive:
                return missile
        return None


class MissileSimulator(BaseSimulator):

    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator, uid: str, missile_model: str = "AIM-9L"):
        assert parent.dt == target.dt, "integration timestep must be same!"
        missile = MissileSimulator(uid, parent.color, missile_model, parent.dt)
        missile.launch(parent)
        missile.target(target)

        parent.num_left_missiles -= 1

        # 把自己加到敌机的under_missile里
        return missile

    def __init__(self,
                 uid="A0101",
                 color="Red",
                 model="AIM-9L",
                 dt=1 / 12):
        super().__init__(uid, color, dt)
        self.__status = MissileSimulator.INACTIVE
        self.model = model
        self.parent_aircraft = None  # type: AircraftSimulator
        self.target_aircraft = None  # type: AircraftSimulator
        self.render_explosion = False

        # missile parameters (for AIM-9L)
        self._g = 9.81      # gravitational acceleration
        self._t_max = 60    # time limitation of missile life
        self._t_thrust = 5  # time limitation of engine
        self._Isp = 120     # average specific impulse
        self._Length = 2.87
        self._Diameter = 0.127
        self._cD = 0.4      # aerodynamic drag factor
        self._m0 = 84       # mass, unit: kg
        self._dm = 6        # mass loss rate, unit: kg/s
        self._K = 3         # proportionality constant of proportional navigation
        self._nyz_max = 30  # max overload
        self._Rc = 300      # radius of explosion, unit: m
        self._v_min = 150   # minimun velocity, unit: m/s

    @property
    def is_alive(self):
        """Missile is still flying"""
        return self.__status == MissileSimulator.LAUNCHED

    @property
    def is_success(self):
        """Missile has hit the target"""
        return self.__status == MissileSimulator.HIT

    @property
    def is_miss(self):
        return self.__status == MissileSimulator.MISS

    @property
    def is_done(self):
        """Missile is already exploded"""
        return self.__status == MissileSimulator.HIT \
            or self.__status == MissileSimulator.MISS

    @property
    def Isp(self):
        return self._Isp if self._t < self._t_thrust else 0

    @property
    def K(self):
        """Proportional Guidance Coefficient"""
        # return self._K
        return max(self._K * (self._t_max - self._t) / self._t_max, 0)

    @property
    def S(self):
        """Cross-Sectional area, unit m^2"""
        S0 = np.pi * (self._Diameter / 2)**2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        """Air Density, unit: kg/m^3"""
        # approximate expression
        return 1.225 * np.exp(-self._geodetic[-1] / 9300)
        # exact expression (Reference: https://www.cnblogs.com/pathjh/p/9127352.html)
        rho0, T0, h = 1.225, 288.15, self._geodetic[-1]
        if h <= 11000:  # Troposphere
            T = T0 - 0.0065 * h
            return rho0 * (T / T0)**4.25588
        elif h <= 20000:  # Lower Stratosphere
            T = 216.65
            return 0.36392 * np.exp((11000 - h) / 6341.62)
        else:  # Upper Stratosphere
            T = 216.65 + 0.001 * (h - 20000)
            return 0.088035 * (T / 216.65)**(-35.1632)

    @property
    def target_distance(self) -> float:
        return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())

    def launch(self, parent: AircraftSimulator):
        # inherit kinetic parameters from parent aricraft
        self.parent_aircraft = parent
        self.parent_aircraft.launch_missiles.append(self)
        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()
        self._velocity[:] = parent.get_velocity()
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0  # missile's roll remains zero
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0
        # init status
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self.__status = MissileSimulator.LAUNCHED
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(5 / self.dt))  # 5s of distance increment -- can't hit
        self._left_t = int(1 / self.dt)  # remove missile 1s after its destroying

    def target(self, target: AircraftSimulator):
        self.target_aircraft = target  # TODO: change target?
        self.target_aircraft.under_missiles.append(self)

    def run(self):
        self._t += self.dt
        action, distance = self._guidance()
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance
        if distance < self._Rc:  # and self.target_aircraft.is_alive:
            self.target_aircraft.shotdown()
            self.__status = MissileSimulator.HIT
        elif (self._t > self._t_max) or (np.linalg.norm(self.get_velocity()) < self._v_min and self._t > self._t_thrust) \
                or np.sum(self._distance_increment) >= self._distance_increment.maxlen:  # or not self.target_aircraft.is_alive:
            self.__status = MissileSimulator.MISS
        else:
            self._state_trans(action)

    def log(self):
        if self.is_alive:
            log_msg = super().log()
        elif self.is_done and (not self.render_explosion):
            self.render_explosion = True
            # remove missile model
            log_msg = f"-{self.uid}\n"
            # add explosion
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
        else:
            log_msg = None
        return log_msg

    def close(self):
        self.target_aircraft = None

    def _guidance(self):
        """
        Guidance law, proportional navigation
        """
        x_m, y_m, z_m = self.get_position()
        dx_m, dy_m, dz_m = self.get_velocity()
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])
        theta_m = np.arcsin(dz_m / v_m)
        x_t, y_t, z_t = self.target_aircraft.get_position()
        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()
        Rxy = np.linalg.norm([x_m - x_t, y_m - y_t])  # distance from missile to target project to X-Y plane
        Rxyz = np.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m])  # distance from missile to target
        # calculate beta & eps, but no need actually...
        # beta = np.arctan2(y_m - y_t, x_m - x_t)  # relative yaw
        # eps = np.arctan2(z_m - z_t, np.linalg.norm([x_m - x_t, y_m - y_t]))  # relative pitch
        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / Rxy**2
        deps = ((dz_t - dz_m) * Rxy**2 - (z_t - z_m) * (
            (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz**2 * Rxy)
        ny = self.K * v_m / self._g * np.cos(theta_m) * dbeta
        nz = self.K * v_m / self._g * deps + np.cos(theta_m)
        return np.clip([ny, nz], -self._nyz_max, self._nyz_max), Rxyz

    def _state_trans(self, action):
        """
        State transition function
        """
        # update position & geodetic
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)
        # update velocity & posture
        v = np.linalg.norm(self.get_velocity())
        theta, phi = self.get_rpy()[1:]
        T = self._g * self.Isp * self._dm
        D = 0.5 * self._cD * self.S * self.rho * v**2
        nx = (T - D) / (self._m * self._g)
        ny, nz = action

        dv = self._g * (nx - np.sin(theta))
        self._dphi = self._g / v * (ny / np.cos(theta))
        self._dtheta = self._g / v * (nz - np.cos(theta))

        v += self.dt * dv
        phi += self.dt * self._dphi
        theta += self.dt * self._dtheta
        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),
            v * np.cos(theta) * np.sin(phi),
            v * np.sin(theta)
        ])
        self._posture[:] = np.array([0, theta, phi])
        # update mass
        if self._t < self._t_thrust:
            self._m = self._m - self.dt * self._dm

    @property
    def is_near_exhaustion(self):
        """综合判断导弹是否快要耗尽能量"""
        # dry_mass = self._m0 - self._dm * self._t_thrust  # 计算干质量

        # v = np.linalg.norm(self.get_velocity())

        return (
                # self._t >= (self._t_thrust * 0.9)  # 飞行时间接近推力结束
                # or self.Isp < (self._Isp * 0.1)  # 剩余推力过小
                # or self._m <= dry_mass * 1.1  # 质量接近干质量
                (self._t > self._t_thrust and np.linalg.norm(self.get_velocity()) < self._v_min * 1.2)  # 速度下降
                or self.K < (self._K * 0.2)  # 机动能力下降 48s之后
        )


# class BVRMissileSimulator(BaseSimulator):
#     INACTIVE = -1
#     LAUNCHED = 0
#     HIT = 1
#     MISS = 2
#
#     @classmethod
#     def create(cls, parent: AircraftSimulator, target: AircraftSimulator, uid: str, missile_model: str = "AIM-9L"):
#         assert parent.dt == target.dt, "integration timestep must be same!"
#         missile = MissileSimulator(uid, parent.color, missile_model, parent.dt)
#         missile.launch(parent)
#         missile.target(target)
#         # 把自己加到敌机的under_missile里
#         return missile
#
#     def __init__(self,
#                  uid="A0101",
#                  color="Red",
#                  model="AIM-9L",
#                  dt=1 / 12):
#         # 动力系统增强
#         self._t_max = 100  # 最大飞行时间100秒
#         self._t_thrust = 15  # 发动机工作时间15秒（双脉冲发动机）
#         self._Isp = 280  # 比冲提升到280s
#         self._m0 = 160  # 初始质量160kg
#         self._dm = 8  # 燃料消耗率8kg/s
#
#         # 空气动力学优化（低空平飞）
#         self._cD = 0.25  # 阻力系数降低
#         self._v_min = 250  # 最小有效速度提升
#         self._Rc = 400  # 杀伤半径
#
#         # 超视距制导参数
#         self._midcourse_flag = True  # 中段制导阶段
#         self._data_link_interval = 2  # 数据链更新间隔秒数
#
#     def launch(self, parent: AircraftSimulator):
#         # 继承载机水平速度，保持平飞初始姿态
#         self._posture[1] = parent.get_rpy()[1]  # 俯仰角与载机一致
#         self._velocity[2] = 0  # 垂直速度归零
#
#     def _guidance(self):
#         if self._midcourse_flag:
#             # 中段制导：惯性导航+数据链修正
#             if self.target_distance < 15000:  # 进入末段距离
#                 self._midcourse_flag = False
#             return self._midcourse_guidance()
#         else:
#             # 末段主动雷达制导
#             return self._terminal_guidance()
#
#     def _midcourse_guidance(self):
#         """中段能量最优平飞策略"""
#         # 数据链周期性更新目标信息（仿真简化实现）
#         if int(self._t) % self._data_link_interval == 0:
#             self.target_aircraft.update_position()  # 假设有目标位置更新方法
#
#         # 保持高度策略
#         current_alt = self._geodetic[2]
#         alt_error = parent.launch_alt - current_alt  # 维持发射高度
#         pitch_cmd = np.clip(alt_error * 0.005, -0.1, 0.1)  # 小角度俯仰调整
#
#         return [0, pitch_cmd]  # 无偏航指令，仅高度微调
#
#     def _terminal_guidance(self):
#         # 预测目标运动（二阶预测模型）
#         t_go = self.target_distance / np.linalg.norm(self.get_velocity())
#         target_acc = self.target_aircraft.get_acceleration()  # 需要目标加速度信息
#         predicted_pos = self.target_aircraft.get_position() + \
#                         self.target_aircraft.get_velocity() * t_go + \
#                         0.5 * target_acc * t_go ** 2
#
#         # 计算视线角变化率
#         LOS = predicted_pos - self.get_position()
#         LOS_prev = self._prev_LOS  # 需要保存上一周期LOS
#         LOS_rate = (LOS - LOS_prev) / self.dt
#         self._prev_LOS = LOS
#
#         # 生成过载指令
#         acc_cmd = self.K * LOS_rate * np.linalg.norm(self.get_velocity())
#         return np.clip(acc_cmd / 9.81, -self._nyz_max, self._nyz_max)
#
#     @property
#     def S(self):
#         """优化低空平飞气动面积"""
#         base_area = np.pi * (self._Diameter / 2) ** 2
#         # 平飞时弹体投影面积最小化
#         return base_area * (1 + 0.1 * abs(np.sin(self._posture[1]))
#
#                             @ property
#
#     def rho(self):
#         """低空空气密度优化计算（<10km）"""
#         # 使用国际标准大气模型（ISA）低空简化版
#         return 1.225 * (288.15 / (288.15 - 0.0065 * self._geodetic[2])) ** 4.25588

