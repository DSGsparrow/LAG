import os
import numpy as np
import random
import math
import copy
from typing import Dict, Any, Tuple
import logging

from torch.utils.tensorboard import SummaryWriter
from collections import deque

from utils.init_state import my_aircraft, calculate_enemy_position

from .env_base import BaseEnv
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask, \
    SingleCombatShootMissileImitationTask
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


def get_unique_filename(env_id, enm_ver, folder="./render_result_random"):
    """
    生成唯一的实验结果文件名，防止覆盖已有文件。

    参数:
    - experiment_name (str): 实验名称
    - folder (str): 存放实验结果的文件夹，默认是 "experiments"

    返回:
    - str: 唯一的实验文件路径
    """
    if not os.path.exists(folder):
        os.makedirs(folder)  # 如果文件夹不存在，则创建它

    txt_name = f'env_{env_id}_{enm_ver}'

    base_filename = os.path.join(folder, txt_name)
    filename = f"{base_filename}.txt.acmi"
    counter = 1

    while os.path.exists(filename):  # 如果文件已存在，递增编号
        filename = f"{base_filename}_{counter}.txt.acmi"
        counter += 1

    return filename


class SingleCombatEnvShoot(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str, env_id):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None
        self.current_enemy = None
        self.env_id = env_id
        # self.enemy_positions = copy.deepcopy(enemy_positions)
        self.cumulative_reward = 0
        self.render_path = getattr(self.config, 'render_path', "render_train/shoot2")
        self.render_file = get_unique_filename(env_id, 'baseline_dodge', self.render_path)

        # === TensorBoard 记录器（env 级别）===
        self.tb_writer = SummaryWriter(log_dir=f"./ppo_fire_debug/env_{env_id}")
        self.tb_step = 0
        self.success_counter = 0
        self.episode_counter = 0
        self.success_queue = deque(maxlen=100)  # 最近100局命中情况

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
        elif taskname == 'singlecombat_shoot_imitation':
            self.task = SingleCombatShootMissileImitationTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def set_random_enemy(self):
        """设置当前敌机信息"""
        init_ego_alt = self.np_random.uniform(14000., 30000.)
        init_ego_speed = self.np_random.uniform(400., 1000.)
        init_enm_alt = self.np_random.uniform(14000., 30000.)
        init_enm_speed = self.np_random.uniform(400., 1000.)
        init_enm_heading = self.np_random.uniform(140., 220.)
        init_enm_distance = self.np_random.uniform(6000., 20000.)
        init_enm_angle = self.np_random.uniform(-30., 30.)

        init_enm_lat, init_enm_lon = calculate_enemy_position(init_enm_distance, init_enm_angle)

        enemy = {
            "ego_alt": init_ego_alt,
            "ego_speed": init_ego_speed,
            "enm_lat": init_enm_lat,
            "enm_lon": init_enm_lon,
            "enm_alt": init_enm_alt,
            "enm_speed": init_enm_speed,
            "enm_heading": init_enm_heading,
            "enm_distance": init_enm_distance,
            "enm_angle": init_enm_angle,
        }

        self.current_enemy = enemy

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.cumulative_reward = 0
        self.set_random_enemy()
        self.reset_simulators(self.current_enemy)
        self.task.reset(self)
        obs = self.get_obs()
        self._create_records = False
        self.render_file = get_unique_filename(self.env_id, 'baseline_dodge', self.render_path)
        return self._pack(obs)

    def reset_simulators(self, enemy):

        # random initial state
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]

        # rand_lat, rand_lon, heading_deg, rand_distance = random_init_state()

        self.init_states[0].update({
            'ic_long_gc_deg': 120.0,  # 经度
            'ic_lat_geod_deg': 60.0,  # 纬度
            'ic_h_sl_ft': enemy["ego_alt"],  # 高度 英尺
            'ic_psi_true_deg': 0,  # 朝向
            'ic_u_fps': enemy["ego_speed"],  # 速度 英尺每秒 243m/s
        })

        self.init_states[1].update({
            'ic_long_gc_deg': enemy['enm_lon'],  # 经度
            'ic_lat_geod_deg': enemy['enm_lat'],  # 纬度
            'ic_h_sl_ft': enemy['enm_alt'],  # 高度 英尺
            'ic_psi_true_deg': enemy['enm_heading'],  # 朝向
            'ic_u_fps': enemy['enm_speed'],  # 速度 英尺每秒 243m/s
        })

        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self.init_states[idx])

        self._tempsims.clear()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.ndarray): the agents' actions, allow opponent's action input

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}
        # apply actions
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        self.task.step(self)

        obs = self.get_obs()

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]

        self.render(mode="txt", filepath=self.render_file, tacview=None)

        ego_done = self._pack(dones)
        ego_reward = self._pack(rewards)

        self.cumulative_reward += ego_reward.item()

        if ego_done:
            shoot_success = info.get("shoot success", False)
            if not shoot_success:
                render_states = self._pack(self.get_states()).reshape(-1, )
                state = {
                    "my_lat": render_states[0], "my_lon": render_states[1], "my_alt": render_states[2],
                    "my_x": render_states[3], "my_y": render_states[4], "my_z": render_states[5],
                    "my_vx": render_states[6], "my_vy": render_states[7], "my_vz": render_states[8],
                    "enemy_lat": render_states[9], "enemy_lon": render_states[10], "enemy_alt": render_states[11],
                    "enemy_x": render_states[12], "enemy_y": render_states[13], "enemy_z": render_states[14],
                    "enemy_vx": render_states[15], "enemy_vy": render_states[16], "enemy_vz": render_states[17],
                }
            else:
                state = None

            total_steps = self.current_step
            result = {"enm_distance": self.current_enemy["enm_distance"],
                      "enm_angle": self.current_enemy["enm_angle"],
                      "enm_alt": self.current_enemy["enm_alt"],
                      "enm_speed": self.current_enemy["enm_speed"],
                      "enm_heading": self.current_enemy["enm_heading"],
                      "ego_alt": self.current_enemy["ego_alt"],
                      "ego_speed": self.current_enemy["ego_speed"],

                      "success": shoot_success,
                      "reward": self.cumulative_reward,
                      "total_steps": total_steps,  # 新增 Total Steps 记录
                      "state": state,
                      }

            # logger = logging.getLogger()  # 继承主 logger
            logging.info("render_result: " + str(result))

            self.tb_writer.add_scalar("success", float(shoot_success), self.tb_step)
            self.tb_writer.add_scalar("episode_reward", self.cumulative_reward, self.tb_step)

            self.success_queue.append(int(shoot_success))
            win_rate = sum(self.success_queue) / len(self.success_queue)
            self.tb_writer.add_scalar("win_rate", win_rate, self.tb_step)

        self.tb_step += 1

        return self._pack(obs), ego_reward, ego_done, info
