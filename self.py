import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import torch
import os
import sys
import geopy.distance
# from stable_baselines3.common.vec_env import DummyVecEnv

from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv, SingleCombatEnvTest
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv
# from envs.env_wrappers import SubprocVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv
from LAGmaster.algorithms.ppo.ppo_policy import PPOPolicy as Policy
from LAGmaster.config import get_config
from LAGmaster.runner.base_runner import Runner, ReplayBuffer

from utils.parse_log_file import parse_log_file



my_aircraft = {
    "lat": 60.0,  # 北纬60度
    "lon": 120.0,  # 东经120度
    "alt": 20000,  # 高度 20000 英尺
    "heading": 0  # 朝向正北（0度）
}


def calculate_enemy_position(distance, angle):
    """根据距离和角度计算敌机的经纬度"""
    origin = (my_aircraft["lat"], my_aircraft["lon"])
    destination = geopy.distance.distance(meters=distance).destination(origin, angle)
    return destination.latitude, destination.longitude  # 纬度，经度


def calculate_bearing(lat1, lon1, lat2, lon2):
    """计算从 (lat1, lon1) 指向 (lat2, lon2) 的方位角"""
    delta_lon = np.radians(lon2 - lon1)
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360  # 转换为 0-360 度范围


def generate_enemy_positions():
    """遍历敌机位置、速度、高度的所有可能情况，并计算其经纬度和朝向"""
    distances = np.linspace(8000, 15000, num=2)  # 8000-15000米
    angles = np.linspace(0, 360, num=3)  # 0-360度
    altitudes = np.linspace(14000, 30000, num=1)  # 14000-30000英尺
    speeds = np.linspace(400, 1000, num=5)  # 400-1000英尺/秒

    enemy_positions = []
    counter = 0
    for distance in distances:
        for angle in angles:
            for altitude in altitudes:
                for speed in speeds:
                    lat, lon = calculate_enemy_position(distance, angle)
                    heading = calculate_bearing(lat, lon, my_aircraft["lat"], my_aircraft["lon"])
                    counter += 1

                    enemy_positions.append({
                        "lat": lat,
                        "lon": lon,
                        "distance": distance,
                        "angle": angle,
                        "alt": altitude,
                        "speed": speed,
                        "heading": heading,  # 敌机朝向我机的角度
                        "counter": counter
                    })
    return enemy_positions

results = []
enemy_positions = generate_enemy_positions()

# 先读文件
parse_log_file(enemy_positions, "./render-result/run.log",
               "./test_result/dodge_test/parsed_results.json",
               "./test_result/dodge_test/parsed_states.json")