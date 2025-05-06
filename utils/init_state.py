import numpy as np
import random
import math
import geopy.distance
from geopy.distance import geodesic


# 设置参数
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


if __name__ == "__main__":
    lat, lon = calculate_enemy_position(10000, 0)
    distance = geodesic((lat, lon), (60, 120)).meters
    print(distance)


def random_init_state(radius_inner = 9000, radius_outer = 14000):
    # 地球半径，单位为米
    EARTH_RADIUS = 6371000

    # 中心点经纬度
    center_lat = 60.0  # 北纬60度
    center_lon = 120.0  # 东经120度

    # 圆环半径范围（转换为米）
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