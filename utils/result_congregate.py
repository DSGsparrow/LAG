import re
import json
import numpy as np
import geopy.distance


def parse_log_file(enemy_positions, log_file, output_file, state_file):
    success_pattern = re.compile(r'A0100 dodge succeeded!|missile dodge missile success True')
    failure_pattern = re.compile(r'A0100 has been shot down!|B0100 mission completed!|missile dodge missile success False')
    reward_pattern = re.compile(r'render episode reward of agent: \[\[\[(.*?)\]\]\]')
    state_start_pattern = re.compile(r'missile down states \[')
    state_end_pattern = re.compile(r'\]')
    end_pattern = re.compile(r'init complete')
    steps_pattern = re.compile(r"Total Steps=(\d+)")  # 提取Total Steps的正则

    with open(log_file, 'r') as file:
        lines = file.readlines()

    success = None  # 记录当前回合的状态
    reward = None  # 记录当前回合的奖励
    state = None  # 记录当前回合的状态
    state_buffer = []  # 用于处理换行的状态
    capturing_state = False  # 记录是否在捕获状态数据
    total_steps = None  # 记录Total Steps

    end_counter = 0

    for line in lines:
        # 捕获Total Steps
        steps_match = steps_pattern.search(line)
        if steps_match:
            total_steps = int(steps_match.group(1))  # 转换为整数

        if capturing_state:
            state_buffer.append(line.strip())
            if state_end_pattern.search(line):  # 结束状态捕获
                capturing_state = False
                state_str = ' '.join(state_buffer)  # 重新组合成一行
                state_values = list(map(float, re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?', state_str)))
                state_values = state_values[-18:]  # 只取最后 12 个数值，过滤掉前面的时间信息
                if len(state_values) == 18:
                    state = {
                        "my_lat": state_values[0], "my_lon": state_values[1], "my_alt": state_values[2],
                        "my_x": state_values[3], "my_y": state_values[4], "my_z": state_values[5],
                        "my_vx": state_values[6], "my_vy": state_values[7], "my_vz": state_values[8],
                        "enemy_lat": state_values[9], "enemy_lon": state_values[10], "enemy_alt": state_values[11],
                        "enemy_x": state_values[12], "enemy_y": state_values[13], "enemy_z": state_values[14],
                        "enemy_vx": state_values[15], "enemy_vy": state_values[16], "enemy_vz": state_values[17]
                    }
                state_buffer = []  # 清空缓存

        if state_start_pattern.search(line):  # 开始状态捕获
            capturing_state = True
            state_buffer = [line.strip()]  # 重新初始化 buffer

        if success_pattern.search(line):
            success = True
        elif failure_pattern.search(line):
            success = False

        match = reward_pattern.search(line)
        if match:
            enemy = enemy_positions[0]
            enemy_positions.pop(0)

            reward = float(match.group(1))
            # **确保状态信息存在，否则存入 None**
            result = {
                "distance": enemy["distance"],
                "angle": enemy["angle"],
                "alt": enemy["alt"],
                "speed": enemy["speed"],
                "success": success,
                "reward": reward,
                "total_steps": total_steps,  # 新增 Total Steps 记录
                "counter": enemy["counter"],
                "state": state,
            }

            # 追加写入 JSON 文件，避免数据丢失
            with open(output_file, 'a') as f:
                json.dump(result, f)
                f.write('\n')  # 每行存储一个 JSON 对象

            # 追加写入状态文件
            if state is not None:
                with open(state_file, 'a') as f:
                    json.dump(state, f)
                    f.write('\n')  # 每行存储一个 JSON 对象

            # 重置变量
            success = None
            reward = None
            state = None
            total_steps = None  # 清空 Total Steps，避免跨回合污染数据

        # end_match = end_pattern.search(line)
        # if end_match:
        #     end_counter += 1
        #     if end_counter > 1:
        #         break

    return enemy["counter"]


def parse_log_file2(enemy_positions, log_file, output_file, state_file):
    success_pattern = re.compile(r'A0100 dodge succeeded!|missile dodge missile success True')
    failure_pattern = re.compile(
        r'A0100 has been shot down!|B0100 mission completed!|missile dodge missile success False')
    reward_pattern = re.compile(r'render episode reward of agent: \[\[\[(.*?)\]\]\]')
    state_start_pattern = re.compile(r'missile down states \[')
    state_end_pattern = re.compile(r'\]')
    end_pattern = re.compile(r'init complete')
    steps_pattern = re.compile(r"Total Steps=(\d+)")  # 提取Total Steps的正则

    with open(log_file, 'r') as file:
        lines = file.readlines()

    # **跳过前 5135 行**
    # lines = lines[5135:]

    success = None  # 记录当前回合的状态
    reward = None  # 记录当前回合的奖励
    state = None  # 记录当前回合的状态
    state_buffer = []  # 用于处理换行的状态
    capturing_state = False  # 记录是否在捕获状态数据
    total_steps = None  # 记录Total Steps
    start_recording = False  # 只有 `end_pattern` 出现后才开始记录

    for line in lines:
        # **检测 end_pattern，第一次出现后才开始记录**
        if not start_recording and end_pattern.search(line):
            start_recording = True
            continue  # 跳过第一次出现 end_pattern 的行

        if not start_recording:
            match = reward_pattern.search(line)
            if match:
                enemy_positions.pop(0)
            continue  # 直到第一次 end_pattern 之前都跳过

        # 捕获Total Steps
        steps_match = steps_pattern.search(line)
        if steps_match:
            total_steps = int(steps_match.group(1))  # 转换为整数

        if capturing_state:
            state_buffer.append(line.strip())
            if state_end_pattern.search(line):  # 结束状态捕获
                capturing_state = False
                state_str = ' '.join(state_buffer)  # 重新组合成一行
                state_values = list(map(float, re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?', state_str)))
                state_values = state_values[-18:]  # 只取最后 12 个数值，过滤掉前面的时间信息
                if len(state_values) == 18:
                    state = {
                        "my_lat": state_values[0], "my_lon": state_values[1], "my_alt": state_values[2],
                        "my_x": state_values[3], "my_y": state_values[4], "my_z": state_values[5],
                        "my_vx": state_values[6], "my_vy": state_values[7], "my_vz": state_values[8],
                        "enemy_lat": state_values[9], "enemy_lon": state_values[10], "enemy_alt": state_values[11],
                        "enemy_x": state_values[12], "enemy_y": state_values[13], "enemy_z": state_values[14],
                        "enemy_vx": state_values[15], "enemy_vy": state_values[16], "enemy_vz": state_values[17]
                    }
                state_buffer = []  # 清空缓存

        if state_start_pattern.search(line):  # 开始状态捕获
            capturing_state = True
            state_buffer = [line.strip()]  # 重新初始化 buffer

        if success_pattern.search(line):
            success = True
        elif failure_pattern.search(line):
            success = False

        match = reward_pattern.search(line)
        if match:
            enemy = enemy_positions[0]
            enemy_positions.pop(0)

            reward = float(match.group(1))
            # **确保状态信息存在，否则存入 None**
            result = {
                "distance": enemy["distance"],
                "angle": enemy["angle"],
                "alt": enemy["alt"],
                "speed": enemy["speed"],
                "success": success,
                "reward": reward,
                "total_steps": total_steps,  # 新增 Total Steps 记录
                "counter": enemy["counter"],
                "state": state,
            }

            # 追加写入 JSON 文件，避免数据丢失
            with open(output_file, 'a') as f:
                json.dump(result, f)
                f.write('\n')  # 每行存储一个 JSON 对象

            # 追加写入状态文件
            if state is not None:
                with open(state_file, 'a') as f:
                    json.dump(state, f)
                    f.write('\n')  # 每行存储一个 JSON 对象

            # 重置变量
            success = None
            reward = None
            state = None
            total_steps = None  # 清空 Total Steps，避免跨回合污染数据

    return enemy["counter"]


def filter_json(input_file, output_file, skip_lines=17949):
    """
    读取 JSON 文件，跳过前 skip_lines 行，并将剩余数据写入新的 JSON 文件。

    :param input_file: 输入 JSON 文件路径
    :param output_file: 输出 JSON 文件路径
    :param skip_lines: 需要跳过的行数（默认为 17949）
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()  # 读取所有行

    # 过滤掉前 skip_lines 行
    filtered_lines = lines[skip_lines:]

    # 写入新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(filtered_lines)

    print(f"处理完成，剩余 {len(filtered_lines)} 行数据已写入 {output_file}")


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
    distances = np.linspace(8000, 15000, num=20)  # 8000-15000米
    angles = np.linspace(0, 360, num=36)  # 0-360度
    altitudes = np.linspace(14000, 30000, num=10)  # 14000-30000英尺
    speeds = np.linspace(500, 1000, num=8)  # 400-1000英尺/秒

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


if __name__ == "__main__":
    enemy_positions = generate_enemy_positions()

    enemy_positions = enemy_positions[42601:]

    # 先读文件
    counter = parse_log_file(enemy_positions, "./render-result/run3.log",
                             "./test_result/dodge_test/parsed_results3_total.json",
                             "./test_result/dodge_test/parsed_states3_total.json")

    # filter_json("./test_result/dodge_test/parsed_results.json", "./test_result/dodge_test/parsed_results1.json", 17949)
    # filter_json("./test_result/dodge_test/parsed_states.json", "./test_result/dodge_test/parsed_states1.json", 17741)


