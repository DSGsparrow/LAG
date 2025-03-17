import re
import json


def parse_log_file(enemy_positions, log_file, output_file, state_file):
    success_pattern = re.compile(r'A0100 step limits! Total Steps=1000')
    failure_pattern = re.compile(r'A0100 has been shot down!|B0100 mission completed!')
    reward_pattern = re.compile(r'render episode reward of agent: \[\[\[(.*?)\]\]\]')
    state_start_pattern = re.compile(r'missile down states \[')
    state_end_pattern = re.compile(r'\]')

    with open(log_file, 'r') as file:
        lines = file.readlines()

    success = None  # 记录当前回合的状态
    reward = None  # 记录当前回合的奖励
    state = None  # 记录当前回合的状态
    state_buffer = []  # 用于处理换行的状态
    capturing_state = False  # 记录是否在捕获状态数据

    # # 去除前 5012 行
    # lines = lines[5135:]

    for line in lines:
        if capturing_state:
            state_buffer.append(line.strip())
            if state_end_pattern.search(line):  # 结束状态捕获
                capturing_state = False
                state_str = ' '.join(state_buffer)  # 重新组合成一行
                state_values = list(map(float, re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?', state_str)))
                state_values = state_values[-12:]  # 只取最后 12 个数值，过滤掉前面的时间信息
                if len(state_values) == 12:
                    state = {
                        "my_x": state_values[0], "my_y": state_values[1], "my_z": state_values[2],
                        "my_vx": state_values[3], "my_vy": state_values[4], "my_vz": state_values[5],
                        "enemy_x": state_values[6], "enemy_y": state_values[7], "enemy_z": state_values[8],
                        "enemy_vx": state_values[9], "enemy_vy": state_values[10], "enemy_vz": state_values[11]
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
                'counter': enemy["counter"],
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

            success = None  # 重置状态
            reward = None  # 重置奖励
            state = None  # 重置状态信息

    return enemy['counter']


def parse_log_file_gap(enemy_positions, log_file, output_file, state_file):
    success_pattern = re.compile(r'A0100 step limits! Total Steps=1000')
    failure_pattern = re.compile(r'A0100 has been shot down!|B0100 mission completed!')
    reward_pattern = re.compile(r'render episode reward of agent: \[\[\[(.*?)\]\]\]')
    state_start_pattern = re.compile(r'missile down states \[')
    state_end_pattern = re.compile(r'\]')

    with open(log_file, 'r') as file:
        lines = file.readlines()

    success = None  # 记录当前回合的状态
    reward = None  # 记录当前回合的奖励
    state = None  # 记录当前回合的状态
    state_buffer = []  # 用于处理换行的状态
    capturing_state = False  # 记录是否在捕获状态数据

    # # 去除前 5012 行
    # lines = lines[5135:]

    for line in lines:
        if capturing_state:
            state_buffer.append(line.strip())
            if state_end_pattern.search(line):  # 结束状态捕获
                capturing_state = False
                state_str = ' '.join(state_buffer)  # 重新组合成一行
                state_values = list(map(float, re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?', state_str)))
                state_values = state_values[-12:]  # 只取最后 12 个数值，过滤掉前面的时间信息
                if len(state_values) == 12:
                    state = {
                        "my_x": state_values[0], "my_y": state_values[1], "my_z": state_values[2],
                        "my_vx": state_values[3], "my_vy": state_values[4], "my_vz": state_values[5],
                        "enemy_x": state_values[6], "enemy_y": state_values[7], "enemy_z": state_values[8],
                        "enemy_vx": state_values[9], "enemy_vy": state_values[10], "enemy_vz": state_values[11]
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
                'counter': enemy["counter"],
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

            success = None  # 重置状态
            reward = None  # 重置奖励
            state = None  # 重置状态信息

    return enemy['counter']


if __name__ == '__main__':
    # 示例使用
    log_file_path = "run.log"  # 你的log文件路径
    output_file_path = "parsed_results.json"  # 输出文件路径
    parse_log_file(log_file_path, output_file_path)

    # 提示已完成解析
    print(f"解析完成，结果已存入 {output_file_path}")