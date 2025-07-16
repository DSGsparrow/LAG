
import re
import json
import ast
from pathlib import Path

def extract_episode_stats(log_lines):
    env_stats = {}
    pattern = re.compile(r"\[ENV (\d+)\] - test (\d+) episode, shoot down enemy (\d+) times")
    for line in log_lines:
        match = pattern.search(line)
        if match:
            env_id, total_ep, win_ep = match.groups()
            env_stats[int(env_id)] = {
                "total_episodes": int(total_ep),
                "win_episodes": int(win_ep)
            }
    return env_stats

def extract_failed_states(log_lines, output_file):
    render_pattern = re.compile(r"render_result: ({.*})")
    count = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for line in log_lines:
            match = render_pattern.search(line)
            if match:
                try:
                    render_result = ast.literal_eval(match.group(1))
                    if not render_result.get("success", True) and render_result.get("state") is not None:
                        json.dump(render_result["state"], outfile)
                        outfile.write("\n")
                        count += 1
                except Exception as e:
                    print(f"解析失败: {e}")
    return count

def extract_launch_mission_states(log_file: str, output_file: str):
    """
    从日志文件中提取发射导弹时的状态并将其保存为 JSON 文件。

    :param log_file: 输入日志文件路径
    :param output_file: 输出 JSON 文件路径
    """
    # 字段顺序（必须和state顺序一一对应）
    state_keys = [
        'my_lon', 'my_lat', 'my_alt',
        'my_x', 'my_y', 'my_z',
        'my_vx', 'my_vy', 'my_vz',
        'enemy_lon', 'enemy_lat', 'enemy_alt',
        'enemy_x', 'enemy_y', 'enemy_z',
        'enemy_vx', 'enemy_vy', 'enemy_vz'
    ]

    # 用于匹配包含 state 的 launch mission 行
    state_line_pattern = re.compile(r'state=\[([^\]]+)\]')

    results = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if "A0100 launch mission!" in line and "state=[" in line:
                match = state_line_pattern.search(line)
                if match:
                    # 提取 state 中的数字列表
                    state_str = match.group(1)
                    state_values = [float(x.strip()) for x in state_str.split(',')]

                    if len(state_values) != len(state_keys):
                        print(f"⚠️ 状态字段数量不匹配，跳过此行：{line.strip()}")
                        continue

                    # 构建字典
                    state_dict = dict(zip(state_keys, state_values))
                    results.append(state_dict)

    # 将结果保存为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    print(f"提取成功，已保存至 {output_file}")

# 调用示例
# extract_launch_mission_states("your_log_file.log", "launch_mission_states.json")


def main(log_file_path, output_jsonl_path):
    with open(log_file_path, "r", encoding="utf-8") as f:
        log_data = f.readlines()

    # 功能 1：统计每个 ENV 的胜负情况
    stats = extract_episode_stats(log_data)
    total_ep = sum(stat["total_episodes"] for stat in stats.values())
    total_win = sum(stat["win_episodes"] for stat in stats.values())

    summary = {
        "per_env_stats": stats,
        "total_episodes": total_ep,
        "total_wins": total_win,
        "overall_win_rate": total_win / total_ep if total_ep > 0 else 0
    }

    print(json.dumps(summary, indent=2))

    # 功能 2：提取失败状态
    fail_count = extract_failed_states(log_data, output_jsonl_path)
    print(f"写入失败状态至: {output_jsonl_path}, 共 {fail_count} 条")


if __name__ == "__main__":
    # 修改路径为你自己的文件路径
    main("train/result/train_guide.log", "test_result/result/states_3_dodge2.jsonl")
    # extract_launch_mission_states("test_result/log/test_shoot_imi_vs_dodge2.log",
    #                               "test_result/result/launch_states.json")
