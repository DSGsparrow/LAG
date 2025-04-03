import re
import os
import json
import ast
from collections import deque, defaultdict

# ✅ 你的路径配置
log_path = "train/result/train_shoot_imi2.log"
output_dir = "test_result/shoot_imi2"
os.makedirs(output_dir, exist_ok=True)

with open(log_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# ✅ 增量写入的文件路径
episode_file = os.path.join(output_dir, "episode_data.jsonl")
acmi_file_path = os.path.join(output_dir, "acmi_results.jsonl")

success_last_100 = deque(maxlen=100)
success_count = 0

pid_buffer = defaultdict(dict)

def safe_parse_result(block):
    try:
        # 替换 Python 风格为 JSON 合法格式
        block = block.replace("None", "null").replace("True", "true").replace("False", "false")

        # 处理 numpy 的 array(...) 表达形式为 JSON 数组
        block = re.sub(
            r"array\(\s*\[(.*?)\]\s*\)",
            lambda m: "[" + " ".join(m.group(1).replace("\n", " ").split()) + "]",
            block,
            flags=re.DOTALL
        )

        # 删除数组中数字后多余的空格（如 1. , → 1.0）
        block = re.sub(r"(\d)\s*\.", r"\1.0", block)

        # 将所有 key 和字符串值用双引号包裹（避免不合法引号）
        block = re.sub(r"(?<=[:,{])\s*'([^']*)'\s*:", r'"\1":', block)  # keys
        block = re.sub(r":\s*'([^']*)'", r':"\1"', block)              # values

        return json.loads(block)
    except Exception as e:
        print(f"[!] fallback ast 失败: {e}\n原始内容:\n{block}\n")
        try:
            return ast.literal_eval(block)
        except Exception as e2:
            print(f"[!] ast 解析失败: {e2}\n")
            return None

for idx, line in enumerate(lines):
    pid_match = re.search(r"\[ENV (\d+)\]", line)
    if not pid_match:
        continue
    pid = int(pid_match.group(1))
    pid_buffer[pid]["env_id"] = pid

    # ✅ 提取 acmi 文件和 env_id
    if "render txt name:" in line:
        acmi_match = re.search(r"render txt name:(.*?env_(\d+)_.*?)\.acmi", line)
        if acmi_match:
            acmi_file = acmi_match.group(1).replace("\\", "/") + ".acmi"
            env_id = int(acmi_match.group(2))
            pid_buffer[pid]["acmi_file"] = acmi_file
            pid_buffer[pid]["env_id"] = env_id

    # ✅ 提取发射信息（多行）
    if "A0100 launch mission!" in line:
        block = line
        for j in range(1, 10):
            if idx + j < len(lines):
                block += lines[idx + j]
            if "current_reward=" in block:
                break
        if "obs=" in line and "state=" in line and "current_reward=" in line:
        #     match = True
        # else:
        #     match = False
        # match = re.search(r"obs=\[(.*?)\].*?state=\[(.*?)\].*?current_reward=([-\d.eE]+)", block, re.DOTALL)
        # if match:
            obs_match = re.search(r"obs=\[(.*?)\]", line)
            state_match = re.search(r"state=\[(.*?)\]", line)
            reward_match = re.search(r"current_reward=([-\d\.eE]+)", line)

            obs = ast.literal_eval("[" + obs_match.group(1) + "]") if obs_match else None
            state = ast.literal_eval("[" + state_match.group(1) + "]") if state_match else None
            reward = float(reward_match.group(1)) if reward_match else None

    # ✅ 提取 render_result（多行）
    if "render_result:" in line:
        block = line.split("render_result:", 1)[1]
        for j in range(1, 10):
            if block.count("{") > 0 and block.count("{") == block.count("}"):
                break
            if idx + j < len(lines):
                block += lines[idx + j]

        result_data = safe_parse_result(block)
        if result_data is None:
            print(f"[PID {pid}] render_result 解析失败，内容如下：\n{block}\n")
            continue

        buf = pid_buffer.get(pid, {})
        acmi_path = buf.get("acmi_file", f"unknown_env_pid_{pid}.acmi")
        env_id = buf.get("env_id", -1)

        episode = {
            "env_id": env_id,
            "pid": pid,
            "acmi_file": acmi_path,
            "success": result_data.get("success", False),
            "final_reward": result_data.get("total_reward", 0),
            "total_steps": result_data.get("total_steps", 0)
        }

        if "launch_obs" in buf:
            episode.update({
                "launch_obs": buf["launch_obs"],
                "launch_state": buf["launch_state"],
                "reward_at_launch": buf["reward_at_launch"],
                "reward_diff": result_data.get("total_reward", 0) - buf["reward_at_launch"]
            })
        else:
            episode["reward_diff"] = result_data.get("total_reward", 0)

        if result_data.get("success") is True:
            episode["final_state"] = {
                k: result_data[k]
                for k in ["enm_distance", "enm_angle", "enm_alt", "enm_speed", "enm_heading", "ego_alt", "ego_speed"]
                if k in result_data
            }
            success_count += 1
            success_last_100.append(1)
        else:
            success_last_100.append(0)

        # ✅ 增量保存到主文件（jsonl 格式）
        with open(episode_file, "a", encoding="utf-8") as ef:
            ef.write(json.dumps(episode) + "\n")

        with open(acmi_file_path, "a", encoding="utf-8") as af:
            af.write(json.dumps({
                "env_id": env_id,
                "acmi_file": acmi_path,
                "success": result_data.get("success", False)
            }) + "\n")

        pid_buffer[pid].clear()

# ✅ 成功率统计输出
total = len(success_last_100)
rate_total = success_count / total if total > 0 else 0
rate_100 = sum(success_last_100) / len(success_last_100) if success_last_100 else 0

print(f"✅ 成功回合数: {success_count}")
print(f"✅ 成功率（整体）: {rate_total:.2%}")
print(f"✅ 成功率（最近100）: {rate_100:.2%}")
