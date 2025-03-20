import json
import numpy as np
from LAGmaster.envs.JSBSim.utils import utils


def evaluate_situation(result_file, output_file, w_hit=1.0, w_posture=0.8, w_alt=0.6, w_time=0.5, w_reward=0.3):
    """
    计算战场态势评分，并进行归一化处理

    :param result_file: 输入的 JSON 文件
    :param output_file: 输出的 JSON 文件
    :param w_hit: 命中惩罚权重
    :param w_posture: 态势奖励权重
    :param w_alt: 高度权重
    :param w_time: 躲弹时长权重
    :param w_reward: 试验奖励权重
    """
    T_max = 302  # 最大躲弹时间步长
    AO_max = np.pi
    TA_max = np.pi

    # **定义姿态奖励函数**
    def posture_reward(AO, TA):
        return 1 / (50 * AO / np.pi + 2) + 1 / 2 + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5

    max_posture_score = 1.5  # posture_reward(AO_max, TA_max)  # **计算最大可能的姿态评分**

    with open(result_file, 'r') as f:
        results = [json.loads(line) for line in f]

    all_S_time, all_S_alt, all_S_posture, all_S_reward = [], [], [], []

    evaluations = []

    for entry in results:
        # **1. 命中惩罚**
        S_hit = 1 if not entry["success"] else 0

        # **2. 躲弹时长评分**
        total_steps = entry["total_steps"]
        S_time = max(0, T_max - total_steps)
        all_S_time.append(S_time)

        # **3. 降高评分**
        if entry["state"]:  # 仅当成功时计算
            H_start = entry["alt"] * 0.3408
            H_end = entry["state"]["my_z"]
            S_alt = max(0, H_start - H_end)
        else:
            S_alt = 0
        all_S_alt.append(S_alt)

        # **4. 终态劣势评分**
        if entry["state"]:
            ego_feature = [entry["state"]["my_x"], entry["state"]["my_y"], entry["state"]["my_z"],
                           entry["state"]["my_vx"], entry["state"]["my_vy"], entry["state"]["my_vz"]]
            enm_feature = [entry["state"]["enemy_x"], entry["state"]["enemy_y"], entry["state"]["enemy_z"],
                           entry["state"]["enemy_vx"], entry["state"]["enemy_vy"], entry["state"]["enemy_vz"]]
            ego_AO, ego_TA, R, side_flag = utils.get_AO_TA_R(ego_feature, enm_feature, return_side=True)

            # AO = np.arctan2(entry["state"]["my_vy"], entry["state"]["my_vx"])  # 计算终态攻角
            # TA = np.arctan2(entry["state"]["enemy_y"] - entry["state"]["my_y"],
            #                 entry["state"]["enemy_x"] - entry["state"]["my_x"])  # 计算终态转向角
            final_posture_score = posture_reward(ego_AO, ego_TA)
            S_posture = max_posture_score - final_posture_score
        else:
            S_posture = max_posture_score  # 如果被击中，按最差情况计算
        all_S_posture.append(S_posture)

        # **5. 奖励**
        S_reward = entry["reward"]
        all_S_reward.append(S_reward)

        evaluations.append({
            "S_hit": S_hit,
            "S_time": S_time,
            "S_alt": S_alt,
            "S_posture": S_posture,
            "S_reward": S_reward,
            "original_entry": entry
        })

        if entry["success"]:
            bp = 0

    # **6. 归一化**
    def normalize(data):
        min_val, max_val = min(data), max(data)
        if max_val - min_val == 0:
            return [0] * len(data)  # 如果所有值相同，全部归零
        return [(x - min_val) / (max_val - min_val) for x in data]

    norm_S_time = normalize(all_S_time)
    norm_S_alt = normalize(all_S_alt)
    norm_S_posture = normalize(all_S_posture)
    norm_S_reward = normalize(all_S_reward)

    # **7. 计算归一化后总态势分数**
    for i, entry in enumerate(evaluations):
        S_total = (w_hit * entry["S_hit"] +
                   w_posture * norm_S_posture[i] +
                   w_alt * norm_S_alt[i] +
                   w_time * norm_S_time[i] +
                   w_reward * norm_S_reward[i])

        entry["original_entry"]["situation_score"] = S_total
        # 改成相对高度
        entry["original_entry"]["alt"] = entry["original_entry"]["alt"] - 20000

    # **8. 保存新的 JSON 文件**
    with open(output_file, 'w') as f:
        for item in evaluations:
            json.dump(item["original_entry"], f)
            f.write("\n")

    print(f"战场态势评分计算完成，结果已保存至 {output_file}")

# 运行示例
a = np.arctanh(1.-1e-4)
evaluate_situation("./test_result/dodge_test/parsed_results_all.json", "./test_result/dodge_test/evaluated_results_all.json")
# evaluate_situation("./test_result/dodge_test/parsed_results2.json", "./test_result/dodge_test/evaluated_result2.json")
