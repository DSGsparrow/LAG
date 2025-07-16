from test_model.elo_test.elo_test import eval_and_update_elo

import os
import re
import json
from tqdm import tqdm
from elo_test import eval_and_update_elo
from env_factory.env_factory_selfplay import make_env
from argparse import Namespace

# === 提取模型训练步数 ===
def extract_step(path):
    match = re.search(r"model_step_(\d+)", path)
    return int(match.group(1)) if match else -1

# === 收集所有模型路径（附带偏移步数） ===
def collect_all_models(folders_with_offset):
    """
    folders_with_offset: list of (folder_path, offset)
    Returns: list of (global_step, model_path)
    """
    model_entries = []
    for folder, offset in folders_with_offset:
        for file in os.listdir(folder):
            if file.endswith(".zip") and "model_step_" in file:
                full_path = os.path.join(folder, file)
                step = extract_step(full_path)
                if step >= 0:
                    model_entries.append((step + offset, full_path))
    return sorted(model_entries, key=lambda x: x[0])

# === 去除重复 step，仅保留第一个出现的模型（例如避免 shoot_selfplay_2 的0步和1的130000重复） ===
def deduplicate_models_by_step(models):
    seen = set()
    unique = []
    for step, path in models:
        if step not in seen:
            unique.append((step, path))
            seen.add(step)
    return unique

def main():
    # ==== 配置评估参数 ====
    args = Namespace(
        eval_threads=8,
        eval_episodes=32,
        # 下面是 make_env 需要用到的参数，如果还有其它字段也一并加进去
        log_file="./test_result/log/shoot_selfplay.log",
        config="1v1/ShootMissile/HierarchySelfPlayShoot",
        target_state=0
    )

    # ==== 模型目录及步数偏移 ====
    folders_with_offset = [
        ("model_pool/shoot_selfplay_1", 0),         # 正常命名
        ("model_pool/shoot_selfplay_2", 130000),    # 偏移130000
    ]

    # ==== 收集模型并去重 ====
    all_models = collect_all_models(folders_with_offset)
    all_models = deduplicate_models_by_step(all_models)  # list of (step, path)

    # ==== 初始化 Elo 表 ====
    elo_dict = {}
    elo_curve = {}

    # ==== 遍历模型，逐步评估 ====
    for i, (step, ego_model_path) in enumerate(tqdm(all_models, desc="Evaluating models")):
        history_models = all_models[:i]
        if not history_models:
            elo_dict[ego_model_path] = 1000
            elo_curve[step] = 1000
            continue

        for _, opp_model_path in history_models:
            elo_dict = eval_and_update_elo(ego_model_path, opp_model_path, elo_dict, args)

        # 当前模型评估完成后，记录 Elo 分数
        elo_curve[step] = elo_dict.get(ego_model_path, 1000)

    # ==== 保存 Elo 数据 ====
    os.makedirs("elo_result", exist_ok=True)

    with open("elo_result/elo_curve.json", "w") as f:
        json.dump(elo_curve, f, indent=2)

    with open("elo_result/elo_scores.json", "w") as f:
        json.dump(elo_dict, f, indent=2)

    print("\n✅ 所有模型评估完成，结果已保存到 elo_result/ 目录")

if __name__ == "__main__":
    main()
