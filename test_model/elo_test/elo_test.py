import os
import numpy as np
import json
from argparse import Namespace

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from env_factory.env_factory_selfplay import make_env


def update_elo(elo_a, elo_b, result_a, k=32):
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    new_elo_a = elo_a + k * (result_a - expected_a)
    new_elo_b = elo_b + k * ((1 - result_a) - (1 - expected_a))
    return new_elo_a, new_elo_b


def eval_and_update_elo(ego_model_path: str, opponent_model_path: str, elo_dict: dict, args) -> dict:
    n_eval_threads = args.eval_threads
    n_eval_episodes = args.eval_episodes

    # 创建评估环境
    env_fns = [lambda i=i: make_env(i, args) for i in range(n_eval_threads)]
    eval_env = SubprocVecEnv(env_fns)

    # 加载模型
    ego_agent = PPO.load(ego_model_path, env=eval_env)
    eval_env.env_method("set_opponent_agent", opponent_model_path)

    # Elo 分数统计
    total_score = 0.0
    count = 0
    result_count = {"win": 0, "lose": 0, "draw_live": 0, "draw_dead": 0}

    for eval_round in range(n_eval_episodes // n_eval_threads):
        obs = eval_env.reset()
        infos = [{} for _ in range(n_eval_threads)]

        for _ in range(1000):  # 给个最大步数上限（防死循环）
            actions, _ = ego_agent.predict(obs)
            obs, rewards, dones, new_infos = eval_env.step(actions)

            for i in range(n_eval_threads):
                if dones[i] and not infos[i]:
                    infos[i] = new_infos[i]

            if all(infos[i] for i in range(n_eval_threads)):
                break

        # 胜负分析
        for info in infos:
            a_success = info.get("A0100 success", False)
            b_success = info.get("B0100 success", False)
            both_alive = info.get("draw and both live", False)
            both_dead = (not both_alive) and (a_success is False) and (b_success is False)

            if a_success is True and b_success is False:
                score = 1.0
                result_count["win"] += 1
            elif a_success is False and b_success is True:
                score = 0.0
                result_count["lose"] += 1
            elif both_alive:
                score = 0.3
                result_count["draw_live"] += 1
            elif both_dead:
                score = 0.7
                result_count["draw_dead"] += 1
            else:
                score = 0.5  # 异常情况 fallback

            total_score += score
            count += 1

    # Elo 更新
    avg_score = total_score / count if count > 0 else 0.5
    ego_elo = elo_dict.get(ego_model_path, 1000)
    opp_elo = elo_dict.get(opponent_model_path, 1000)
    new_ego_elo, new_opp_elo = update_elo(ego_elo, opp_elo, avg_score)

    # 更新 Elo 字典
    elo_dict[ego_model_path] = new_ego_elo
    elo_dict[opponent_model_path] = new_opp_elo

    # 输出日志
    print(f"\n[EVAL] {os.path.basename(ego_model_path)} vs {os.path.basename(opponent_model_path)}")
    print(f"  Win: {result_count['win']}, Lose: {result_count['lose']}, "
          f"Draw(Live): {result_count['draw_live']}, Draw(Dead): {result_count['draw_dead']}")
    print(f"  Avg Score: {avg_score:.3f}, Elo: {ego_elo:.1f} → {new_ego_elo:.1f}")

    return elo_dict


if __name__ == "__main__":
    # ========== 配置参数 ==========
    args = Namespace(
        eval_threads=8,
        eval_episodes=32,
        # 下面是 make_env 需要用到的参数，如果还有其它字段也一并加进去
        log_file="./test_result/log/shoot_selfplay.log",
        config="1v1/ShootMissile/HierarchySelfPlayShoot",
        target_state=0
    )

    # ========== 模型路径 ==========
    ego_model_path = "model_pool/shoot_selfplay_1/model_step_10000.zip"
    opponent_model_path = "model_pool/shoot_selfplay_1/model_step_0.zip"

    # ========== Elo 分数记录文件 ==========
    elo_json_path = "model_pool/elo_scores.json"

    # 读取或初始化 Elo 分数字典
    if os.path.exists(elo_json_path):
        with open(elo_json_path, "r") as f:
            elo_dict = json.load(f)
    else:
        elo_dict = {}

    # 执行对打评估并更新 Elo
    elo_dict = eval_and_update_elo(ego_model_path, opponent_model_path, elo_dict, args)

    # 保存新的 Elo 分数
    with open(elo_json_path, "w") as f:
        json.dump(elo_dict, f, indent=2)

    print("\n✅ Elo 分数已更新并保存到 elo_scores.json")




