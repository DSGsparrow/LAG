import os
import argparse
import torch
import torch.nn as nn
import logging
import numpy as np
from argparse import Namespace

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
# from adapter.adapter_shoot_back import SB3SingleCombatEnv
from adapter.adapter_dodge_missile import SB3SingleCombatEnv
from net.net_shoot_imitation import CustomImitationPolicy, MLPBase, GRULayer
from utils.shoot_rule import fuzzy_should_attack


# 模仿学习策略网络（直接输出连续动作）
class ImitationPolicy(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.feature_extractor = MLPBase(obs_dim, 128)
        self.gru = GRULayer(128, 128)
        self.action_head = nn.Linear(128, 5)  # 连续动作输出 5维

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = self.gru(feat)
        raw = self.action_head(feat)
        # 前3维 tanh [-1, 1]，第4维 sigmoid * 0.5 + 0.4 => [0.4, 0.9]，第5维 sigmoid => [0,1]
        aileron = torch.tanh(raw[:, 0:1])
        elevator = torch.tanh(raw[:, 1:2])
        rudder = torch.tanh(raw[:, 2:3])
        throttle = torch.sigmoid(raw[:, 3:4]) * 0.5 + 0.4
        shoot = torch.sigmoid(raw[:, 4:5])
        return torch.cat([aileron, elevator, rudder, throttle, shoot], dim=-1)


class EnvIDFilter(logging.Filter):
    def __init__(self, env_id):
        super().__init__()
        self.env_id = env_id

    def filter(self, record):
        record.env_id = f"{self.env_id}"
        return True


def setup_logging(env_id=0, log_file=None):
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    """配置 logging，让日志既输出到终端，又写入文件，标明 ENV ID"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 创建 Filter，用于注入 env_id
    env_filter = EnvIDFilter(env_id)

    # 日志格式带 env_id
    formatter = logging.Formatter("%(asctime)s - %(levelname)s [ENV %(env_id)s] - %(message)s")

    # 终端 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(env_filter)

    # 文件 handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(env_filter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info(f"Logger for ENV {env_id} initialized, log path: {log_file}")


def make_env(env_id, config_name, log_file):
    def _init():
        setup_logging(env_id, log_file)
        env = SB3SingleCombatEnv(
            env_id,
            config_name=config_name,
        )
        return env
    return _init


def load_imitation_model(model_path, obs_dim, device):
    model = ImitationPolicy(obs_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_model(model_path, env, device, obs_dim):
    if model_path.endswith(".zip"):
        print("✅ 加载 SB3 .zip 模型")
        return PPO.load(model_path, env=env, device=device)
    elif model_path.endswith(".pt"):
        print("🔄 加载 .pt 模仿学习特征提取器")
        model = load_imitation_model(model_path, obs_dim, device)
        return model
    else:
        raise ValueError("❌ 不支持的模型文件类型，请提供 .zip 或 .pt 文件")

def main_origin_imitation_vs_dodge(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_fns = [make_env(i, args.config_name, args.log_file) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    obs = env.reset()
    obs_dim = obs.shape[1]
    model = load_model(args.model_path, env, device, obs_dim)

    episode_nums = 0
    win_nums = 0

    # 推理循环
    for step in range(args.max_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions = model(obs_tensor).cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)

        if step % 2_000 == 0:
            print(f"current_step: {step}")

            for env_id in range(args.num_envs):
                episode_num = env.get_attr('episode_num', env_id)
                episode_nums += episode_num[0]
                win_num = env.get_attr('win_num', env_id)
                win_nums += win_num[0]

            print(
                f'test {episode_nums} episode, shoot down enemy {win_nums} times, win rate is{win_nums / (episode_nums + 1e6) * 100}%')

    env.close()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_fns = [make_env(i, args.config_name, args.log_file) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    model = load_model(args.model_path, env, device, 21)

    episode_nums = 0
    win_nums = 0

    obs = env.reset()
    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        # 加上shoot_action

        # 保存


        obs, rewards, dones, infos = env.step(action)

        if step % 2_000 == 0:
            print(f"current_step: {step}")

            for env_id in range(args.num_envs):
                episode_num = env.get_attr('episode_num', env_id)
                episode_nums += episode_num[0]
                win_num = env.get_attr('win_num', env_id)
                win_nums += win_num[0]

            print(
                f'test {episode_nums} episode, shoot down enemy {win_nums} times, win rate is{win_nums / (episode_nums + 1e-6) * 100}%')

    env.close()
    print("🎯 推理完成")


def main_save_npz(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_fns = [make_env(i, args.config_name, args.log_file) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    model = load_model(args.model_path, env, device, 21)

    episode_nums = 0
    win_nums = 0

    obs = env.reset()

    # 👇 保存数据用的列表
    all_obs = []
    all_actions = []
    all_rewards = []

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        raw_obs = obs.copy()  # shape: (num_envs, obs_dim)

        # ✅ 为每个 env 单独计算 shoot_action，并拼接到 action
        extended_actions = []
        for i in range(obs.shape[0]):
            single_obs = raw_obs[i]  # (obs_dim,)

            distance = single_obs[13] * 10000
            angle = single_obs[11] * 180 / np.pi
            speed = np.linalg.norm([single_obs[9], single_obs[10], single_obs[11]]) * 340
            alt_diff = -single_obs[10] * 1000

            shoot_action = fuzzy_should_attack(distance, angle, speed, alt_diff)  # 0 or 1
            shoot_action = np.array([shoot_action], dtype=np.float32)

            full_action = np.concatenate([action[i], shoot_action], axis=0)  # 原动作 + shoot_action
            extended_actions.append(full_action)

        extended_actions = np.array(extended_actions)  # shape: (num_envs, original_action_dim + 1)

        # ✅ 保存
        all_obs.append(raw_obs.copy())
        all_actions.append(extended_actions.copy())

        obs, rewards, dones, infos = env.step(action)
        all_rewards.append(rewards.copy())

        if step % 2_000 == 0:
            print(f"current_step: {step}")

            for env_id in range(args.num_envs):
                episode_num = env.get_attr('episode_num', env_id)
                episode_nums += episode_num[0]
                win_num = env.get_attr('win_num', env_id)
                win_nums += win_num[0]

            print(
                f'test {episode_nums} episode, shoot down enemy {win_nums} times, win rate is {win_nums / (episode_nums + 1e-6) * 100:.2f}%')

    # 🔚 最后补一帧 obs 作为终止状态
    all_obs.append(obs.copy())

    env.close()
    print("🎯 推理完成")

    # ✅ 如果开启保存专家数据
    if getattr(args, 'save_npz', True):
        all_obs_np = np.concatenate(all_obs, axis=0)         # shape: (N, obs_dim)
        all_actions_np = np.concatenate(all_actions, axis=0) # shape: (N, action_dim)
        all_rewards_np = np.concatenate(all_rewards, axis=0) # shape: (N,)

        save_path = f"{args.save_path}/expert_data2.npz"
        np.savez_compressed(save_path,
                            observations=all_obs_np,
                            actions=all_actions_np,
                            rewards=all_rewards_np)
        print(f"✅ 已保存专家数据到: {save_path}")



if __name__ == "__main__":
    # 1, change args: model path, config name, output_dir
    # 2, yaml: render path, baselines
    args3 = Namespace(
        config_name='1v1/DodgeMissile/HierarchyVsBaselineSelf',
        model_path='./trained_model/dodge_missile/ppo_air_combat_dodge4.zip',
        log_file='./test_result/log/test_dodge4_vs_shoot_imi.log',
        save_path='./test_result/expert_data',
        save_npz=True,
        num_envs=1,
        max_steps=50_001,
    )
    main(args3)


    # # 调用第一个模型（模仿学习 .pt）
    # args1 = Namespace(
    #     config_name='1v1/ShootMissile/HierarchyVsBaselineImitationOrigin',
    #     model_path='./trained_model/imitation_shoot/imitation_pretrained_pytorch.pt',
    #     log_file='./test_result/log/test_shoot_imi_origin_vs_dodge2.log',
    #     num_envs=16,
    #     max_steps=50_001,
    # )
    # main_origin_imitation_vs_dodge(args1)
    #
    # # 调用第二个模型（PPO .zip）
    # args2 = Namespace(
    #     config_name='1v1/ShootMissile/HierarchyVsBaselineSelf',
    #     model_path='./trained_model/shoot_missile/ppo_air_combat_3.zip',
    #     log_file='./test_result/log/test_shoot3_vs_dodge2.log',
    #     num_envs=16,
    #     max_steps=50_001
    # )
    # main(args2)