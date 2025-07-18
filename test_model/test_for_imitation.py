import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from env_factory.env_factory_from_imitation import make_env  # 你自己的封装路径
from tqdm import tqdm

def test_dodge_agent(model_path, num_envs=8, num_episodes=20, log_file=None):
    # 创建并行环境
    vec_env = SubprocVecEnv([lambda env_id=i: make_env(env_id, log_file) for i in range(num_envs)])

    # 加载模型
    model = PPO.load(model_path)

    episode_rewards = [[] for _ in range(num_envs)]
    episode_counts = [0] * num_envs
    total_episodes = 0

    obs = vec_env.reset()
    pbar = tqdm(total=num_episodes)

    while total_episodes < num_episodes:
        actions, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(actions)

        for i in range(num_envs):
            episode_rewards[i].append(rewards[i])
            if dones[i]:
                total_episodes += 1
                pbar.update(1)
                ep_reward = sum(episode_rewards[i])
                print(f"[Env {i}] Episode {episode_counts[i]+1} finished, reward: {ep_reward:.2f}")
                episode_counts[i] += 1
                episode_rewards[i] = []  # 清空该环境的记录

    vec_env.close()
    pbar.close()


if __name__ == "__main__":
    model_path = "trained_model/dodge_missile/ppo_air_combat_dodge4.zip"
    log_file = "./test_result/log/test_for_imitation.log"
    test_dodge_agent(model_path, num_envs=16, num_episodes=20000, log_file=log_file)
