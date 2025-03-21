from stable_baselines3 import PPO
import gym
import torch
import numpy as np
from env.env_shoot_missile import SB3SingleCombatEnv


def run_model_test(model_path, env_config, num_episodes=10, render=False):
    """
    测试 PPO 模型在指定环境中的表现。

    参数：
    - model_path: 模型文件路径
    - env_config: 用于初始化环境的配置（你自定义的 SB3SingleCombatEnv 参数）
    - num_episodes: 测试回合数
    - render: 是否渲染环境
    """

    print(f"🚀 开始测试模型：{model_path}，测试回合数：{num_episodes}")

    # 创建单个测试环境（不用并行）
    env = SB3SingleCombatEnv(env_id=0, config_name=env_config)

    # 加载模型（注意要设置 env）
    model = PPO.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if render:
                env.render()

        episode_rewards.append(total_reward)
        print(f"🎯 Episode {episode + 1}: Reward = {total_reward:.2f}")

    env.close()

    avg_reward = np.mean(episode_rewards)
    print(f"\n✅ 测试完成！平均奖励：{avg_reward:.2f}")
    return avg_reward


if __name__ == "__main__":
    avg = run_model_test(
        model_path="model/shoot_missile/ppo_air_combat.zip",
        env_config='1v1/ShootMissile/HierarchyVsBaseline_self',
        num_episodes=5,
        render=False  # 如果你的环境支持图形渲染，可以设为 True
    )
