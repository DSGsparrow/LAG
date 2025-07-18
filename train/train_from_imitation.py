import torch
import numpy as np
from gymnasium import spaces

from net import PPOCustomImitationPolicy
from agent.PPO import PPOAgent
from env_factory.env_factory_from_imitation import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    num_envs = 8
    log_file = "./train/result/train_shoot_back3.log"
    model_path = "./trained_model/imitation_shoot/imitation_pretrained_pytorch.pt"
    save_dir = "./trained_model/ppo_from_imitation"

    # 并行环境
    vec_env = SubprocVecEnv([lambda env_id=i: make_env(env_id, log_file) for i in range(num_envs)])

    # 用于初始化策略的空间信息
    obs_shape = vec_env.observation_space.shape
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    # 创建策略和 PPOAgent
    policy = PPOCustomImitationPolicy(observation_space=observation_space)
    agent = PPOAgent(policy=policy, vec_env=vec_env, model_path=model_path)

    total_steps = 2_000_000
    rollout_len = 2048

    for update in range(total_steps // (rollout_len * num_envs)):
        obs, actions, logp, returns, advantages = agent.collect_rollout(rollout_len)
        agent.ppo_update(obs, actions, logp, returns, advantages)

        if (update + 1) % 10 == 0:
            print(f"✅ PPO迭代 {update+1} 完成")
            torch.save(agent.policy.state_dict(), f"{save_dir}/model_{update+1}.pt")
