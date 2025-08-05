import os
import torch
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn

from torch.utils.tensorboard import SummaryWriter

from net.net_imitation_for_ppo import PPOGRUPolicy, PPOMLPPolicy, MLPPolicy, GRUPolicy
from env_factory.env_factory_from_imitation import make_env


def main_gru(
    num_envs=8,
    total_steps=2_000_000,
    rollout_len=2048,
    log_dir="./train/result/gru_tensorboard",
    model_path="./trained_model/imitation_shoot/imitation_pretrained_pytorch.pt",
    save_dir="./trained_model/ppo_from_imitation",
    log_file="./train/result/train_shoot_back3.log"
):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 并行环境 + VecMonitor 封装
    vec_env = SubprocVecEnv([lambda i=i: make_env(i, log_file) for i in range(num_envs)])
    vec_env = VecMonitor(vec_env)

    # 2. 构建 observation_space（SB3 需要）
    obs_shape = vec_env.observation_space.shape  # 应该是 [T, obs_dim]
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    # 3. 构建 policy_kwargs 并加载模仿模型参数
    # policy_kwargs = dict(observation_space=observation_space)
    # policy = PPOGRUPolicy(**policy_kwargs)
    # policy.load_state_dict(torch.load(model_path), strict=False)
    # 1. 加载模仿学习模型
    imitation_model = GRUPolicy(obs_dim=obs_shape[0], act_dim=2)  # 根据实际模型修改
    imitation_model.load_state_dict(torch.load(model_path))

    # 2. 初始化 PPO 策略
    policy_kwargs = dict(observation_space=observation_space)
    ppo_policy = PPOGRUPolicy(**policy_kwargs)

    # 3. 拆分迁移 feature_extractor、gru、actor
    with torch.no_grad():
        # 迁移 MLP feature extractor
        for p_tgt, p_src in zip(ppo_policy.feature_extractor.parameters(),
                                imitation_model.feature_extractor.parameters()):
            p_tgt.copy_(p_src)

        # 迁移 GRU 层
        for p_tgt, p_src in zip(ppo_policy.gru.parameters(), imitation_model.gru.parameters()):
            p_tgt.copy_(p_src)

        # 迁移 actor 输出层
        for p_tgt, p_src in zip(ppo_policy.actor.parameters(), imitation_model.actor.parameters()):
            p_tgt.copy_(p_src)

    # 4. 构建 SB3 的 PPO 模型（接入自定义策略）
    model = PPO(
        policy=PPOGRUPolicy,
        env=vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
    )

    # 5. 将预训练参数手动赋值到 SB3 模型中
    model.policy.load_state_dict(ppo_policy.state_dict(), strict=False)

    # 6. TensorBoard 记录器
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # 7. Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=rollout_len * num_envs * 10,  # 每10轮保存一次
        save_path=save_dir,
        name_prefix="ppo_gru",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # 8. 训练开始
    model.learn(
        total_timesteps=total_steps,
        callback=checkpoint_callback
    )

    # 9. 保存最终模型
    final_model_path = os.path.join(save_dir, "ppo_gru_final.zip")
    model.save(final_model_path)
    print(f"✅ PPO-GRU 强化学习完成，模型保存至: {final_model_path}")


def main_mlp(
    num_envs=1,
    total_steps=2_000_000,
    rollout_len=2048,
    log_dir="./train/result/mlp_tensorboard",
    model_path="./trained_model/imitation_shoot/mlp_imitation_policy.pt",
    save_dir="./trained_model/ppo_from_mlp",
    log_file="./train/result/train_shoot_back_mlp.log"
):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 多环境创建与封装
    vec_env = SubprocVecEnv([lambda i=i: make_env(i, log_file) for i in range(num_envs)])
    vec_env = VecMonitor(vec_env)

    # 2. 构建 observation_space
    obs_shape = vec_env.observation_space.shape
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    # 3. 加载模仿学习模型
    imitation_model = MLPPolicy(obs_dim=obs_shape[0], act_dim=4)
    imitation_model.load_state_dict(torch.load(model_path))

    # 4. 构建 PPO 策略并迁移权重
    # 构造必要参数
    policy_kwargs = dict()
    ppo_policy = PPOMLPPolicy(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        lr_schedule=get_schedule_fn(3e-4),  # PPO 默认学习率
        **policy_kwargs
    )

    with torch.no_grad():
        for p_tgt, p_src in zip(ppo_policy.feature_extractor.parameters(),
                                imitation_model.feature_extractor.parameters()):
            p_tgt.copy_(p_src)

        for p_tgt, p_src in zip(ppo_policy.actor.parameters(),
                                imitation_model.actor.parameters()):
            p_tgt.copy_(p_src)

    # 5. 创建 SB3 的 PPO 模型
    model = PPO(
        policy=PPOMLPPolicy,
        env=vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
    )
    model.policy.load_state_dict(ppo_policy.state_dict(), strict=False)

    # 6. 配置 Logger 和 Callback
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=rollout_len * num_envs * 10,
        save_path=save_dir,
        name_prefix="ppo_mlp",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # 7. 启动训练
    model.learn(
        total_timesteps=total_steps,
        callback=checkpoint_callback
    )

    final_path = os.path.join(save_dir, "ppo_mlp_final.zip")
    model.save(final_path)
    print(f"✅ PPO-MLP 强化训练完成，模型已保存到: {final_path}")


if __name__ == "__main__":
    main_mlp()

    # num_envs = 8
    # log_file = "./train/result/train_shoot_back3.log"
    # model_path = "./trained_model/imitation_shoot/imitation_pretrained_pytorch.pt"
    # save_dir = "./trained_model/ppo_from_imitation"
    #
    # # 并行环境
    # vec_env = SubprocVecEnv([lambda env_id=i: make_env(env_id, log_file) for i in range(num_envs)])
    #
    # # 用于初始化策略的空间信息
    # obs_shape = vec_env.observation_space.shape
    # observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
    #
    # # 创建策略和 PPOAgent
    # policy = PPOGRUPolicy(observation_space=observation_space)
    # agent = PPOAgent(policy=policy, vec_env=vec_env, model_path=model_path)
    #
    # total_steps = 2_000_000
    # rollout_len = 2048
    #
    # for update in range(total_steps // (rollout_len * num_envs)):
    #     obs, actions, logp, returns, advantages = agent.collect_rollout(rollout_len)
    #     agent.ppo_update(obs, actions, logp, returns, advantages)
    #
    #     if (update + 1) % 10 == 0:
    #         print(f"✅ PPO迭代 {update+1} 完成")
    #         torch.save(agent.policy.state_dict(), f"{save_dir}/model_{update+1}.pt")
