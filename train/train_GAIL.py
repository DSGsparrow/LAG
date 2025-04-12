import os
import argparse
import torch
import torch.nn as nn
import logging
import numpy as np
from argparse import Namespace

from imitation.data.types import TrajectoryWithRew
from stable_baselines3.common.vec_env import VecEnvWrapper
from imitation.data import rollout
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util.util import make_vec_env

from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.common.torch_layers import MlpExtractor

from imitation.data.wrappers import RolloutInfoWrapper

from imitation.algorithms.adversarial.gail import GAIL
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from imitation.data.rollout import flatten_trajectories
from adapter.adapter_shoot_back import SB3SingleCombatEnv
from net.net_shoot_imitation import CustomImitationPolicy, MLPBase, GRULayer
from stable_baselines3.common.callbacks import BaseCallback


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


class Freezer:
    def __init__(self, policy, num_old_logits=11, unfreeze_step=50000):
        self.policy = policy
        self.num_old_logits = num_old_logits
        self.unfreeze_step = unfreeze_step
        self.unfrozen = False
        self._freeze_feature_extractor()
        print("✅ 特征提取器已冻结")

    def _freeze_feature_extractor(self):
        for param in self.policy.features_extractor.parameters():
            param.requires_grad = False

    def filter_gradients(self, current_step):
        """手动清除前三维 action 输出层的梯度"""
        if current_step <= self.unfreeze_step:
            action_weight = self.policy.action_net.weight
            action_bias = self.policy.action_net.bias

            if action_weight.grad is not None:
                action_weight.grad[:self.num_old_logits] = 0.0
            if action_bias.grad is not None:
                action_bias.grad[:self.num_old_logits] = 0.0

    def maybe_unfreeze(self, current_step):
        if not self.unfrozen and current_step >= self.unfreeze_step:
            for param in self.policy.parameters():
                param.requires_grad = True
            self.unfrozen = True
            print("🔓 已解冻全部参数")


def gail_custom_callback(rollout_result):
    global last_save_step
    step = gail_trainer.gen_algo.num_timesteps

    # 冻结控制
    freezer.maybe_unfreeze(step)
    freezer.filter_gradients(step)

    # # 获取最近 rollout 中的观测值
    # obs = rollout_result.observations
    # obs_tensor = torch.as_tensor(obs, device=ppo.device)
    #
    # # 手动 forward 策略，获取 logits
    # with torch.no_grad():
    #     features = ppo.policy.extract_features(obs_tensor)
    #     latent_pi, _ = ppo.policy.mlp_extractor(features)
    #     logits = ppo.policy.action_net(latent_pi)
    #     if logits.shape[1] > 12:
    #         fire_logit = logits[:, 12].mean().item()
    #         ppo.logger.record("custom/fire_logit", fire_logit)

    # ✅ 每 X 步保存一次模型
    save_interval = 100_000
    if step - last_save_step >= save_interval:
        save_path = f"./trained_model/shoot_back/check_point/gail_ppo_checkpoint_{step}.zip"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ppo.save(save_path)
        last_save_step = step
        print(f"📦 模型已保存到：{save_path}")


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


def load_expert_trajectories(npz_path, max_episode_len=None):
    data = np.load(npz_path)
    observations = data["observations"]
    actions = data["actions"]
    rewards = data["rewards"]  # ✅ 加载 reward

    if max_episode_len is None:
        max_episode_len = 200

    trajectories = []
    total_steps = observations.shape[0]
    num_trajs = total_steps // max_episode_len

    for i in range(num_trajs):
        start = i * max_episode_len
        end = (i + 1) * max_episode_len

        traj_obs = observations[start:end+1]
        traj_acts = actions[start:end]
        traj_rews = rewards[start:end]  # ✅ 真实 reward 传进来

        traj = TrajectoryWithRew(
            obs=traj_obs,
            acts=traj_acts,
            rews=traj_rews,
            infos=None,
            terminal=True,
        )
        trajectories.append(traj)

    print(f"✅ 成功加载 {len(trajectories)} 条 expert trajectory")
    return trajectories


class HybridRewardWrapper(VecEnvWrapper):
    def __init__(self, venv, gail_reward_fn, alpha=0.9, beta=0.1):
        super().__init__(venv)
        self.gail_reward_fn = gail_reward_fn
        self.alpha = alpha
        self.beta = beta

    def step_wait(self):
        obs, env_rewards, dones, infos = self.venv.step_wait()
        acts = [info["act"] for info in infos]  # 提前在 rollout 时记录 act

        # 获取 GAIL reward
        gail_rewards = self.gail_reward_fn(obs, acts, None)

        # 混合 reward
        mixed_rewards = self.alpha * gail_rewards + self.beta * env_rewards
        return obs, mixed_rewards, dones, infos


# 4. 设置奖励混合机制
class MixedRewardEnv(RewardVecEnvWrapper):
    def __init__(self, venv, reward_fn, imitation_weight=0.9, env_weight=0.1):
        super().__init__(venv, reward_fn)
        self.imitation_weight = imitation_weight
        self.env_weight = env_weight
        self.last_obs = None
        self.last_act = None

    def reset(self):
        obs = self.venv.reset()
        self.last_obs = obs
        return obs

    def step_async(self, actions):
        self.last_act = actions
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, env_rewards, dones, infos = self.venv.step_wait()
        # imitation_rewards 使用当前 obs, action, next_obs, done
        imitation_rewards = self.reward_fn(
            self.last_obs, self.last_act, obs, dones
        )
        mixed_rewards = self.imitation_weight * imitation_rewards + self.env_weight * env_rewards
        self.last_obs = obs
        return obs, mixed_rewards, dones, infos



class ActRecorderWrapper(RolloutInfoWrapper):
    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        actions = self.actions
        for i, info in enumerate(infos):
            info["act"] = actions[i]
        return obs, rewards, dones, infos


# ✅ 添加：从旧模型加载参数到新模型（尽可能多地迁移）
def load_partial_policy_weights(old_model_path, new_model):
    if not os.path.exists(old_model_path):
        raise FileNotFoundError(f"找不到模型路径：{old_model_path}")

    print(f"🚀 加载旧模型参数中: {old_model_path}")
    old_model = PPO.load(old_model_path, device=new_model.device)
    old_policy = old_model.policy
    new_policy = new_model.policy

    old_state_dict = old_policy.state_dict()
    new_state_dict = new_policy.state_dict()

    loaded_keys = []
    skipped_keys = []

    for name, param in old_state_dict.items():
        if name in new_state_dict and param.shape == new_state_dict[name].shape:
            new_state_dict[name].copy_(param.data)
            loaded_keys.append(name)
        else:
            skipped_keys.append(name)

    print(f"✅ 成功加载权重数: {len(loaded_keys)}")
    if skipped_keys:
        print(f"⚠️ 跳过不匹配的权重: {skipped_keys[:5]}{' ...' if len(skipped_keys) > 5 else ''}")

    with torch.no_grad():
        old_logits = old_model.policy.action_net.weight
        new_logits = new_model.policy.action_net.weight
        old_bias = old_model.policy.action_net.bias
        new_bias = new_model.policy.action_net.bias

        # ✅ 如果 old 是 3 维，新是 4 维，就复制前三维参数
        with torch.no_grad():
            new_logits[:10] = old_logits[:10]
            new_bias[:10] = old_bias[:10]

            new_bias[12] = -5.0

            print(f"✅ 迁移前 {10} logits，动作 {12} 设置 bias={-5}")




if __name__ == "__main__":
    last_save_step = 0

    expert_trajs = load_expert_trajectories(f"./test_result/expert_data/expert_data2.npz")

    args3 = Namespace(
        config_name='1v1/ShootMissile/HierarchyVsBaselineShootBack',
        model_path='./trained_model/shoot_back/ppo_air_combat.zip',
        log_file='./train/result/train_shoot_back2_gail3.log',
        save_path='./test_result/expert_data',
        save_npz=True,
        num_envs=16,
        max_steps=50_001,
    )

    # 构造环境
    env_fns = [make_env(i, args3.config_name, args3.log_file) for i in range(args3.num_envs)]
    venv = SubprocVecEnv(env_fns)
    # 构建 reward_net（用于 imitation reward）
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=None,
    )

    # 包装环境以使用混合奖励
    venv_wrapped = MixedRewardEnv(venv, reward_net.predict, imitation_weight=0.9, env_weight=0.1)

    # 初始化 PPO 策略
    policy_kwargs = dict(
        features_extractor_class=CustomImitationPolicy,
        features_extractor_kwargs={}
    )

    ppo = PPO(
        "MlpPolicy",
        venv_wrapped,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        verbose=1,
        tensorboard_log="./ppo_air_combat_tb/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # ✅ 加载旧策略参数（包括前三维动作输出）
    load_partial_policy_weights(args3.model_path, ppo)

    # 初始化 Freezer + TensorBoard Callback
    freezer = Freezer(ppo.policy, num_old_logits=11, unfreeze_step=1_000_000)

    # 初始化 GAIL trainer
    gail_trainer = GAIL(
        demonstrations=expert_trajs,  # ✅ 不需要 flatten
        demo_batch_size=64,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv_wrapped,
        gen_algo=ppo,
        reward_net=reward_net,
        allow_variable_horizon=True,  # ✅ 允许不同长度的 episode
    )

    # GAIL 训练
    gail_trainer.train(total_timesteps=4_000_000, callback=gail_custom_callback)

    # 保存最终模型
    ppo.save("./trained_model/shoot_back/gail_ppo_policy3.zip")
    print("✅ 模型已保存到 './trained_model/shoot_back/gail_ppo_policy3.zip'")
