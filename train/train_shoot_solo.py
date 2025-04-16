import gym
import gymnasium
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces
import argparse
import os
import logging

from LAGmaster.envs.JSBSim.envs import SingleCombatEnvShootSolo
from net import (CustomImitationShootBackPolicy, CustomActorCriticShootBackPolicy,
                 CustomImitationPolicy, CustomTransformerExtractor)

from adapter.adapter_shoot_solo import ShootControlWrapper

class SB3SingleCombatEnv(gymnasium.Env):
    """将 SingleCombatEnvTest 适配为 SB3 兼容的 Gym 环境"""

    def __init__(self, env_id, config_name):
        super(SB3SingleCombatEnv, self).__init__()
        self.env = SingleCombatEnvShootSolo(config_name, env_id)  # 你的原始环境
        # obs_shape = self.env.get_obs().shape[0]  # 获取观测空间维度
        # act_shape = self.env.get_action_space().shape[0]  # 获取动作空间维度
        # 继承原始环境的动作空间和观察空间
        # self.action_space = self.env.action_space

        # 提取原始环境的 action_space
        if isinstance(self.env.action_space, spaces.Tuple):
            # 获取所有离散动作空间的维度
            action_dims = []
            for space in self.env.action_space.spaces:
                if isinstance(space, spaces.MultiDiscrete):
                    action_dims.extend(space.nvec)  # 展开 MultiDiscrete
                elif isinstance(space, spaces.Discrete):
                    action_dims.append(space.n)  # Discrete 直接添加
                else:
                    raise ValueError("Unsupported action space type: {}".format(type(space)))

            # 转换为 MultiDiscrete
            self.action_space = spaces.MultiDiscrete(action_dims)
        else:
            # raise ValueError("Unexpected action space type: {}".format(type(self.env.action_space)))
            self.action_space = self.env.action_space

        self.observation_space = self.env.observation_space

        # # 定义 Gym 兼容的观测和动作空间
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(act_shape,), dtype=np.float32)

    def step(self, action):
        # 将长度为 4 的动作转换为长度为 (1,4) 的动作
        # actual_action = action.reshape(-1, 3)  # 取第一个值
        # 因为内部的环境，假设可能有多架飞机在控制，所有第一位都加了个序号
        # reward 和dones什么的直接取值

        action = np.expand_dims(action, axis=0) if action.ndim == 1 else action

        action = action.astype(int)

        obs, rewards, dones, info = self.env.step(action)

        timeout = info.get('timeout', False)

        observation, reward, terminated, truncated, info = obs[0], rewards.item(), dones.item(), timeout, info

        # logging.info('test')

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """重置环境，支持 `seed` 以适配 SB3"""
        super().reset(seed=seed)  # 让 Gym 兼容 SB3 的 `seed`
        obs = self.env.reset()
        observation = obs[0]
        return observation, None

    def close(self):
        return self.env.close()

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi', tacview=None):
        self.env.render(mode=mode, filepath=filepath, tacview=tacview)


class EnvIDFilter(logging.Filter):
    def __init__(self, env_id):
        super().__init__()
        self.env_id = env_id

    def filter(self, record):
        record.env_id = f"{self.env_id}"
        return True


def setup_logging(env_id=0, log_file=None):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="1v1/ShootMissile/HierarchyVsBaselineShootSolo")

    # 基本路径
    parser.add_argument("--log_file", type=str, default="./train/result/train_shoot_solo3.log")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--pretrained_pt_path", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="./trained_model/shoot_solo3/shoot_solo_checkpoints/")
    parser.add_argument("--tb_log", type=str, default="./ppo_air_combat_tb/")
    parser.add_argument("--save_model_path", type=str, default="./trained_model/shoot_solo3/ppo_air_combat")

    # 模型路径
    parser.add_argument("--fly_model_path", type=str, default="trained_model/shoot_back/ppo_air_combat.zip")
    parser.add_argument("--guide_model_path", type=str, default="trained_model/guide/ppo_air_combat.zip")

    # 环境参数
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--raw_obs_dim", type=int, default=21)
    parser.add_argument("--fly_act_dim", type=int, default=3)
    parser.add_argument("--fire_act_dim", type=int, default=2)
    parser.add_argument("--warmup_action", nargs='+', type=float, default=[1, 2, 1, 0.0, 0.0])

    # 多线程
    parser.add_argument("--num_envs", type=int, default=16)

    # 训练参数
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--save_freq", type=int, default=10_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.02)

    # Transformer 网络参数
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser.parse_args()


def make_env(env_id, args):
    setup_logging(env_id, args.log_file)
    return ShootControlWrapper(lambda: SB3SingleCombatEnv(env_id, config_name=args.config), args)


def main():
    args = parse_args()

    env_fns = [lambda env_id=i: make_env(env_id, args) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=CustomTransformerExtractor,
        features_extractor_kwargs=dict(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            seq_len=args.history_len,
            input_dim=args.raw_obs_dim + args.fly_act_dim + args.fire_act_dim,
        ),
        net_arch=[],
    )

    if os.path.exists(args.model_path):
        print("✅ 加载已有模型继续训练...")
        model = PPO.load(
            args.model_path,
            env=env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.tb_log,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        print("🆕 没有旧模型，创建新 PPO 模型")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=args.tb_log,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        if os.path.exists(args.pretrained_pt_path):
            print("🔄 加载模仿学习预训练参数...")
            policy = model.policy
            pretrained = torch.load(args.pretrained_pt_path, map_location="cpu")

            mlp_state = {k.replace("feature_extractor.", ""): v for k, v in pretrained.items() if k.startswith("feature_extractor.")}
            policy.features_extractor.mlp.load_state_dict(mlp_state)

            gru_state = {k.replace("gru.", ""): v for k, v in pretrained.items() if k.startswith("gru.")}
            policy.features_extractor.gru.gru.load_state_dict(gru_state)

            act_state = {k.replace("action_head.", ""): v for k, v in pretrained.items() if k.startswith("action_head.")}
            policy.action_net.load_state_dict(act_state)

            print("✅ 模仿学习参数加载成功！")

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.checkpoint_path,
        name_prefix="ppo_air_combat_shoot"
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name="shoot_solo3",
        callback=checkpoint_callback
    )

    # 最终训练完成后保存一次完整模型
    model.save(args.save_model_path)


if __name__ == "__main__":
    main()
