import os
import argparse
import numpy as np
import torch
import gymnasium as gym  # 新版 gymnasium
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn as nn

from env_factory.env_factory_selfplay import make_env, make_normal_env

# === 你已有的：parse_args / make_normal_env 等 ===
# from your_code import parse_args, make_normal_env

MANEUVER_MODEL_PATH = "trained_model/shoot_static3/final_model.zip"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="1v1/ShootMissile/HierarchySelfPlayShoot")
    parser.add_argument("--target_state", type=int, default=0)

    # 基本路径
    parser.add_argument("--log_file", type=str, default="./train/result/train_shoot_static_solo2.log")
    parser.add_argument("--model_path", type=str, default="trained_model/shoot_static_solo/final_model.zip")
    parser.add_argument("--pretrained_pt_path", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="./trained_model/shoot_static_solo2/checkpoints/")
    parser.add_argument("--tb_log", type=str, default="./ppo_air_combat_sp_tb/")
    parser.add_argument("--save_model_path", type=str, default="./trained_model/shoot_static_solo2")
    parser.add_argument("--model_dir", type=str, default="./model_pool/shoot_static")

    # 模型路径
    parser.add_argument("--fly_model_path", type=str, default="trained_model/shoot_back_t2/ppo_air_combat.zip")
    parser.add_argument("--fire_model_path", type=str, default="./trained_model/shoot_solo5/ppo_air_combat.zip")
    parser.add_argument("--guide_model_path", type=str, default="trained_model/guide/ppo_air_combat.zip")
    parser.add_argument("--dodge_model_path", type=str, default="trained_model/dodge_missile/ppo_air_combat_dodge4.zip")

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
    parser.add_argument("--save_interval", type=int, default=20_000)
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


class HybridActionWrapper(gym.Wrapper):
    """
    只训练“发射布尔值”，其余三维机动动作由预训练模型产出。
    - 原始 env 的动作空间形如 MultiDiscrete([3,5,3,2]) 或 Box([...])：
      * 取预训练模型输出的前三个分量作为机动控制
      * 将当前可训练 PPO 输出的 0/1（或连续值阈值化）作为最后一维发射决策
      * 合并为原环境需要的完整动作再 step
    - 观测空间不变
    - 本 wrapper 对外暴露的 action_space 为 Discrete(2)（仅发射 0/1）
    """
    def __init__(self, env, maneuver_model_path=MANEUVER_MODEL_PATH, use_gpu=False):
        super().__init__(env)
        # 加载预训练模型（仅推理用）
        self.maneuver_model = PPO.load(
            maneuver_model_path,
            device=("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        )

        # 对外只训练“发射布尔值”
        self.observation_space = env.observation_space
        self.action_space = spaces.Discrete(2)  # 0=不发射, 1=发射

        # 保存上一时刻观测，供预训练模型预测机动动作
        self._last_obs = None

        # 记下底层动作类型（MultiDiscrete / Box），以便正确拼装
        self._is_multidiscrete = isinstance(env.action_space, spaces.MultiDiscrete)
        self._is_box = isinstance(env.action_space, spaces.Box)

        # 检查原动作维度
        if self._is_multidiscrete:
            nvec = env.action_space.nvec
            assert len(nvec) == 4, \
                f"期望原始动作为4段，多离散维度为4，现在是 {len(nvec)}"
        elif self._is_box:
            assert env.action_space.shape[0] >= 4, \
                f"期望原始连续动作至少4维，现在是 {env.action_space.shape}"
        else:
            raise TypeError(
                "原环境动作空间既不是 MultiDiscrete 也不是 Box，"
                "请确认原动作空间是否为 [3,5,3,2] 或等价的 Box。"
            )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # gym 与 gymnasium 兼容：reset 可能返回 (obs, info)
        if isinstance(obs, tuple) and len(obs) == 2:
            self._last_obs, info = obs
            return self._last_obs, info
        else:
            self._last_obs = obs
            return self._last_obs

    def _predict_maneuver(self, obs):
        """用预训练模型预测完整动作，然后取前三维作为机动动作。"""
        # stable-baselines3 的 predict 接受单步 obs
        act, _ = self.maneuver_model.predict(obs, deterministic=True)

        if isinstance(act, (list, tuple)):
            act = np.array(act)

        # 如果原模型输出是标量（不应发生），转成 1D
        act = np.asarray(act).reshape(-1)

        # 取前三维（机动）
        maneuver = act[:3]
        return maneuver

    def _merge_action(self, maneuver, fire_bool):
        """拼装完整的四段动作，匹配底层 env 的动作空间格式和 dtype。"""
        fire = int(fire_bool)
        if self._is_multidiscrete:
            # 汇总为 [a0, a1, a2, fire]（整型）
            full = np.array([int(maneuver[0]), int(maneuver[1]), int(maneuver[2]), fire], dtype=np.int64)
            return full
        elif self._is_box:
            # 连续空间：将前三维原样、最后一维为 0/1（或按需要映射到连续区间）
            full = np.zeros((self.env.action_space.shape[0],), dtype=np.float32)
            full[:3] = np.asarray(maneuver, dtype=np.float32)
            # 若最后一维 Box 有上下界，按需要将 fire 映射到该区间，这里直接写 0/1
            full[3] = float(fire)
            # 如果原 Box 还有更多维（>=4），保持其余维度为 0（或按需处理）
            return np.clip(full, self.env.action_space.low, self.env.action_space.high)
        else:
            raise RuntimeError("未知动作空间类型。")

    def step(self, action):
        """
        外部传入的 action 仅为发射布尔值（0 或 1）。
        在此处组合为原环境需要的四段动作，调用底层 env.step。
        """
        if isinstance(action, (np.ndarray, list, tuple)):
            # Discrete(2) 在 VecEnv 下也可能传 ndarray([0/1])
            fire_bool = int(np.asarray(action).reshape(-1)[0])
        else:
            fire_bool = int(action)

        # 用上一时刻观测预测机动动作
        assert self._last_obs is not None, "内部状态异常：_last_obs 为空"
        maneuver = self._predict_maneuver(self._last_obs)
        full_action = self._merge_action(maneuver, fire_bool)

        step_out = self.env.step(full_action)

        # gym 与 gymnasium 兼容的返回解包
        if len(step_out) == 5:
            # gymnasium: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = step_out
            self._last_obs = obs
            return obs, reward, terminated, truncated, info
        else:
            # gym: obs, reward, done, info
            obs, reward, done, info = step_out
            self._last_obs = obs
            return obs, reward, done, info


def make_wrapped_env(env_id, args):
    """工厂函数：创建单个底层 env 并包上 HybridActionWrapper。"""
    base = make_normal_env(env_id, args)  # 你原来的环境创建
    wrapped = HybridActionWrapper(base, MANEUVER_MODEL_PATH, use_gpu=False)
    return wrapped


def main_shoot_static():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # === 修改处：用包装后的工厂函数，仍然支持多进程 SubprocVecEnv ===
    env_fns = [lambda env_id=i: make_wrapped_env(env_id, args) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)

    # Checkpoint 回调
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_interval,
        save_path=args.checkpoint_path,
        name_prefix="ppo_model"
    )

    # ✅ 只训练“发射布尔值”的 PPO，因此 policy 仍用 MLP，但动作空间来自 wrapper 的 Discrete(2)
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
        activation_fn=nn.ReLU
    )

    # 加载或新建模型（这里指“发射决策”模型）
    if os.path.exists(args.model_path):
        print("✅ 加载已有模型继续训练...")
        model = PPO.load(
            args.model_path,
            env=env,
            tensorboard_log=args.tb_log,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        print("🆕 没有旧模型，创建新 PPO 模型（仅训练发射布尔值）")
        model = PPO(
            "MlpPolicy",
            env,
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
            device="cuda" if torch.cuda.is_available() else "cpu",
            policy_kwargs=policy_kwargs
        )

    # 训练（其余与过去一致）
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="ppo_run"
    )

    # 最后保存一次
    final_model_path = os.path.join(args.save_model_path, "final_model.zip")
    model.save(final_model_path)
    print(f"✅ 最终模型已保存到 {final_model_path}")


if __name__ == "__main__":
    main_shoot_static()