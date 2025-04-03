import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from my_env import SB3SingleCombatEnv
from my_policy import CustomImitationPolicy


def make_env(env_id, config_name, save_dir):
    def _init():
        env = SB3SingleCombatEnv(
            env_id,
            config_name=config_name,
            save_acmi=True,
            save_dir=save_dir,
            record_render=True
        )
        return env
    return _init


def load_model(model_path, env, device):
    if model_path.endswith(".zip"):
        print("✅ 加载 SB3 .zip 模型")
        return PPO.load(model_path, env=env, device=device)
    elif model_path.endswith(".pt"):
        print("🔄 加载 .pt 模仿学习特征提取器")
        policy_kwargs = dict(
            features_extractor_class=CustomImitationPolicy,
            features_extractor_kwargs={}
        )
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device
        )
        pretrained_dict = torch.load(model_path, map_location="cpu")
        current_dict = model.policy.features_extractor.state_dict()
        matched_dict = {k: v for k, v in pretrained_dict.items() if k in current_dict}
        current_dict.update(matched_dict)
        model.policy.features_extractor.load_state_dict(current_dict)
        print("✅ 成功加载模仿学习参数")
        return model
    else:
        raise ValueError("❌ 不支持的模型文件类型，请提供 .zip 或 .pt 文件")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_fns = [make_env(i, args.config_name, args.output_dir) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    model = load_model(args.model_path, env, device)

    obs = env.reset()
    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        print(f"[Step {step}] Rewards: {rewards}")

    env.close()
    print("🎯 推理完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True,
                        help="环境配置名，例如 '1v1/ShootMissile/HierarchyVsBaselineImitation'")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径，支持 .zip 或 .pt")
    parser.add_argument("--output_dir", type=str, default="./eval_output/",
                        help="acmi 或渲染结果的保存路径")
    parser.add_argument("--num_envs", type=int, default=8,
                        help="并行环境数量")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="推理步数")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
