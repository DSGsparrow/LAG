import os
import argparse
import torch
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from adapter.adapter_shoot_missile import SB3SingleCombatEnv
from net.net_shoot_missile import CustomPolicy
from net.net_shoot_imitation import CustomImitationPolicy


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
    env_fns = [make_env(i, args.config_name, args.log_file) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    model = load_model(args.model_path, env, device)

    obs = env.reset()
    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        if step % 50_000 == 0 or step == args.max_steps - 1:
            print(f"current_step: {step}")

            episode_nums =0
            win_nums = 0
            for env_id in args.num_envs:
                episode_num = env.get_attr('episode_num', env_id)
                episode_nums += episode_num[0]
                win_num = env.get_attr('win_num', env_id)
                win_nums += win_num[0]

            print(f'test {episode_nums} episode, shoot down enemy {win_nums} times, win rate is{win_nums / episode_nums * 100}%')

    env.close()
    print("🎯 推理完成")


if __name__ == "__main__":
    # 1, change args: model path, config name, output_dir
    # 2, yaml: render path, baselines

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str,
                        default='1v1/ShootMissile/HierarchyVsBaselineSelf',
                        help="环境配置名")
    parser.add_argument("--model_path", type=str,
                        default="./trained_model/shoot_missile/ppo_air_combat_3.zip",
                        help="模型路径，支持 .zip 或 .pt")
    parser.add_argument("--log_file", type=str, default="./test_result/log/test_shoot3_vs_dodge.log",
                        help="acmi 或渲染结果的保存路径")
    parser.add_argument("--num_envs", type=int, default=16,
                        help="并行环境数量")
    parser.add_argument("--max_steps", type=int, default=100_000,
                        help="推理步数")
    args = parser.parse_args()

    # os.makedirs(args.output_dir, exist_ok=True)
    main(args)

