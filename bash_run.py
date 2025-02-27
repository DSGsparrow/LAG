import subprocess
import os

# 设置环境变量
env = "SingleControl"
scenario = "1/heading"
algo = "ppo"
exp = "v1"
seed = 5

# 输出参数
print(f"env is {env}, scenario is {scenario}, algo is {algo}, exp is {exp}, seed is {seed}")

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 构造训练命令
command = [
    "python", "LAGmaster/scripts/train/train_jsbsim.py",
    "--env-name", env,
    "--algorithm-name", algo,
    "--scenario-name", scenario,
    "--experiment-name", exp,
    "--seed", str(seed),
    "--n-training-threads", "1",
    "--n-rollout-threads", "32",  # ?
    "--cuda",
    "--log-interval", "1",
    "--save-interval", "1",
    "--num-mini-batch", "5",
    "--buffer-size", "3000",
    "--num-env-steps", "1e8",
    "--lr", "3e-4",
    "--gamma", "0.99",
    "--ppo-epoch", "4",
    "--clip-params", "0.2",
    "--max-grad-norm", "2",
    "--entropy-coef", "1e-3",
    "--hidden-size", "128 128",
    "--act-hidden-size", "128 128",
    "--recurrent-hidden-size", "128",
    "--recurrent-hidden-layers", "1",
    "--data-chunk-length", "8"
]

# 使用 subprocess 执行命令
subprocess.run(command)
