import gym
import gymnasium
from gym import spaces
import torch.cuda
from typing import Optional

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv

from LAGmaster.envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, SingleCombatEnvTest


# 定义适配层
class ActionAdapter(gymnasium.Env):
    def __init__(self, env):
        super(ActionAdapter, self).__init__()
        self.env = env
        # 继承原始环境的动作空间和观察空间
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        # 将长度为 4 的动作转换为长度为 (1,4) 的动作
        actual_action = action.reshape(-1, 4)  # 取第一个值

        obs, rewards, dones, info = self.env.step(actual_action)
        observation, reward, terminated, truncated, info = obs, rewards, dones, dones, info

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 123

        return self.env.reset(), None

    def close(self):
        return self.env.close()

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi', tacview=None):
        self.env.render(mode=mode, filepath=filepath, tacview=tacview)


def get_config():
    parser = argparse.ArgumentParser(description="PPO Training for Single Combat Dodge Missile Scenario")

    # 环境相关参数
    parser.add_argument("--env-name", type=str, default="SingleCombat", help="环境名称")
    parser.add_argument("--algorithm-name", type=str, default="ppo", help="算法名称")
    parser.add_argument("--scenario-name", type=str, default="1v1/DodgeMissile/HierarchyVsBaseline", help="场景名称")
    parser.add_argument("--experiment-name", type=str, default="1v1", help="实验名称")

    # 训练设置
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--n-training-threads", type=int, default=1, help="训练时的线程数")
    parser.add_argument("--n-rollout-threads", type=int, default=1, help="采样线程数")
    parser.add_argument("--cuda", action="store_true", help="是否使用 CUDA 加速")

    # 记录与保存
    parser.add_argument("--log-interval", type=int, default=1, help="日志记录间隔（单位：回合）")
    parser.add_argument("--save-interval", type=int, default=1, help="模型保存间隔（单位：回合）")

    # 评估设置
    parser.add_argument("--n-choose-opponents", type=int, default=1, help="选择的对手数量")
    parser.add_argument("--use-eval", action="store_true", help="是否使用评估模式")
    parser.add_argument("--n-eval-rollout-threads", type=int, default=1, help="评估时的 rollout 线程数")
    parser.add_argument("--eval-interval", type=int, default=1, help="评估间隔")
    parser.add_argument("--eval-episodes", type=int, default=1, help="每次评估的 episode 数")

    # PPO 训练超参数
    parser.add_argument("--num-mini-batch", type=int, default=5, help="PPO 的 mini-batch 数量")
    parser.add_argument("--buffer-size", type=int, default=200, help="经验缓冲区大小")
    parser.add_argument("--num-env-steps", type=float, default=1e8, help="训练环境步数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--ppo-epoch", type=int, default=4, help="PPO 训练的 epoch 数")
    parser.add_argument("--clip-params", type=float, default=0.2, help="PPO 裁剪参数")
    parser.add_argument("--max-grad-norm", type=float, default=2, help="梯度裁剪最大范数")
    parser.add_argument("--entropy-coef", type=float, default=1e-3, help="熵正则系数")

    # 神经网络结构
    parser.add_argument("--hidden-size", type=int, nargs="+", default=[128, 128], help="Actor-Critic 网络的隐藏层大小")
    parser.add_argument("--act-hidden-size", type=int, nargs="+", default=[128, 128], help="Actor 网络的隐藏层大小")
    parser.add_argument("--recurrent-hidden-size", type=int, default=128, help="RNN 隐藏层大小")
    parser.add_argument("--recurrent-hidden-layers", type=int, default=1, help="RNN 隐藏层数")
    parser.add_argument("--data-chunk-length", type=int, default=8, help="RNN 训练时的数据块长度")

    return parser



def main():
    env_origin = SingleCombatEnvTest(config_name='1v1/DodgeMissile/HierarchyVsBaseline')
    env = ActionAdapter(env_origin)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, "./logs/")  # 将日志保存到 ./logs/
    obs = env.reset()

    model = PPO("MlpPolicy", env, verbose=1, device='cuda', tensorboard_log="./logs/")  # 创建模型
    model.learn(total_timesteps=800000)  # 训练模型
    model.save("ppo_cartpole")  # 保存模型

    test_model(model)  # 测试模型


def test_model(model):
    env_origin = SingleControlEnv(config_name='1/heading')
    env = ActionAdapter(env_origin)
    obs, info = env.reset()

    # env = gym.make('CartPole-v1', render_mode='human')  # 可视化只能在初始化时指定
    # obs, _ = env.reset()

    done1, done2 = False, False
    total_reward = 0

    while not (done1 or done2):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done1, done2, info = env.step(action)
        if len(info) > 0:
            bp = 0
        total_reward += reward
        env.render(mode='txt', filepath="../self-control.txt.acmi")

    print(f'Total Reward: {total_reward}')
    env.close()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()