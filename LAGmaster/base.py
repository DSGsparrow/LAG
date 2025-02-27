import gym
import gymnasium
from gym import spaces
import torch.cuda
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv


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


def main():
    env_origin = SingleControlEnv(config_name='1/heading')
    env = ActionAdapter(env_origin)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, "./logs/")  # 将日志保存到 ./logs/
    obs = env.reset()
    # env.render(filepath="self.control.txt.acmi")

    # env = gym.make('CartPole-v1')  # 创建环境

    # 设置 EvalCallback
    # eval_callback = EvalCallback(
    #     eval_env,  # 评估环境
    #     best_model_save_path="./logs/",  # 保存最佳模型的路径
    #     log_path="./logs/",  # 保存评估结果的路径
    #     eval_freq=1000,  # 每 1000 步评估一次
    #     deterministic=True,  # 使用确定性动作
    #     render=False,  # 是否渲染评估环境
    # )

    model = PPO("MlpPolicy", env, verbose=1, device='cuda', tensorboard_log="./logs/")  # 创建模型
    model.learn(total_timesteps=80000)  # 训练模型
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
        env.render(mode='txt', filepath="self-control.txt.acmi")

    print(f'Total Reward: {total_reward}')
    env.close()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()