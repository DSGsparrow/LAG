import gym
import gymnasium
from gym import spaces
import torch.cuda
import numpy as np
from typing import Optional
import time
from LAGmaster.envs.JSBSim.core.catalog import Catalog as c

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv

from LAGmaster.envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv

from utils.pid import *


# 定义适配层
class PIDActionAdapter(gymnasium.Env):
    def __init__(self, env):
        super(PIDActionAdapter, self).__init__()
        self.env = env
        # 继承原始环境的动作空间和观察空间
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.state_var = [
            c.delta_altitude,  # 0. delta_h   (unit: m)
            c.delta_heading,  # 1. delta_heading  (unit: °)
            c.delta_velocities_u,  # 2. delta_v   (unit: m/s)
            c.position_h_sl_m,  # 3. altitude  (unit: m)
            c.attitude_roll_rad,  # 4. roll      (unit: rad)
            c.attitude_pitch_rad,  # 5. pitch     (unit: rad)
            c.velocities_u_mps,  # 6. v_body_x   (unit: m/s)
            c.velocities_v_mps,  # 7. v_body_y   (unit: m/s)
            c.velocities_w_mps,  # 8. v_body_z   (unit: m/s)
            c.velocities_vc_mps,  # 9. vc        (unit: m/s)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,  # [-1., 1.]  副翼 滚转角
            c.fcs_elevator_cmd_norm,  # [-1., 1.]  升降舵 俯仰角
            c.fcs_rudder_cmd_norm,  # [-1., 1.]  舵 偏航角
            c.fcs_throttle_cmd_norm,  # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def step(self, action):
        # 将长度为 4 的动作转换为长度为 (1,4) 的动作
        actual_action = action.reshape(-1, 4)  # 取第一个值

        obs, rewards, dones, info = self.env.step(actual_action)
        observation, reward, terminated, truncated, info = obs, rewards, dones, dones, info

        agent_ids = self.env.agents.keys()
        agent_ids_list = list(agent_ids)
        origin_obs = np.array(self.env.agents[agent_ids_list[0]].get_property_values(self.state_var))  # 只有一个飞机

        return observation, reward, terminated, truncated, info, origin_obs

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 123

        # todo 改成迭代形式
        agent_ids = self.env.agents.keys()
        agent_ids_list = list(agent_ids)
        origin_obs = np.array(self.env.agents[agent_ids_list[0]].get_property_values(self.state_var))  # 只有一个飞机

        return self.env.reset(), origin_obs

    def close(self):
        return self.env.close()

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi', tacview=None):
        self.env.render(mode=mode, filepath=filepath, tacview=tacview)


def pid_test():
    env_origin = SingleControlEnv(config_name='1/heading')
    env = PIDActionAdapter(env_origin)
    obs, origin_obs = env.reset()  # obs:(1, 12), origin_obs:(10, )

    done1, done2 = False, False
    total_reward = 0

    # 得到一个序列的/一个 姿态目标
    pitch_init = [10]  # 俯仰 角度
    drift_init = [0]  # 偏航 角度 不输入pid
    roll_init = [0]  # 滚转 角度
    v_init = [120]  # 速度 也是不到pid

    # pid result record
    result = []

    pid_de = PIDDe()
    pid_da = PIDDa()

    # pid init
    pid_de.init_De(0.2, 0.3, 0.04, pitch_init[0])
    pid_da.init_Da(0.085, 0.095, 0.01, roll_init[0])

    while not (done1 and done2):
        roll = np.rad2deg(origin_obs[4])
        pitch = np.rad2deg(origin_obs[5])

        print('roll: ', roll, ', pitch: ', pitch, )

        done1 = abs(roll - roll_init[0]) < 0.1
        done2 = abs(pitch - pitch_init[0]) < 0.1

        if done1 and done2:
            break

        # pid update
        output_De = pid_de.update_De(pitch, 0.2)  # 俯仰
        output_Da = pid_da.update_Da(roll, 0.2)  # 滚转

        max_output = 40  # 假设 PID 输出的最大值为 100
        min_output = 0  # 假设 PID 输出的最小值为 -100
        normalized_output_De = 2 * (output_De - min_output) / (max_output - min_output) - 1
        normalized_output_Da = 2 * (output_Da - min_output) / (max_output - min_output) - 1

        output_De = (normalized_output_De + 1.) / 2 * 40
        output_Da = (normalized_output_Da + 1.) / 2 * 40

        print('output_De: ', output_De, ', output_Da: ', output_Da)

        # action pid
        action = [output_Da, output_De, 0, 1]  # [滚转，俯仰，偏航，速度]
        action = np.array(action)
        observation, reward, terminated, truncated, info, origin_obs = env.step(action)

        env.render(mode='txt', filepath="pid-control.txt.acmi")

        result.append(origin_obs)

        time.sleep(1)


def main():
    env_origin = SingleControlEnv(config_name='1/heading')
    env = PIDActionAdapter(env_origin)
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

    # test_model(model)  # 测试模型


# def test_model(model):
#     env_origin = SingleControlEnv(config_name='1/heading')
#     env = PIDActionAdapter(env_origin)
#     obs, info = env.reset()
#
#     done1, done2 = False, False
#     total_reward = 0
#
#     while not (done1 or done2):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done1, done2, info = env.step(action)
#         if len(info) > 0:
#             bp = 0
#         total_reward += reward
#         env.render(mode='txt', filepath="self-control.txt.acmi")
#
#     print(f'Total Reward: {total_reward}')
#     env.close()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    pid_test()