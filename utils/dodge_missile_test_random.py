import copy

import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import torch
import os
import sys
import geopy.distance
# from stable_baselines3.common.vec_env import DummyVecEnv

from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv, SingleCombatEnvTest
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv
# from envs.env_wrappers import SubprocVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv
from LAGmaster.algorithms.ppo.ppo_policy import PPOPolicy as Policy
from LAGmaster.config import get_config
from LAGmaster.runner.base_runner import Runner, ReplayBuffer

from parse_log_file import parse_log_file


# 设置参数
my_aircraft = {
    "lat": 60.0,  # 北纬60度
    "lon": 120.0,  # 东经120度
    "alt": 20000,  # 高度 20000 英尺
    "heading": 0  # 朝向正北（0度）
}


def calculate_enemy_position(distance, angle):
    """根据距离和角度计算敌机的经纬度"""
    origin = (my_aircraft["lat"], my_aircraft["lon"])
    destination = geopy.distance.distance(meters=distance).destination(origin, angle)
    return destination.latitude, destination.longitude  # 纬度，经度


def calculate_bearing(lat1, lon1, lat2, lon2):
    """计算从 (lat1, lon1) 指向 (lat2, lon2) 的方位角"""
    delta_lon = np.radians(lon2 - lon1)
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360  # 转换为 0-360 度范围


def generate_random_enemy_positions():
    pass


def generate_enemy_positions():
    """遍历敌机位置、速度、高度的所有可能情况，并计算其经纬度和朝向"""
    distances = np.linspace(8000, 15000, num=20)  # 8000-15000米
    angles = np.linspace(0, 360, num=36)  # 0-360度
    altitudes = np.linspace(14000, 30000, num=10)  # 14000-30000英尺
    speeds = np.linspace(500, 1000, num=8)  # 400-1000英尺/秒

    enemy_positions = []
    counter = 0
    for distance in distances:
        for angle in angles:
            for altitude in altitudes:
                for speed in speeds:
                    lat, lon = calculate_enemy_position(distance, angle)
                    heading = calculate_bearing(lat, lon, my_aircraft["lat"], my_aircraft["lon"])
                    counter += 1

                    enemy_positions.append({
                        "lat": lat,
                        "lon": lon,
                        "distance": distance,
                        "angle": angle,
                        "alt": altitude,
                        "speed": speed,
                        "heading": heading,  # 敌机朝向我机的角度
                        "counter": counter
                    })
    return enemy_positions


def parse_args(args, parser):
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--scenario-name', type=str, default='singlecombat_simple',
                       help="Which scenario to run on")
    group.add_argument('--render-mode', type=str, default='txt',
                       help="txt or real_time")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnvTest(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.env_name == "MultipleCombat":
        if all_args.n_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    else:
        if all_args.n_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def _t2n(x):
    return x.detach().cpu().numpy()


def get_unique_filename(experiment_name, ego_ver, enm_ver, folder="./render_result_random"):
    """
    生成唯一的实验结果文件名，防止覆盖已有文件。

    参数:
    - experiment_name (str): 实验名称
    - folder (str): 存放实验结果的文件夹，默认是 "experiments"

    返回:
    - str: 唯一的实验文件路径
    """
    if not os.path.exists(folder):
        os.makedirs(folder)  # 如果文件夹不存在，则创建它

    txt_name = f'{experiment_name}_{ego_ver}_{enm_ver}'

    base_filename = os.path.join(folder, txt_name)
    filename = f"{base_filename}.txt.acmi"
    counter = 1

    while os.path.exists(filename):  # 如果文件已存在，递增编号
        filename = f"{base_filename}_{counter}.txt.acmi"
        counter += 1

    return filename


def render(args, ego_path, ego_ver):
    parser = get_config()
    all_args = parse_args(args, parser)

    env = make_train_env(all_args)
    obs_space = env.observation_space
    act_space = env.action_space
    num_agents = env.num_agents
    experiment_name = all_args.experiment_name

    save_acmi_path = get_unique_filename('dodge missile', ego_ver, 'pursue')

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # buffer
    buffer = ReplayBuffer(all_args, num_agents, obs_space, act_space)

    ego_policy = Policy(all_args, obs_space, act_space, device=device)

    ego_policy.actor.load_state_dict(torch.load(ego_path, weights_only=True))
    ego_policy.prep_rollout()


    logging.info("\nStart render, render mode is {self.render_mode} ... ...")
    render_episode_rewards = 0
    render_obs = env.reset()
    render_masks = np.ones((1, *buffer.masks.shape[2:]), dtype=np.float32)
    render_rnn_states = np.zeros((1, *buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
    env.render(mode='txt', filepath=save_acmi_path)
    while True:
        ego_policy.prep_rollout()
        render_actions, render_rnn_states = ego_policy.act(np.concatenate(render_obs),
                                                            np.concatenate(render_rnn_states),
                                                            np.concatenate(render_masks),
                                                            deterministic=True)
        render_actions = np.expand_dims(_t2n(render_actions), axis=0)
        render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)

        # Obser reward and next obs
        render_obs, render_rewards, render_dones, render_infos = env.step(render_actions)

        render_episode_rewards += render_rewards
        env.render(mode='txt', filepath=save_acmi_path)
        if render_dones.all():
            a = render_infos[0]
            break
    render_infos = {}
    render_infos['render_episode_reward'] = render_episode_rewards
    logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))


def simulate_missile(ego_policy, env, buffer, output_states_file):
    """模拟导弹发射，返回是否成功躲避"""
    # render_episode_rewards = 0
    # env.envs[0].set_enemy(enemy)
    # env.set_attr("set_enemy", enemy)
    # render_obs = env.envs[0]._pack(env.envs[0].get_obs()).reshape(1, 1, -1)
    render_masks = np.ones((1, *buffer.masks.shape[2:]), dtype=np.float32)
    render_rnn_states = np.zeros((1, *buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
    # a = np.concatenate(render_rnn_states)
    save_acmi_path = get_unique_filename('dm', 'l', 'p')
    env.render(mode='txt', filepath=save_acmi_path)

    # logging.info('simulate init state: ' + str(enemy))

    # missile_done = False
    #
    # state = None

    while True:
        ego_policy.prep_rollout()
        render_actions, render_rnn_states = ego_policy.act(np.concatenate(render_obs),
                                                            np.concatenate(render_rnn_states),
                                                            np.concatenate(render_masks),
                                                            deterministic=True)
        render_actions = np.expand_dims(_t2n(render_actions), axis=0)
        render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)

        # Obser reward and next obs
        render_obs, render_rewards, render_dones, render_infos = env.step(render_actions)

        # if np.all(render_obs[0][0][15:21] == 0) and not render_dones[0] and not missile_done:
        #     # 弹失效，且未结束回合
        #     missile_done = True
        #     render_states = env.envs[0]._pack(env.envs[0].get_states()).reshape(-1,)
        #
        #     logging.info("missile down states " + str(render_states))

        # if render_dones.all():
        #     success = render_infos[0].get("dodge success", False)
        #     logging.info("missile dodge missile success " + str(success))
        #     if success:
        #         logging.info("missile down states " + str(render_states))
        #
        #         state = {
        #             "my_lat": render_states[0], "my_lon": render_states[1], "my_alt": render_states[2],
        #             "my_x": render_states[3], "my_y": render_states[4], "my_z": render_states[5],
        #             "my_vx": render_states[6], "my_vy": render_states[7], "my_vz": render_states[8],
        #             "enemy_lat": render_states[9], "enemy_lon": render_states[10], "enemy_alt": render_states[11],
        #             "enemy_x": render_states[12], "enemy_y": render_states[13], "enemy_z": render_states[14],
        #             "enemy_vx": render_states[15], "enemy_vy": render_states[16], "enemy_vz": render_states[17],
        #         }
        #         state_file = output_states_file  # "./test_result/dodge_test/parsed_states.json"
        #         with open(state_file, 'a') as f:
        #             json.dump(state, f)
        #             f.write('\n')  # 每行存储一个 JSON 对象
        #
        #     break

        # render_states = copy.deepcopy(env.envs[0]._pack(env.envs[0].get_states()).reshape(-1, ))

        # render_episode_rewards += render_rewards
        env.render(mode='txt', filepath=save_acmi_path)
        # if render_dones.all():
        #     success = render_infos[0]['success']
        #     break
        # elif missile_done:
        #     success = True
        #     break
    # render_infos = {}
    # render_infos['render_episode_reward'] = render_episode_rewards
    # logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    # return success, render_episode_rewards.item(), state


def run_simulation(args, ego_path, log_file, output_file, output_states_file, gap):
    # gap 跳过的对局设置数
    """运行仿真实验，统计数据，并保存到文件"""
    # results = []
    enemy_positions = generate_enemy_positions()

    # 先读文件
    # counter = parse_log_file(enemy_positions, log_file,
    #                output_file,
    #                output_states_file)
    # if gap:
    #     enemy_positions = enemy_positions[gap:]

    # env and policy and render set
    parser = get_config()
    all_args = parse_args(args, parser)

    env = make_train_env(all_args)
    obs_space = env.observation_space
    act_space = env.action_space
    num_agents = 1  # env.num_agents
    experiment_name = all_args.experiment_name

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # buffer
    buffer = ReplayBuffer(all_args, num_agents, obs_space, act_space)

    # policy
    ego_policy = Policy(all_args, obs_space, act_space, device=device)

    ego_policy.actor.load_state_dict(torch.load(ego_path, weights_only=True))
    ego_policy.prep_rollout()

    logging.info("\nStart render, render mode is {self.render_mode} ... ...")

    a = env.reset()
    # simulate circulate
    while True:
        simulate_missile(ego_policy, env, buffer, output_states_file)


    # for enemy in enemy_positions:
    #     success, render_episode_rewards, state = simulate_missile(enemy, ego_policy, env, buffer, output_states_file)
    #
    #     result = {
    #         "distance": enemy["distance"],
    #         "angle": enemy["angle"],
    #         "alt": enemy["alt"],
    #         "speed": enemy["speed"],
    #         "success": success,
    #         "reward": render_episode_rewards,
    #         "counter": enemy["counter"],
    #         "state": state,
    #     }
    #
    #     # output_file = "./test_result/dodge_test/parsed_results.json"
    #
    #     # 追加写入 JSON 文件，避免数据丢失
    #     with open(output_file, 'a') as f:
    #         json.dump(result, f)
    #         f.write('\n')  # 每行存储一个 JSON 对象
    #
    #     # results.append({
    #     #     'counter': enemy["counter"],
    #     #     "distance": enemy["distance"],
    #     #     "angle": enemy["angle"],
    #     #     "alt": enemy["alt"],
    #     #     "speed": enemy["speed"],
    #     #     "success": success,
    #     #     "reward": render_episode_rewards
    #     # })

    # 保存结果到文件
    # with open(save_path, "w") as f:
    #     json.dump(results, f, indent=4)
    #
    # print(f"Simulation results saved to {save_path}")
    # return results


def plot_heatmap(results):
    """可视化不同方向、距离的危险程度"""
    angles = [r["angle"] for r in results]
    distances = [r["distance"] for r in results]
    successes = [1 if r["success"] else 0 for r in results]  # 成功躲避标记

    plt.figure(figsize=(8, 6))
    plt.scatter(angles, distances, c=successes, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label="躲避成功(1) / 失败(0)")
    plt.xlabel("敌机角度 (度)")
    plt.ylabel("敌机距离 (米)")
    plt.title("敌机攻击危险程度分析")
    plt.show()


def setup_logging(run_dir, log_file = None):
    """配置 logging，让日志既输出到终端，又写入 run.log 文件"""
    if not log_file:
        os.makedirs(run_dir, exist_ok=True)  # 确保日志目录存在
        log_file = os.path.join(run_dir, "run.log")  # 日志文件路径

    # 获取全局 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设定最低日志级别

    # 清除已有的 handlers，防止重复添加
    logger.handlers.clear()

    # 终端 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 文件 Handler
    file_handler = logging.FileHandler(log_file, mode="a")  # "a" 追加模式
    file_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加 handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info("init complete, log path: " + log_file)



if __name__ == "__main__":
    args = sys.argv[1:]
    ego_path = 'LAGmaster/scripts/results/SingleCombat/1v1/DodgeMissile/HierarchyVsBaseline/ppo/1v1/run42/actor_latest.pt'
    save_path = './test_result/dodge_test/simulation_results.json'
    log_file = "./render_result_random/run_random.log"
    output_file = "./test_result/dodge_test/parsed_results_random.json"
    output_states_file = "./test_result/dodge_test/parsed_states_random.json"

    setup_logging('./render-result', log_file)

    run_simulation(args, ego_path, log_file, output_file, output_states_file, 0)
    # plot_heatmap(results)
