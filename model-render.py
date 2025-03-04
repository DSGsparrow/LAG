import torch
import numpy as np
import logging
import sys

from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv
from LAGmaster.algorithms.ppo.ppo_policy import PPOPolicy as Policy
from LAGmaster.config import get_config
from LAGmaster.runner.base_runner import Runner, ReplayBuffer


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
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

def parse_args(args, parser):
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--scenario-name', type=str, default='singlecombat_simple',
                       help="Which scenario to run on")
    group.add_argument('--render-mode', type=str, default='txt',
                       help="txt or real_time")
    all_args = parser.parse_known_args(args)[0]
    return all_args

def render(args, ego_path, enm_path, ego_ver, enm_ver):
    # args: 训练输入的超参
    parser = get_config()
    all_args = parse_args(args, parser)

    env = make_train_env(all_args)
    obs_space = env.observation_space
    act_space = env.action_space
    num_agents = env.num_agents
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
    buffer = ReplayBuffer(all_args, num_agents // 2, obs_space, act_space)

    ego_policy = Policy(all_args, obs_space, act_space, device=device)
    enm_policy = Policy(all_args, obs_space, act_space, device=device)

    ego_policy.actor.load_state_dict(torch.load(ego_path, weights_only=True))
    ego_policy.prep_rollout()

    enm_policy.actor.load_state_dict(torch.load(enm_path, weights_only=True))
    enm_policy.prep_rollout()

    logging.info("\nStart render ...")

    render_episode_rewards = 0

    render_obs = env.reset()
    env.render(mode='txt', filepath=f'render-result/{experiment_name}-{ego_ver}-{enm_ver}.txt.acmi')
    render_masks = np.ones((1, *buffer.masks.shape[2:]), dtype=np.float32)
    render_rnn_states = np.zeros((1, *buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
    render_opponent_obs = render_obs[:, num_agents // 2:, ...]
    render_obs = render_obs[:, :num_agents // 2, ...]
    render_opponent_masks = np.ones_like(render_masks, dtype=np.float32)
    render_opponent_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
    while True:
        ego_policy.prep_rollout()
        render_actions, render_rnn_states = ego_policy.act(np.concatenate(render_obs),
                                                            np.concatenate(render_rnn_states),
                                                            np.concatenate(render_masks),
                                                            deterministic=True)
        render_actions = np.expand_dims(_t2n(render_actions), axis=0)
        render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
        render_opponent_actions, render_opponent_rnn_states \
            = enm_policy.act(np.concatenate(render_opponent_obs),
                                            np.concatenate(render_opponent_rnn_states),
                                            np.concatenate(render_opponent_masks),
                                            deterministic=True)
        render_opponent_actions = np.expand_dims(_t2n(render_opponent_actions), axis=0)
        render_opponent_rnn_states = np.expand_dims(_t2n(render_opponent_rnn_states), axis=0)
        render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)

        # Obser reward and next obs
        render_obs, render_rewards, render_dones, render_infos = env.step(render_actions)
        render_rewards = render_rewards[:, :num_agents // 2, ...]
        render_episode_rewards += render_rewards
        env.render(mode='txt', filepath=f'render-result/{experiment_name}-{ego_ver}-{enm_ver}.txt.acmi')
        if render_dones.all():
            break
        render_opponent_obs = render_obs[:, num_agents // 2:, ...]
        render_obs = render_obs[:, :num_agents // 2, ...]
    print(render_episode_rewards)


if __name__ == "__main__":
    args = sys.argv[1:]
    ego_ver = 'latest'
    enm_ver = '400'
    ego_path = 'LAGmaster/scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/run16/actor_' + ego_ver + '.pt'
    enm_path = 'LAGmaster/scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/v1/run16/actor_' + enm_ver + '.pt'
    render(args, ego_path, enm_path, ego_ver, enm_ver)