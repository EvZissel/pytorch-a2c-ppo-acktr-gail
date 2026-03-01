import copy
import glob
import os
import time
from collections import deque
import sys

from torch.utils.tensorboard import SummaryWriter
from procgen import ProcgenEnv, ProcgenGym3Env

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
# from a2c_ppo_acktr.envs import make_vec_envs, make_ProcgenEnvs
from a2c_ppo_acktr.envs import make_ProcgenEnvs
from procgen import ProcgenEnv
from a2c_ppo_acktr.model import Policy, MLPAttnBase, MLPHardAttnBase, MLPHardAttnReinforceBase, ImpalaModel
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate_procgen
from a2c_ppo_acktr.utils import save_obj, load_obj
from a2c_ppo_acktr.procgen_wrappers import *
from a2c_ppo_acktr.logger import Logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from a2c_ppo_acktr.utils import init
from gym3.trajectory_recorder import TrajectoryRecorderWrapper


def main():
    args = get_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    env_name = 'maze'
    start_level = 1
    num_level = 1
    distribution_mode = 'easy'
    save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_55664_num_env_200_entro_0.01_gama_0.999_27-04-2023_18-13-16_noRNN_original"
    saved_epoch = 3048

    logdir = './logs'
    logdir = os.path.join(logdir, "record_traj_" + env_name + "_" + str(start_level))
    utils.cleanup_log_dir(logdir)


    print("logdir: " + logdir)

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(0))

    print('making envs...')
    # Making envs
    envs = ProcgenGym3Env(num=1,
                         env_name=env_name,
                         start_level=start_level,
                         num_levels=num_level,
                         distribution_mode=distribution_mode,
                         use_generated_assets=False,
                         center_agent=True,
                         use_backgrounds=True,
                         restrict_themes=False,
                         use_monochrome_assets=False)

    envs = TrajectoryRecorderWrapper(env=envs, directory=logdir)


    actor_critic = Policy(
        (3,64,64),
        envs.ac_space.eltype,
        base=ImpalaModel,
        base_kwargs={'recurrent': False,'hidden_size': 256})
    actor_critic.to(device)

    # Load previous model
    if (saved_epoch > 0) and save_dir != "":
        save_path = save_dir
        actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
        actor_critic.load_state_dict(actor_critic_weighs['state_dict'])

    rew, obs, first = envs.observe()

    done = [not first_ for first_ in first]
    obs = torch.from_numpy(obs['rgb'])
    plt.imshow(obs[0])
    plt.savefig(logdir + '/fig.png')

    masks = torch.FloatTensor([1.0])
    recurrent_hidden_states = torch.zeros(1, 256)
    # policy rollouts
    actor_critic.eval()
    while not done[0]:
            # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, _, recurrent_hidden_states, _, _, _, _ = actor_critic.act(
                    obs.transpose(1,3).float().to(device), recurrent_hidden_states.to(device), masks.to(device))

        # Observe reward and next obs
        envs.act(action[0].cpu().numpy())
        rew, obs, done = envs.observe()

        # If done then clean the history of observations.
        masks = torch.FloatTensor(
            [[1.0] if done_ else [0.0] for done_ in done])
        obs = torch.from_numpy(obs['rgb'])

    print('done recording trajectories...')


if __name__ == "__main__":
    main()
