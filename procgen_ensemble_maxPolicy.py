import copy
import glob
import os
import time
from collections import deque
import sys

from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
# from a2c_ppo_acktr.envs import make_vec_envs, make_ProcgenEnvs
from a2c_ppo_acktr.envs import make_ProcgenEnvs
from procgen import ProcgenEnv
from a2c_ppo_acktr.model import Policy, MLPAttnBase, MLPHardAttnBase, MLPHardAttnReinforceBase, ImpalaModel
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate_procgen, evaluate_procgen_ensemble, evaluate_procgen_LEEP
from a2c_ppo_acktr.utils import save_obj, load_obj
from a2c_ppo_acktr.procgen_wrappers import *
from a2c_ppo_acktr.logger import Logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from a2c_ppo_acktr.utils import init
import wandb
from a2c_ppo_acktr.distributions import FixedCategorical

EVAL_ENVS = ['train_eval','test_eval']
EVAL_ENVS_nondet = ['train_eval_nondet','test_eval_nondet']
# EVAL_ENVS = ['test_eval']

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                       constant_(x, 0), nn.init.calculate_gain('relu'))
init_2 = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=1)

def main():
    args = get_args()
    # import random; random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logdir_ = args.env_name + '_seed_' + str(args.seed) + '_num_env_' + str(args.num_level) + '_entro_' + str(args.entropy_coef) + '_gama_' + str(args.gamma) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    if args.normalize_rew:
        logdir_ = logdir_ + '_normalize_rew'
    if not args.recurrent_policy:
        logdir_ = logdir_ + '_noRNN'
    if args.mask_all:
        logdir_ = logdir_ + '_mask_all'
    if args.mask_size > 0:
        logdir_ = logdir_ + '_mask_' + str(args.mask_size)
    if not args.use_generated_assets and args.use_backgrounds and not args.restrict_themes and not args.use_monochrome_assets:
        logdir_ = logdir_ + '_original'

    # logdir = os.path.join(os.path.expanduser(args.log_dir), logdir_)
    # utils.cleanup_log_dir(logdir)

    wandb.init(project=args.env_name + "_PPO_Ensemble_maxPolicy", entity="ev_zisselman", config=args, name=logdir_, id=logdir_)

    # # Ugly but simple logging
    # log_dict = {
    #     'num_steps': args.num_steps,
    #     'seed': args.seed,
    #     'recurrent': args.recurrent_policy,
    #     'train_env': args.env_name,
    #     'test_env': args.val_env_name,
    #     'cmd': ' '.join(sys.argv)
    # }
    # for eval_disp_name in EVAL_ENVS:
    #     log_dict[eval_disp_name] = []
    #
    #
    # argslog = pd.DataFrame(columns=['args', 'value'])
    # for key in vars(args):
    #     log = [key] + [vars(args)[key]]
    #     argslog.loc[len(argslog)] = log
    #
    # print("logdir: " + logdir)
    # for key in vars(args):
    #     print(key, ':', vars(args)[key])
    #
    # with open(logdir + '/args.csv', 'w') as f:
    #     argslog.to_csv(f, index=False)
    #
    #
    # # Tensorboard logging
    # summary_writer = SummaryWriter(log_dir=logdir)
    #
    # summary_writer.add_hparams(vars(args), {})

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.gpu_device) if args.cuda else "cpu")

    print('making envs...')
    # Training envs
    envs = make_ProcgenEnvs(num_envs=args.num_processes,
                      env_name=args.env_name,
                      start_level=args.start_level,
                      num_levels=args.num_level,
                      distribution_mode=args.distribution_mode,
                      use_generated_assets=args.use_generated_assets,
                      use_backgrounds=args.use_backgrounds,
                      restrict_themes=args.restrict_themes,
                      use_monochrome_assets=args.use_monochrome_assets,
                      center_agent=args.center_agent,
                      rand_seed=args.seed,
                      mask_size=args.mask_size,
                      normalize_rew=args.normalize_rew,
                      mask_all=args.mask_all,
                      device=device)

    # Test envs
    eval_envs_dic = {}
    eval_envs_dic['train_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                      env_name=args.env_name,
                                                      start_level=args.start_level,
                                                      num_levels=args.num_level,
                                                      distribution_mode=args.distribution_mode,
                                                      use_generated_assets=args.use_generated_assets,
                                                      use_backgrounds=args.use_backgrounds,
                                                      restrict_themes=args.restrict_themes,
                                                      use_monochrome_assets=args.use_monochrome_assets,
                                                      center_agent=args.center_agent,
                                                      rand_seed=args.seed,
                                                      mask_size=args.mask_size,
                                                      normalize_rew= args.normalize_rew,
                                                      mask_all=args.mask_all,
                                                      device=device)

    test_start_level = args.start_level + args.num_level + 1
    eval_envs_dic['test_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                     env_name=args.env_name,
                                                     start_level=test_start_level,
                                                     num_levels=0,
                                                     distribution_mode=args.distribution_mode,
                                                     use_generated_assets=args.use_generated_assets,
                                                     use_backgrounds=args.use_backgrounds,
                                                     restrict_themes=args.restrict_themes,
                                                     use_monochrome_assets=args.use_monochrome_assets,
                                                     center_agent=args.center_agent,
                                                     rand_seed=args.seed,
                                                     mask_size=args.mask_size,
                                                     normalize_rew=args.normalize_rew,
                                                     mask_all=args.mask_all,
                                                     device=device)

    eval_envs_dic_nondet = {}
    eval_envs_dic_nondet['test_eval_nondet'] =  make_ProcgenEnvs(num_envs=args.num_processes,
                                                     env_name=args.env_name,
                                                     start_level=test_start_level,
                                                     num_levels=0,
                                                     distribution_mode=args.distribution_mode,
                                                     use_generated_assets=args.use_generated_assets,
                                                     use_backgrounds=args.use_backgrounds,
                                                     restrict_themes=args.restrict_themes,
                                                     use_monochrome_assets=args.use_monochrome_assets,
                                                     center_agent=args.center_agent,
                                                     rand_seed=args.seed,
                                                     mask_size=args.mask_size,
                                                     normalize_rew=args.normalize_rew,
                                                     mask_all=args.mask_all,
                                                     device=device)

    print('done')

    actor_critic = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False,'hidden_size': args.recurrent_hidden_size})
        # base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic.to(device)

    actor_critic1 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False,'hidden_size': args.recurrent_hidden_size})
        # base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic1.to(device)

    actor_critic2 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False,'hidden_size': args.recurrent_hidden_size})
    # base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic2.to(device)

    actor_critic3 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False,'hidden_size': args.recurrent_hidden_size})
    # base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic3.to(device)

    actor_critic4 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False, 'hidden_size': args.recurrent_hidden_size})
    actor_critic4.to(device)

    actor_critic5 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False, 'hidden_size': args.recurrent_hidden_size})
    actor_critic5.to(device)

    actor_critic6 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False, 'hidden_size': args.recurrent_hidden_size})
    actor_critic6.to(device)

    actor_critic7 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False, 'hidden_size': args.recurrent_hidden_size})
    actor_critic7.to(device)

    actor_critic8 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False, 'hidden_size': args.recurrent_hidden_size})
    actor_critic8.to(device)

    actor_critic9 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False, 'hidden_size': args.recurrent_hidden_size})
    actor_critic9.to(device)

    actor_critic_maxEnt = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': True,'hidden_size': args.recurrent_hidden_size})
        # base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic_maxEnt.to(device)


    if args.algo != 'ppo':
        raise print("only PPO is supported")


    # training agent
    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        num_tasks=args.num_processes,
        attention_policy=False,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # rollout storage for agent
    # rollouts = RolloutStorage(args.num_steps, args.num_processes,
    #                           envs.observation_space.shape, envs.action_space,
    #                           actor_critic.recurrent_hidden_state_size, args.mask_size, device=device)
    #
    # rollouts1 = RolloutStorage(args.num_steps, args.num_processes,
    #                           envs.observation_space.shape, envs.action_space,
    #                           actor_critic.recurrent_hidden_state_size, args.mask_size, device=device)
    #
    # rollouts2 = RolloutStorage(args.num_steps, args.num_processes,
    #                           envs.observation_space.shape, envs.action_space,
    #                           actor_critic.recurrent_hidden_state_size, args.mask_size, device=device)
    #
    # rollouts3 = RolloutStorage(args.num_steps, args.num_processes,
    #                           envs.observation_space.shape, envs.action_space,
    #                           actor_critic.recurrent_hidden_state_size, args.mask_size, device=device)

    rollouts_maxEnt = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.observation_space.shape, envs.action_space,
                              actor_critic_maxEnt.recurrent_hidden_state_size, args.mask_size, device=device)

    # Load previous model
    if (args.continue_from_epoch > 0) and args.save_dir != "":
        save_path = args.save_dir
        actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(args.continue_from_epoch)), map_location=device)
        actor_critic.load_state_dict(actor_critic_weighs['state_dict'])
        agent.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict'])
        # rollouts.obs                            = actor_critic_weighs['buffer_obs']
        # rollouts.recurrent_hidden_states = actor_critic_weighs['buffer_recurrent_hidden_states']
        # rollouts.rewards                 = actor_critic_weighs['buffer_rewards']
        # rollouts.seeds                   = actor_critic_weighs['buffer_seeds']
        # rollouts.value_preds             = actor_critic_weighs['buffer_value_preds']
        # rollouts.returns                 = actor_critic_weighs['buffer_returns']
        # rollouts.action_log_probs        = actor_critic_weighs['buffer_action_log_probs']
        # rollouts.actions                 = actor_critic_weighs['buffer_actions']
        # rollouts.masks                   = actor_critic_weighs['buffer_masks']
        # rollouts.bad_masks               = actor_critic_weighs['buffer_bad_masks']
        # rollouts.info_batch              = actor_critic_weighs['buffer_info_batch']
        # rollouts.num_steps               = actor_critic_weighs['buffer_num_steps']
        # rollouts.step                    = actor_critic_weighs['buffer_step']
        # rollouts.num_processes           = actor_critic_weighs['buffer_num_processes']

    # Load previous model
    if (args.saved_epoch > 0) and args.save_dir != "":
        save_path = args.save_dir
        actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch)), map_location=device)
        actor_critic.load_state_dict(actor_critic_weighs['state_dict'])

    if (args.saved_epoch1 > 0) and args.save_dir1 != "":
        save_path = args.save_dir1
        actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch1)), map_location=device)
        actor_critic1.load_state_dict(actor_critic_weighs['state_dict'])

    if (args.saved_epoch2 > 0) and args.save_dir2 != "":
        save_path = args.save_dir2
        actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch2)), map_location=device)
        actor_critic2.load_state_dict(actor_critic_weighs['state_dict'])

    if (args.saved_epoch3 > 0) and args.save_dir3 != "":
        save_path = args.save_dir3
        actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch3)), map_location=device)
        actor_critic3.load_state_dict(actor_critic_weighs['state_dict'])

    # if (args.saved_epoch_maxEnt > 0) and args.save_dir_maxEnt != "":
    #     save_path = args.save_dir_maxEnt
    #     actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch_maxEnt)), map_location=device)
    #     actor_critic_maxEnt.load_state_dict(actor_critic_weighs['state_dict'])

    if args.num_ensemble > 4:
        if (args.saved_epoch4 > 0) and args.save_dir4 != "":
            save_path = args.save_dir4
            actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch4)),map_location=device)
            actor_critic4.load_state_dict(actor_critic_weighs['state_dict'])

        if (args.saved_epoch5 > 0) and args.save_dir5 != "":
            save_path = args.save_dir5
            actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch5)),map_location=device)
            actor_critic5.load_state_dict(actor_critic_weighs['state_dict'])

    if args.num_ensemble > 6:
        if (args.saved_epoch6 > 0) and args.save_dir6 != "":
            save_path = args.save_dir6
            actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch6)),map_location=device)
            actor_critic6.load_state_dict(actor_critic_weighs['state_dict'])

        if (args.saved_epoch7 > 0) and args.save_dir7 != "":
            save_path = args.save_dir7
            actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch7)),map_location=device)
            actor_critic7.load_state_dict(actor_critic_weighs['state_dict'])

    if args.num_ensemble > 8:
        if (args.saved_epoch8 > 0) and args.save_dir8 != "":
            save_path = args.save_dir8
            actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch8)),map_location=device)
            actor_critic8.load_state_dict(actor_critic_weighs['state_dict'])

        if (args.saved_epoch9 > 0) and args.save_dir9 != "":
            save_path = args.save_dir9
            actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch9)),map_location=device)
            actor_critic9.load_state_dict(actor_critic_weighs['state_dict'])

    logger = Logger(args.num_processes, envs.observation_space.shape, envs.observation_space.shape, actor_critic_maxEnt.recurrent_hidden_state_size, device=device)

    obs = envs.reset()
    # rollouts.obs[0].copy_(torch.FloatTensor(obs))
    # rollouts.obs[0].copy_(obs)
    # rollouts1.obs[0].copy_(obs)
    # rollouts2.obs[0].copy_(obs)
    # rollouts3.obs[0].copy_(obs)
    rollouts_maxEnt.obs[0].copy_(obs)
    # rollouts.to(device)

    obs_train = eval_envs_dic['train_eval'].reset()
    logger.obs['train_eval'].copy_(obs_train)
    logger.obs_sum['train_eval'].copy_(obs_train)

    obs_test = eval_envs_dic['test_eval'].reset()
    logger.obs['test_eval'].copy_(obs_test)
    logger.obs_sum['test_eval'].copy_(obs_test)

    obs_test_nondet = eval_envs_dic_nondet['test_eval_nondet'].reset()
    logger.obs['test_eval_nondet'].copy_(obs_test_nondet)
    logger.obs_sum['test_eval_nondet'].copy_(obs_test_nondet)

    # fig = plt.figure(figsize=(20, 20))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(obs_train[0][i].transpose(0,2))
    # plt.show()

    seeds = torch.zeros(args.num_processes, 1)
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    # save_copy = True
    save_image_every = int(num_updates/10)
    # episode_len_buffer = []
    # for _ in range(args.num_processes):
    #     episode_len_buffer.append(0)
    seeds_train = np.zeros((args.num_steps, args.num_processes))
    seeds_test = np.zeros((args.num_steps, args.num_processes))

    #freeze layers
    # if args.freeze1:
    #     for name, param in actor_critic.base.main[0].named_parameters():
    #         param.requires_grad = False
    # if args.freeze2:
    #     for name, param in actor_critic.base.main[0].named_parameters():
    #         param.requires_grad = False
    #     for name, param in actor_critic.base.main[1].named_parameters():
    #         param.requires_grad = False
    # if args.freeze_all:
    #     for name, param in actor_critic.base.main.named_parameters():
    #         param.requires_grad = False
    # if args.freeze_all_gru:
    #     for name, param in actor_critic.base.main.named_parameters():
    #         param.requires_grad = False
    #     for name, param in actor_critic.base.gru.named_parameters():
    #         param.requires_grad = False
    #
    # is_novel = torch.ones(args.num_processes,1,dtype=torch.bool, device=device)
    # m = FixedCategorical(torch.tensor([0.55, 0.25, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]).repeat(args.num_processes, 1)) # worked for maze #approximrtly Geometric distribution with \alpha = 0.5
    # # m = FixedCategorical(torch.tensor([0.75, 0.15, 0.05, 0.05]).repeat(num_processes, 1))
    # rand_action = FixedCategorical(torch.tensor([ 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 1-14*0.067]).repeat(args.num_processes, 1))
    # # m = FixedCategorical(torch.tensor([ 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 1-14*0.067]).repeat(args.num_processes, 1))
    # maxEnt_steps = torch.zeros(args.num_processes,1, device=device)


    for j in range(args.continue_from_epoch, args.continue_from_epoch+num_updates):
        #  plot mazes
        # if j % save_image_every == 0:
        #     fig = plt.figure(figsize=(20, 20))
        #     columns = 5
        #     rows = 5
        #     for i in range(1, columns * rows + 1):
        #         fig.add_subplot(rows, columns, i)
        #         plt.imshow(rollouts.obs[0][i].transpose(0,2))
        #     summary_writer.add_images('samples_step_{}'.format(j), rollouts.obs[0][0:25], global_step=(j) * args.num_processes * args.num_steps)
        #     plt.show()

        # policy rollouts
        actor_critic.eval()
        # episode_rewards = []
        # episode_len = []
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action0, action_log_prob, dist_probs, recurrent_hidden_states, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic.act(
                    rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                    rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                    rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                    rollouts_maxEnt.attn_masks3[step].to(device))

                value, action1, action_log_prob, dist_probs_1, recurrent_hidden_states1, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic1.act(
                    rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                    rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                    rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                    rollouts_maxEnt.attn_masks3[step].to(device))

                value, action2, action_log_prob, dist_probs_2, recurrent_hidden_states2, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic2.act(
                    rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                    rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                    rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                    rollouts_maxEnt.attn_masks3[step].to(device))

                value, action3, action_log_prob, dist_probs_3, recurrent_hidden_states3, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic3.act(
                    rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                    rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                    rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                    rollouts_maxEnt.attn_masks3[step].to(device))

                value, action_maxEnt, action_log_prob, _, recurrent_hidden_states_maxEnt, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic_maxEnt.act(
                    rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                    rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                    rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                    rollouts_maxEnt.attn_masks3[step].to(device))

                if args.num_ensemble > 4:
                    value, action4, action_log_prob, dist_probs_4, _, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic4.act(
                        rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                        rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                        rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                        rollouts_maxEnt.attn_masks3[step].to(device))

                    value, action5, action_log_prob, dist_probs_5, _, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic5.act(
                        rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                        rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                        rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                        rollouts_maxEnt.attn_masks3[step].to(device))

                if args.num_ensemble > 6:
                    value, action6, action_log_prob, dist_probs_6, recurrent_hidden_states2, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic6.act(
                        rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                        rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                        rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                        rollouts_maxEnt.attn_masks3[step].to(device))

                    value, action7, action_log_prob, dist_probs_7, recurrent_hidden_states3, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic7.act(
                        rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                        rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                        rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                        rollouts_maxEnt.attn_masks3[step].to(device))

                if args.num_ensemble > 8:
                    value, action8, action_log_prob, dist_probs_8, recurrent_hidden_states2, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic8.act(
                        rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                        rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                        rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                        rollouts_maxEnt.attn_masks3[step].to(device))

                    value, action9, action_log_prob, dist_probs_9, recurrent_hidden_states3, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic9.act(
                        rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                        rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                        rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                        rollouts_maxEnt.attn_masks3[step].to(device))

            max_policy = torch.max(torch.max(torch.max(dist_probs, dist_probs_1), dist_probs_2), dist_probs_3)
            if args.num_ensemble > 4:
                max_policy = torch.max(torch.max(max_policy, dist_probs_4), dist_probs_5)
            if args.num_ensemble > 6:
                max_policy = torch.max(torch.max(max_policy, dist_probs_6), dist_probs_7)
            if args.num_ensemble > 8:
                max_policy = torch.max(torch.max(max_policy, dist_probs_8), dist_probs_9)

            max_policy = torch.div(max_policy, max_policy.sum(1).unsqueeze(1))

            x = FixedCategorical(logits=max_policy)
            action = x.sample()

            # if args.num_ensemble > 4:
            #     actions.append(action4)
            #     actions.append(action5)
            #     cardinal_left += 1 * (action4 == 0) + 1 * (action4 == 1) + 1 * (action4 == 2) + 1 * (action5 == 0) + 1 * (action5 == 1) + 1 * (action5 == 2)
            #     cardinal_right += 1 * (action4 == 6) + 1 * (action4 == 7) + 1 * (action4 == 8) + 1 * (action5 == 6) + 1 * (action5 == 7) + 1 * (action5 == 8)
            #     if (args.env_name == "maze" or args.env_name == "miner"):
            #         cardinal_down += 1 * (action4 == 3) + 1 * (action5 == 3)
            #         cardinal_up += 1 * (action4 == 5) + 1 * (action5 == 5)
            #     else:
            #         cardinal_down += 1 * (action4 == 3) + 1 * (action5 == 3) + 1 * (action4 == 0) + 1 * (action5 == 0) + 1 * (action4 == 6) + 1 * (action5 == 6)
            #         cardinal_up += 1 * (action4 == 5) + 1 * (action5 == 5) + 1 * (action4 == 2) + 1 * (action5 == 2) + 1 * (action4 == 8) + 1 * (action5 == 8)
            #         cardinal_fire += 1 * (action4 == 9) + 1 * (action5 == 9)
            #         cardinal_else = 1 * (action4 == 4) + 1 * (action4 == 10) + 1 * (action4 == 11) + 1 * (action4 == 12) + 1 * (action4 == 13) + 1 * (action4 == 14) \
            #                       + 1 * (action5 == 9) + 1 * (action5 == 10) + 1 * (action5 == 11) + 1 * (action5 == 12) + 1 * (action5 == 13) + 1 * (action5 == 14)
            #
            # if args.num_ensemble > 6:
            #     actions.append(action6)
            #     actions.append(action7)
            #     cardinal_left += 1 * (action6 == 0) + 1 * (action6 == 1) + 1 * (action6 == 2) + 1 * (action7 == 0) + 1 * (action7 == 1) + 1 * (action7 == 2)
            #     cardinal_right += 1 * (action6 == 6) + 1 * (action6 == 7) + 1 * (action6 == 8) + 1 * (action7 == 6) + 1 * (action7 == 7) + 1 * (action7 == 8)
            #     if (args.env_name == "maze" or args.env_name == "miner"):
            #         cardinal_down += 1 * (action6 == 3) + 1 * (action7 == 3)
            #         cardinal_up += 1 * (action6 == 5) + 1 * (action7 == 5)
            #     else:
            #         cardinal_down += 1 * (action6 == 3) + 1 * (action7 == 3) + 1 * (action6 == 0) + 1 * (action7 == 0) + 1 * (action6 == 6) + 1 * (action7 == 6)
            #         cardinal_up += 1 * (action6 == 5) + 1 * (action7 == 5) + 1 * (action6 == 2) + 1 * (action7 == 2) + 1 * (action6 == 8) + 1 * (action7 == 8)
            #         cardinal_fire += 1 * (action6 == 9) + 1 * (action7 == 9)
            #         cardinal_else = 1 * (action6 == 4) + 1 * (action6 == 10) + 1 * (action6 == 11) + 1 * (action6 == 12) + 1 * (action6 == 13) + 1 * (action6 == 14) \
            #                       + 1 * (action7 == 9) + 1 * (action7 == 10) + 1 * (action7 == 11) + 1 * (action7 == 12) + 1 * (action7 == 13) + 1 * (action7 == 14)
            #
            # if args.num_ensemble > 8:
            #     actions.append(action8)
            #     actions.append(action9)
            #     cardinal_left += 1 * (action8 == 0) + 1 * (action8 == 1) + 1 * (action8 == 2) + 1 * (action9 == 0) + 1 * (action9 == 1) + 1 * (action9 == 2)
            #     cardinal_right += 1 * (action8 == 6) + 1 * (action8 == 7) + 1 * (action8 == 8) + 1 * (action9 == 6) + 1 * (action9 == 7) + 1 * (action9 == 8)
            #     if (args.env_name == "maze" or args.env_name == "miner"):
            #         cardinal_down += 1 * (action8 == 3) + 1 * (action9 == 3)
            #         cardinal_up += 1 * (action8 == 5) + 1 * (action9 == 5)
            #     else:
            #         cardinal_down += 1 * (action8 == 3) + 1 * (action9 == 3) + 1 * (action8 == 0) + 1 * (action9 == 0) + 1 * (action8 == 6) + 1 * (action9 == 6)
            #         cardinal_up += 1 * (action8 == 5) + 1 * (action9 == 5) + 1 * (action8 == 2) + 1 * (action9 == 2) + 1 * (action8 == 8) + 1 * (action9 == 8)
            #         cardinal_fire += 1 * (action8 == 9) + 1 * (action9 == 9)
            #         cardinal_else = 1 * (action8 == 4) + 1 * (action8 == 10) + 1 * (action8 == 11) + 1 * (action8 == 12) + 1 * (action8 == 13) + 1 * (action8 == 14) \
            #                       + 1 * (action9 == 9) + 1 * (action9 == 10) + 1 * (action9 == 11) + 1 * (action9 == 12) + 1 * (action9 == 13) + 1 * (action9 == 14)

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action.squeeze().cpu().numpy())

            for i, info in enumerate(infos):
                seeds[i] = info["level_seed"]

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts_maxEnt.insert(obs, recurrent_hidden_states_maxEnt, action,
                            action_log_prob, value, torch.from_numpy(reward).unsqueeze(1), masks, bad_masks, attn_masks,
                            attn_masks1, attn_masks2, attn_masks3, seeds, infos, obs)


        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts_maxEnt.obs[-1].to(device), rollouts_maxEnt.recurrent_hidden_states[-1].to(device),
                rollouts_maxEnt.masks[-1].to(device), rollouts_maxEnt.attn_masks[-1].to(device), rollouts_maxEnt.attn_masks1[-1].to(device),
                rollouts_maxEnt.attn_masks2[-1].to(device), rollouts_maxEnt.attn_masks3[-1].to(device)).detach()

        actor_critic.train()
        rollouts_maxEnt.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        # value_loss, action_loss, dist_entropy, _ = agent.update(rollouts)

        rollouts_maxEnt.after_update()

        rew_batch, done_batch = rollouts_maxEnt.fetch_log_data()
        logger.feed_train(rew_batch, done_batch[1:])

        actor_critic.eval()
        actor_critic1.eval()
        actor_critic2.eval()
        actor_critic3.eval()
        actor_critic4.eval()
        actor_critic5.eval()
        actor_critic6.eval()
        actor_critic7.eval()
        actor_critic8.eval()
        actor_critic9.eval()
        actor_critic_maxEnt.eval()
        maze_miner = False
        if (args.env_name == "maze" or args.env_name == "miner"):
            maze_miner = True

        eval_dic_rew = {}
        eval_dic_done = {}
        for eval_disp_name in EVAL_ENVS:
            eval_dic_rew[eval_disp_name], _, eval_dic_done[eval_disp_name], \
                _ = evaluate_procgen_LEEP(actor_critic, actor_critic1, actor_critic2,
                                          actor_critic3,
                                          eval_envs_dic, eval_disp_name, args.num_processes,
                                          device, args.num_steps, logger, num_ensemble=args.num_ensemble,
                                          actor_critic_4=actor_critic4,
                                          actor_critic_5=actor_critic5,
                                          actor_critic_6=actor_critic6,
                                          actor_critic_7=actor_critic7,
                                          actor_critic_8=actor_critic8,
                                          actor_critic_9=actor_critic9)

        eval_test_nondet_rew, _, eval_test_nondet_done, _ = evaluate_procgen_LEEP(actor_critic, actor_critic1,
                                                                                  actor_critic2, actor_critic3,
                                                                                  eval_envs_dic, 'test_eval',
                                                                                  args.num_processes, device,
                                                                                  args.num_steps, logger,
                                                                                  deterministic=False, num_ensemble=args.num_ensemble,
                                                                                  actor_critic_4=actor_critic4,
                                                                                  actor_critic_5=actor_critic5,
                                                                                  actor_critic_6=actor_critic6,
                                                                                  actor_critic_7=actor_critic7,
                                                                                  actor_critic_8=actor_critic8,
                                                                                  actor_critic_9=actor_critic9)

        logger.feed_eval(eval_dic_rew['train_eval'], eval_dic_done['train_eval'], eval_dic_rew['test_eval'], eval_dic_done['test_eval'], seeds_train, seeds_test,
                         eval_dic_rew['train_eval'], eval_dic_rew['test_eval'], eval_test_nondet_rew, eval_test_nondet_done)
        if len(logger.episode_reward_buffer)>0 and len(logger.episode_reward_buffer_test_nondet)>0:
            episode_statistics = logger.get_episode_statistics()
            print("train and test eval")
            print(episode_statistics)


            if (len(episode_statistics)>0):
                for key, value in episode_statistics.items():
                    if isinstance(value, dict):
                        # summary_writer.add_scalars(key, value,(j + 1) * args.num_processes * args.num_steps)
                        for key_v, value_v in value.items():
                             wandb.log({key + "/" + key_v: value_v}, step=(j + 1) * args.num_processes * args.num_steps)

    for eval_disp_name in EVAL_ENVS:
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
