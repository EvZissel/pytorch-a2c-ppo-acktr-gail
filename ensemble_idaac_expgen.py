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
from a2c_ppo_acktr.model_daac_idaac import IDAACnet, LinearOrderClassifier, NonlinearOrderClassifier
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.storage_daac_idaac import DAACRolloutStorage, IDAACRolloutStorage
from evaluation import evaluate_procgen, evaluate_procgen_ensemble
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

    wandb.init(project=args.env_name + "_PPO_Ensemble_idaac_expgen", entity="ev_zisselman", config=args, name=logdir_, id=logdir_)

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

    obs_shape = envs.observation_space.shape
    actor_critic = IDAACnet(
        obs_shape,
        envs.action_space.n,
        base_kwargs={'hidden_size': args.hidden_size})
    actor_critic.to(device)

    actor_critic1 = IDAACnet(
        obs_shape,
        envs.action_space.n,
        base_kwargs={'hidden_size': args.hidden_size})
    actor_critic.to(device)

    actor_critic2 = IDAACnet(
        obs_shape,
        envs.action_space.n,
        base_kwargs={'hidden_size': args.hidden_size})
    actor_critic.to(device)

    actor_critic3 = IDAACnet(
        obs_shape,
        envs.action_space.n,
        base_kwargs={'hidden_size': args.hidden_size})
    actor_critic.to(device)

    actor_critic4 = IDAACnet(
        obs_shape,
        envs.action_space.n,
        base_kwargs={'hidden_size': args.hidden_size})
    actor_critic.to(device)

    actor_critic5 = IDAACnet(
        obs_shape,
        envs.action_space.n,
        base_kwargs={'hidden_size': args.hidden_size})
    actor_critic.to(device)

    actor_critic6 = IDAACnet(
        obs_shape,
        envs.action_space.n,
        base_kwargs={'hidden_size': args.hidden_size})
    actor_critic.to(device)

    actor_critic7 = IDAACnet(
        obs_shape,
        envs.action_space.n,
        base_kwargs={'hidden_size': args.hidden_size})
    actor_critic.to(device)

    actor_critic8 = IDAACnet(
        obs_shape,
        envs.action_space.n,
        base_kwargs={'hidden_size': args.hidden_size})
    actor_critic.to(device)

    actor_critic9 = IDAACnet(
        obs_shape,
        envs.action_space.n,
        base_kwargs={'hidden_size': args.hidden_size})
    actor_critic.to(device)

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
    # if (args.continue_from_epoch > 0) and args.save_dir != "":
    #     save_path = args.save_dir
    #     actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(args.continue_from_epoch)), map_location=device)
    #     actor_critic.load_state_dict(actor_critic_weighs['state_dict'])
    #     agent.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict'])
    #     # rollouts.obs                            = actor_critic_weighs['buffer_obs']
    #     # rollouts.recurrent_hidden_states = actor_critic_weighs['buffer_recurrent_hidden_states']
    #     # rollouts.rewards                 = actor_critic_weighs['buffer_rewards']
    #     # rollouts.seeds                   = actor_critic_weighs['buffer_seeds']
    #     # rollouts.value_preds             = actor_critic_weighs['buffer_value_preds']
    #     # rollouts.returns                 = actor_critic_weighs['buffer_returns']
    #     # rollouts.action_log_probs        = actor_critic_weighs['buffer_action_log_probs']
    #     # rollouts.actions                 = actor_critic_weighs['buffer_actions']
    #     # rollouts.masks                   = actor_critic_weighs['buffer_masks']
    #     # rollouts.bad_masks               = actor_critic_weighs['buffer_bad_masks']
    #     # rollouts.info_batch              = actor_critic_weighs['buffer_info_batch']
    #     # rollouts.num_steps               = actor_critic_weighs['buffer_num_steps']
    #     # rollouts.step                    = actor_critic_weighs['buffer_step']
    #     # rollouts.num_processes           = actor_critic_weighs['buffer_num_processes']

    # Load previous model
    if (args.saved_epoch > 0) and args.save_dir != "":
        save_path = args.save_dir
        actor_critic_weighs = torch.load(os.path.join(save_path,  "agent-{}-idaac-s{}iter1525.pt".format(args.env_name, args.seed0)), map_location=device)
        actor_critic.load_state_dict(actor_critic_weighs['state_dict'])


    if (args.saved_epoch1 > 0) and args.save_dir1 != "":
        save_path = args.save_dir1
        actor_critic_weighs = torch.load(os.path.join(save_path,  "agent-{}-idaac-s{}iter1525.pt".format(args.env_name, args.seed1)), map_location=device)
        actor_critic1.load_state_dict(actor_critic_weighs['state_dict'])

    if (args.saved_epoch2 > 0) and args.save_dir2 != "":
        save_path = args.save_dir2
        actor_critic_weighs = torch.load(os.path.join(save_path,  "agent-{}-idaac-s{}iter1525.pt".format(args.env_name, args.seed2)), map_location=device)
        actor_critic2.load_state_dict(actor_critic_weighs['state_dict'])

    if (args.saved_epoch3 > 0) and args.save_dir3 != "":
        save_path = args.save_dir3
        actor_critic_weighs = torch.load(os.path.join(save_path,  "agent-{}-idaac-s{}iter1525.pt".format(args.env_name, args.seed3)), map_location=device)
        actor_critic3.load_state_dict(actor_critic_weighs['state_dict'])

    if (args.saved_epoch_maxEnt > 0) and args.save_dir_maxEnt != "":
        save_path = args.save_dir_maxEnt
        actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch_maxEnt)), map_location=device)
        actor_critic_maxEnt.load_state_dict(actor_critic_weighs['state_dict'])

    if args.num_ensemble > 4:
        if (args.saved_epoch4 > 0) and args.save_dir4 != "":
            save_path = args.save_dir4
            actor_critic_weighs = torch.load(os.path.join(save_path,  "agent-{}-idaac-s{}iter1525.pt".format(args.env_name, args.seed4)), map_location=device)
            actor_critic4.load_state_dict(actor_critic_weighs['state_dict'])

        if (args.saved_epoch5 > 0) and args.save_dir5 != "":
            save_path = args.save_dir5
            actor_critic_weighs = torch.load(os.path.join(save_path,  "agent-{}-idaac-s{}iter1525.pt".format(args.env_name, args.seed5)), map_location=device)
            actor_critic5.load_state_dict(actor_critic_weighs['state_dict'])

    if args.num_ensemble > 6:
        if (args.saved_epoch6 > 0) and args.save_dir6 != "":
            save_path = args.save_dir6
            actor_critic_weighs = torch.load(os.path.join(save_path,  "agent-{}-idaac-s{}iter1525.pt".format(args.env_name, args.seed6)), map_location=device)
            actor_critic6.load_state_dict(actor_critic_weighs['state_dict'])

        if (args.saved_epoch7 > 0) and args.save_dir7 != "":
            save_path = args.save_dir7
            actor_critic_weighs = torch.load(os.path.join(save_path,  "agent-{}-idaac-s{}iter1525.pt".format(args.env_name, args.seed7)), map_location=device)
            actor_critic7.load_state_dict(actor_critic_weighs['state_dict'])

    if args.num_ensemble > 8:
        if (args.saved_epoch8 > 0) and args.save_dir8 != "":
            save_path = args.save_dir8
            actor_critic_weighs = torch.load(os.path.join(save_path,  "agent-{}-idaac-s{}iter1525.pt".format(args.env_name, args.seed8)), map_location=device)
            actor_critic8.load_state_dict(actor_critic_weighs['state_dict'])

        if (args.saved_epoch9 > 0) and args.save_dir9 != "":
            save_path = args.save_dir9
            actor_critic_weighs = torch.load(os.path.join(save_path,  "agent-{}-idaac-s{}iter1525.pt".format(args.env_name, args.seed9)), map_location=device)
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
    is_novel = torch.ones(args.num_processes,1,dtype=torch.bool, device=device)
    m = FixedCategorical(torch.tensor([0.55, 0.25, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]).repeat(args.num_processes, 1)) # worked for maze #approximrtly Geometric distribution with \alpha = 0.5
    # m = FixedCategorical(torch.tensor([0.75, 0.15, 0.05, 0.05]).repeat(num_processes, 1))
    rand_action = FixedCategorical(torch.tensor([ 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 1-14*0.067]).repeat(args.num_processes, 1))
    # m = FixedCategorical(torch.tensor([ 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 1-14*0.067]).repeat(args.num_processes, 1))
    maxEnt_steps = torch.zeros(args.num_processes,1, device=device)


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
                adv, value, action0, action_log_prob = actor_critic.act(rollouts_maxEnt.obs[step].to(device))
                adv, value, action1, action_log_prob = actor_critic1.act(rollouts_maxEnt.obs[step].to(device))
                adv, value, action2, action_log_prob = actor_critic2.act(rollouts_maxEnt.obs[step].to(device))
                adv, value, action3, action_log_prob = actor_critic3.act(rollouts_maxEnt.obs[step].to(device))

                value, action_maxEnt, action_log_prob, _, recurrent_hidden_states_maxEnt, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic_maxEnt.act(
                    rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                    rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                    rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                    rollouts_maxEnt.attn_masks3[step].to(device))

                if args.num_ensemble > 4:
                    adv, value, action4, action_log_prob = actor_critic4.act(rollouts_maxEnt.obs[step].to(device))
                    adv, value, action5, action_log_prob = actor_critic5.act(rollouts_maxEnt.obs[step].to(device))

                if args.num_ensemble > 6:
                    adv, value, action6, action_log_prob = actor_critic6.act(rollouts_maxEnt.obs[step].to(device))
                    adv, value, action7, action_log_prob = actor_critic7.act(rollouts_maxEnt.obs[step].to(device))

                if args.num_ensemble > 8:
                    adv, value, action8, action_log_prob = actor_critic8.act(rollouts_maxEnt.obs[step].to(device))
                    adv, value, action9, action_log_prob = actor_critic9.act(rollouts_maxEnt.obs[step].to(device))

            # actions_vec = torch.zeros([args.num_processes, envs.action_space.n], device=device)
            # for i in range(args.num_processes):
            #     actions_vec[i, action0[i]] += 1
            #     actions_vec[i, action1[i]] += 1
            #     actions_vec[i, action2[i]] += 1
            #     actions_vec[i, action3[i]] += 1
            #     if args.num_ensemble > 4:
            #         actions_vec[i, action4[i]] += 1
            #         actions_vec[i, action5[i]] += 1
            #     if args.num_ensemble > 6:
            #         actions_vec[i, action6[i]] += 1
            #         actions_vec[i, action7[i]] += 1
            #     if args.num_ensemble > 8:
            #         actions_vec[i, action8[i]] += 1
            #         actions_vec[i, action9[i]] += 1
            #
            # actions_max = actions_vec.max(1)
            # cardinal_value = actions_max[0]
            # cardinal_index = actions_max[1]
            # is_majority = (cardinal_value >= args.num_agree).unsqueeze(1)
            # # action_NN = cardinal_index.unsqueeze(1)
            # action_NN = action0

            actions = []
            actions.append(action0)
            actions.append(action1)
            actions.append(action2)
            actions.append(action3)
            cardinal_left = 1*(action0 == 0)+ 1*(action0 == 1) + 1*(action0 == 2) + 1*(action1 == 0)+1*(action1 == 1) + 1*(action1 == 2) + 1*(action2 == 0)+1*(action2 == 1) + 1*(action2 == 2)\
                            + 1 * (action3 == 0) + 1 * (action3 == 1) + 1 * (action3 == 2)
            cardinal_right  = 1*(action0 == 6)+1*(action0 == 7) + 1*(action0 == 8) + 1*(action1 == 6)+1*(action1 == 7) + 1*(action1 == 8) + 1*(action2 == 6)+1*(action2 == 7) + 1*(action2 == 8)\
                            + 1 * (action3 == 6) + 1 * (action3 == 7) + 1 * (action3 == 8)
            if (args.env_name=="maze" or args.env_name=="miner"):
                cardinal_down = 1 * (action0 == 3) + 1 * (action1 == 3) + 1 * (action2 == 3) + 1 * (action3 == 3)
                cardinal_up = 1 * (action0 == 5) + 1 * (action1 == 5) + 1 * (action2 == 5) + 1 * (action3 == 5)
            else:
                cardinal_down  = 1*(action0 == 3) + 1*(action1 == 3) + 1*(action2 == 3) + 1*(action3 == 3) + 1*(action0 == 0) + 1*(action1 == 0) + 1*(action2 == 0) + 1*(action3 == 0)\
                                + 1*(action0 == 6) + 1*(action1 == 6) + 1*(action2 == 6) + 1*(action3 == 6)
                cardinal_up  = 1*(action0 == 5) + 1*(action1 == 5) + 1*(action2 == 5) + 1*(action3 == 5) + 1*(action0 == 2) + 1*(action1 == 2) + 1*(action2 == 2) + 1*(action3 == 2) \
                               + 1 * (action0 == 8) + 1 * (action1 == 8) + 1 * (action2 == 8) + 1 * (action3 == 8)
                cardinal_fire  = 1*(action0 == 9) + 1*(action1 == 9) + 1*(action2 == 9) + 1*(action3 == 9)
                cardinal_else  = 1*(action0 == 4) + 1*(action0 == 10) + 1*(action0 == 11) + 1*(action0 == 12) + 1*(action0 == 13) + 1*(action0 == 14) \
                               + 1*(action1 == 9) + 1*(action1 == 10) + 1*(action1 == 11) + 1*(action1 == 12) + 1*(action1 == 13) + 1*(action1 == 14)  \
                               + 1*(action2 == 9) + 1*(action2 == 10) + 1*(action2 == 11) + 1*(action2 == 12) + 1*(action2 == 13) + 1*(action2 == 14)  \
                               + 1*(action3 == 9) + 1*(action3 == 10) + 1*(action3 == 11) + 1*(action3 == 12) + 1*(action3 == 13) + 1*(action3 == 14)

            if args.num_ensemble > 4:
                actions.append(action4)
                actions.append(action5)
                cardinal_left += 1 * (action4 == 0) + 1 * (action4 == 1) + 1 * (action4 == 2) + 1 * (action5 == 0) + 1 * (action5 == 1) + 1 * (action5 == 2)
                cardinal_right += 1 * (action4 == 6) + 1 * (action4 == 7) + 1 * (action4 == 8) + 1 * (action5 == 6) + 1 * (action5 == 7) + 1 * (action5 == 8)
                if (args.env_name == "maze" or args.env_name == "miner"):
                    cardinal_down += 1 * (action4 == 3) + 1 * (action5 == 3)
                    cardinal_up += 1 * (action4 == 5) + 1 * (action5 == 5)
                else:
                    cardinal_down += 1 * (action4 == 3) + 1 * (action5 == 3) + 1 * (action4 == 0) + 1 * (action5 == 0) + 1 * (action4 == 6) + 1 * (action5 == 6)
                    cardinal_up += 1 * (action4 == 5) + 1 * (action5 == 5) + 1 * (action4 == 2) + 1 * (action5 == 2) + 1 * (action4 == 8) + 1 * (action5 == 8)
                    cardinal_fire += 1 * (action4 == 9) + 1 * (action5 == 9)
                    cardinal_else = 1 * (action4 == 4) + 1 * (action4 == 10) + 1 * (action4 == 11) + 1 * (action4 == 12) + 1 * (action4 == 13) + 1 * (action4 == 14) \
                                  + 1 * (action5 == 9) + 1 * (action5 == 10) + 1 * (action5 == 11) + 1 * (action5 == 12) + 1 * (action5 == 13) + 1 * (action5 == 14)

            if args.num_ensemble > 6:
                actions.append(action6)
                actions.append(action7)
                cardinal_left += 1 * (action6 == 0) + 1 * (action6 == 1) + 1 * (action6 == 2) + 1 * (action7 == 0) + 1 * (action7 == 1) + 1 * (action7 == 2)
                cardinal_right += 1 * (action6 == 6) + 1 * (action6 == 7) + 1 * (action6 == 8) + 1 * (action7 == 6) + 1 * (action7 == 7) + 1 * (action7 == 8)
                if (args.env_name == "maze" or args.env_name == "miner"):
                    cardinal_down += 1 * (action6 == 3) + 1 * (action7 == 3)
                    cardinal_up += 1 * (action6 == 5) + 1 * (action7 == 5)
                else:
                    cardinal_down += 1 * (action6 == 3) + 1 * (action7 == 3) + 1 * (action6 == 0) + 1 * (action7 == 0) + 1 * (action6 == 6) + 1 * (action7 == 6)
                    cardinal_up += 1 * (action6 == 5) + 1 * (action7 == 5) + 1 * (action6 == 2) + 1 * (action7 == 2) + 1 * (action6 == 8) + 1 * (action7 == 8)
                    cardinal_fire += 1 * (action6 == 9) + 1 * (action7 == 9)
                    cardinal_else = 1 * (action6 == 4) + 1 * (action6 == 10) + 1 * (action6 == 11) + 1 * (action6 == 12) + 1 * (action6 == 13) + 1 * (action6 == 14) \
                                  + 1 * (action7 == 9) + 1 * (action7 == 10) + 1 * (action7 == 11) + 1 * (action7 == 12) + 1 * (action7 == 13) + 1 * (action7 == 14)

            if args.num_ensemble > 8:
                actions.append(action8)
                actions.append(action9)
                cardinal_left += 1 * (action8 == 0) + 1 * (action8 == 1) + 1 * (action8 == 2) + 1 * (action9 == 0) + 1 * (action9 == 1) + 1 * (action9 == 2)
                cardinal_right += 1 * (action8 == 6) + 1 * (action8 == 7) + 1 * (action8 == 8) + 1 * (action9 == 6) + 1 * (action9 == 7) + 1 * (action9 == 8)
                if (args.env_name == "maze" or args.env_name == "miner"):
                    cardinal_down += 1 * (action8 == 3) + 1 * (action9 == 3)
                    cardinal_up += 1 * (action8 == 5) + 1 * (action9 == 5)
                else:
                    cardinal_down += 1 * (action8 == 3) + 1 * (action9 == 3) + 1 * (action8 == 0) + 1 * (action9 == 0) + 1 * (action8 == 6) + 1 * (action9 == 6)
                    cardinal_up += 1 * (action8 == 5) + 1 * (action9 == 5) + 1 * (action8 == 2) + 1 * (action9 == 2) + 1 * (action8 == 8) + 1 * (action9 == 8)
                    cardinal_fire += 1 * (action8 == 9) + 1 * (action9 == 9)
                    cardinal_else = 1 * (action8 == 4) + 1 * (action8 == 10) + 1 * (action8 == 11) + 1 * (action8 == 12) + 1 * (action8 == 13) + 1 * (action8 == 14) \
                                  + 1 * (action9 == 9) + 1 * (action9 == 10) + 1 * (action9 == 11) + 1 * (action9 == 12) + 1 * (action9 == 13) + 1 * (action9 == 14)

            if (args.env_name == "maze" or args.env_name == "miner"):
                directions = torch.cat((cardinal_up, cardinal_right, cardinal_down, cardinal_left), dim=1)
            else:
                directions = torch.cat((cardinal_up, cardinal_right, cardinal_down, cardinal_left, cardinal_fire, cardinal_else), dim=1)
            # cardinal_value = torch.max(directions, dim=1)[0]
            # cardinal_index = torch.max(directions, dim=1)[1].unsqueeze(1)

            action_cardinal_left =  1 * ( actions[args.num_agent] == 0) + 1 * ( actions[args.num_agent] == 1) + 1 * ( actions[args.num_agent] == 2)
            action_cardinal_right =  1 * ( actions[args.num_agent] == 6) + 1 * ( actions[args.num_agent] == 7) + 1 * ( actions[args.num_agent] == 8)
            if (args.env_name == "maze" or args.env_name == "miner"):
                action_cardinal_down = 1 * (actions[args.num_agent] == 3)
                action_cardinal_up = 1 * (actions[args.num_agent] == 5)
                action_directions = torch.cat((action_cardinal_up, action_cardinal_right, action_cardinal_down, action_cardinal_left), dim=1)
            else:
                action_cardinal_down = 1 * (actions[args.num_agent] == 3) + 1 * (actions[args.num_agent] == 0) + 1 * (actions[args.num_agent] == 6)
                action_cardinal_up = 1 * (actions[args.num_agent] == 5) + 1 * (actions[args.num_agent] == 2) + 1 * (actions[args.num_agent] == 8)
                action_cardinal_fire = 1 * (actions[args.num_agent] == 9)
                action_cardinal_else = 1 * (actions[args.num_agent] == 4) + 1 * (actions[args.num_agent] == 10) + 1 * (actions[args.num_agent] == 11) + 1 * (actions[args.num_agent] == 12) + 1 * (actions[args.num_agent] == 13) + 1 * (actions[args.num_agent] == 14)
                action_directions = torch.cat((action_cardinal_up, action_cardinal_right, action_cardinal_down, action_cardinal_left, action_cardinal_fire, action_cardinal_else), dim=1)

            action_cardinal_index = torch.max(action_directions, dim=1)[1]


            is_majority = (directions[torch.arange(32),action_cardinal_index] >= args.num_agree).unsqueeze(1)
            # is_majority = ((cardinal_index == action_cardinal_index)*(cardinal_value >= args.num_agree).unsqueeze(1))
            # is_majority = (cardinal_value >= args.num_agree).unsqueeze(1)
            # lookup = torch.tensor([5, 7, 3, 1, 9], device=device)
            # action_NN = lookup[cardinal_index].unsqueeze(1)
            action_NN = actions[args.num_agent]

            maxEnt_steps = maxEnt_steps - 1

            # is_majority = (action0 == action1) * (action0 == action2) * (action0 == action3)
            # is_majority = (action0 == action1) * (action0 == action2) * (action0 == action3)
            # step_count = (step_count+1)*is_majority
            # is_maxEnt = (step_count<10)
            # is_pure_action = is_novel*is_majority
            maxEnt_steps_sample = (~is_majority)*(maxEnt_steps<=0)
            maxEnt_steps = (m.sample() + 1).to(device)*maxEnt_steps_sample + maxEnt_steps*(~maxEnt_steps_sample)
            # maxEnt_steps = (3*torch.ones(num_processes,1, device=device))*is_pure_action + maxEnt_steps*(~is_pure_action)

            is_action = is_majority*(maxEnt_steps<=0)
            action = action_NN*is_action + action_maxEnt*(~is_action)
            if args.rand_act:
                action = action_NN * is_action + rand_action.sample().to(device) * (~is_action)
            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action.squeeze().cpu().numpy())
            # if max(reward) < 10 and max(reward) >0:
            #     print(reward)

            for i, info in enumerate(infos):
                seeds[i] = info["level_seed"]
                # episode_len_buffer[i] += 1
                # if done[i] == True:
                #     episode_rewards.append(reward[i])
                #     episode_len.append(episode_len_buffer[i])
                #     episode_len_buffer[i] = 0

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            # rollouts.insert(obs, recurrent_hidden_states, action,
            #                 action_log_prob, value, torch.from_numpy(reward).unsqueeze(1), masks, bad_masks, attn_masks,
            #                 attn_masks1, attn_masks2, attn_masks3, seeds, infos, obs)
            # rollouts1.insert(obs, recurrent_hidden_states1, action,
            #                 action_log_prob, value, torch.from_numpy(reward).unsqueeze(1), masks, bad_masks, attn_masks,
            #                 attn_masks1, attn_masks2, attn_masks3, seeds, infos, obs)
            # rollouts2.insert(obs, recurrent_hidden_states2, action,
            #                 action_log_prob, value, torch.from_numpy(reward).unsqueeze(1), masks, bad_masks, attn_masks,
            #                 attn_masks1, attn_masks2, attn_masks3, seeds, infos, obs)
            # rollouts3.insert(obs, recurrent_hidden_states3, action,
            #                 action_log_prob, value, torch.from_numpy(reward).unsqueeze(1), masks, bad_masks, attn_masks,
            #                 attn_masks1, attn_masks2, attn_masks3, seeds, infos, obs)
            rollouts_maxEnt.insert(obs, recurrent_hidden_states_maxEnt, action,
                            action_log_prob, value, torch.from_numpy(reward).unsqueeze(1), masks, bad_masks, attn_masks,
                            attn_masks1, attn_masks2, attn_masks3, seeds, infos, obs)


        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts_maxEnt.obs[-1]).detach()

        actor_critic.train()
        rollouts_maxEnt.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        # value_loss, action_loss, dist_entropy, _ = agent.update(rollouts)

        rollouts_maxEnt.after_update()

        rew_batch, done_batch = rollouts_maxEnt.fetch_log_data()
        logger.feed_train(rew_batch, done_batch[1:])

    #     # save for every interval-th episode or for the last epoch
    #     if (j % args.save_interval == 0 or j == args.continue_from_epoch + num_updates - 1):
    #         torch.save({'state_dict': actor_critic.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(),
    #                     'step': j}, os.path.join(logdir, args.env_name + "-epoch-{}.pt".format(j)))
    #                     # 'buffer_obs': rollouts.obs,
    #                     # 'buffer_recurrent_hidden_states': rollouts.recurrent_hidden_states,
    #                     # 'buffer_rewards': rollouts.rewards,
    #                     # 'buffer_seeds': rollouts.seeds,
    #                     # 'buffer_value_preds': rollouts.value_preds,
    #                     # 'buffer_returns': rollouts.returns,
    #                     # 'buffer_action_log_probs': rollouts.action_log_probs,
    #                     # 'buffer_actions': rollouts.actions,
    #                     # 'buffer_masks': rollouts.masks,
    #                     # 'buffer_bad_masks': rollouts.bad_masks,
    #                     # 'buffer_info_batch': rollouts.info_batch,
    #                     # 'buffer_num_steps': rollouts.num_steps,
    #                     # 'buffer_step': rollouts.step,
    #                     # 'buffer_num_processes': rollouts.num_processes}, os.path.join(logdir, args.env_name + "-epoch-{}.pt".format(j)))
    #
    #     # print some stats
    #     if j % args.log_interval == 0:
    #         total_num_steps = (j + 1) * args.num_processes * args.num_steps
    #         end = time.time()
    #
    #         train_statistics = logger.get_train_val_statistics()
    #         print(
    #             "Updates {}, num timesteps {}, FPS {}, num training episodes {} \n Last 128 training episodes: mean/median reward {:.1f}/{:.1f}, "
    #             "min/max reward {:.1f}/{:.1f}, dist_entropy {} , value_loss {}, action_loss {}, unique seeds {}\n"
    #             .format(j, total_num_steps,
    #                     int(total_num_steps / (end - start)),
    #                     logger.num_episodes, train_statistics['Rewards_mean_episodes'],
    #                     train_statistics['Rewards_median_episodes'], train_statistics['Rewards_min_episodes'], train_statistics['Rewards_max_episodes'], dist_entropy, value_loss,
    #                     action_loss, np.unique(rollouts.seeds.squeeze().numpy()).size))
    #     # evaluate agent on evaluation tasks
    #     if ((args.eval_interval is not None and j % args.eval_interval == 0) or j == args.continue_from_epoch):
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
            eval_dic_rew[eval_disp_name], eval_dic_done[eval_disp_name] = evaluate_procgen_ensemble(actor_critic, actor_critic1, actor_critic2, actor_critic3, actor_critic4, actor_critic5, actor_critic6, actor_critic7, actor_critic8, actor_critic9,
                                                                                                    actor_critic_maxEnt, eval_envs_dic, eval_disp_name,
                                                                                                    args.num_processes, device, args.num_steps, logger, deterministic=True, num_detEnt=args.num_detEnt, rand_act=args.rand_act,
                                                                                                    num_ensemble=args.num_ensemble, num_agree=args.num_agree, maze_miner=maze_miner, num_agent=args.num_agent)


                # if ((args.eval_nondet_interval is not None and j % args.eval_nondet_interval == 0) or j == args.continue_from_epoch):
                #     eval_test_nondet_rew, eval_test_nondet_done = evaluate_procgen(actor_critic, eval_envs_dic, 'test_eval',
                #                                       args.num_processes, device, args.num_steps, deterministic=False)

        # eval_test_nondet_rew, eval_test_nondet_done = evaluate_procgen_ensemble(actor_critic, actor_critic1, actor_critic2, actor_critic3, actor_critic_maxEnt, eval_envs_dic, 'test_eval',
        #                                                                args.num_processes, device, args.num_steps, logger,
        #                                                                attention_features=False, det_masks=False,
        #                                                                deterministic=False)
        eval_test_nondet_rew, eval_test_nondet_done = evaluate_procgen_ensemble(actor_critic, actor_critic1, actor_critic2, actor_critic3, actor_critic4, actor_critic5, actor_critic6, actor_critic7, actor_critic8, actor_critic9,
                                                                                actor_critic_maxEnt, eval_envs_dic_nondet, 'test_eval_nondet',
                                                                                args.num_processes, device, args.num_steps, logger, attention_features=False, det_masks=False, deterministic=False, num_detEnt=args.num_detEnt, rand_act=args.rand_act,
                                                                                num_ensemble=args.num_ensemble, num_agree=args.num_agree, maze_miner=maze_miner, num_agent=args.num_agent)

        logger.feed_eval(eval_dic_rew['train_eval'], eval_dic_done['train_eval'], eval_dic_rew['test_eval'], eval_dic_done['test_eval'], seeds_train, seeds_test,
                         eval_dic_rew['train_eval'], eval_dic_rew['test_eval'], eval_test_nondet_rew, eval_test_nondet_done)
        if len(logger.episode_reward_buffer)>0 and len(logger.episode_reward_buffer_test_nondet)>0:
            episode_statistics = logger.get_episode_statistics()
            print("train and test eval")
            print(episode_statistics)

        #         # reinitialize the last layers of networks + GRU unit
        #         if args.reinitialization and (j % 500 == 0):
        #             print('initialize weights j = {}'.format(j))
        #             init_2(actor_critic.base.critic_linear)
        #             init_(actor_critic.base.main[5])
        #             for name, param in actor_critic.base.gru.named_parameters():
        #                 if 'bias' in name:
        #                     nn.init.constant_(param, 0)
        #                 elif 'weight' in name:
        #                     nn.init.orthogonal_(param)
        #
        #
        #         # summary_writer.add_scalars('eval_mean_rew', {f'{eval_disp_name}': np.mean(eval_dic_rew[eval_disp_name])},
        #         #                               (j+1) * args.num_processes * args.num_steps)
        #         # summary_writer.add_scalars('eval_min_rew', {f'{eval_disp_name}': np.min(eval_dic_rew[eval_disp_name])},
        #         #                               (j+1) * args.num_processes * args.num_steps)
        #         # summary_writer.add_scalars('eval_max_rew',{f'{eval_disp_name}': np.max(eval_dic_rew[eval_disp_name])},
        #         #                               (j+1) * args.num_processes * args.num_steps)
        #         # summary_writer.add_scalars('eval_mean_len', {f'{eval_disp_name}': np.mean(eval_dic_len[eval_disp_name])},
        #         #                               (j+1) * args.num_processes * args.num_steps)
        #         # summary_writer.add_scalars('eval_min_len', {f'{eval_disp_name}': np.min(eval_dic_len[eval_disp_name])},
        #         #                               (j+1) * args.num_processes * args.num_steps)
        #         # summary_writer.add_scalars('eval_max_len',{f'{eval_disp_name}': np.max(eval_dic_len[eval_disp_name])},
        #         #                               (j+1) * args.num_processes * args.num_steps)
        #         #
        #         # summary_writer.add_scalars('eval_mean_rew', {'train': np.mean(episode_rewards)},
        #         #                           (j + 1) * args.num_processes * args.num_steps)
        #         # summary_writer.add_scalars('eval_min_rew', {'train': np.min(episode_rewards)},
        #         #                           (j + 1) * args.num_processes * args.num_steps)
        #         # summary_writer.add_scalars('eval_max_rew', {'train': np.max(episode_rewards)},
        #         #                           (j + 1) * args.num_processes * args.num_steps)
        #         # summary_writer.add_scalars('eval_mean_len', {'train': np.mean(episode_len)},
        #         #                           (j + 1) * args.num_processes * args.num_steps)
        #         # summary_writer.add_scalars('eval_min_len', {'train': np.min(episode_len)},
        #         #                           (j + 1) * args.num_processes * args.num_steps)
        #         # summary_writer.add_scalars('eval_max_len', {'train': np.max(episode_len)},
        #         #                           (j + 1) * args.num_processes * args.num_steps)
        #
            if (len(episode_statistics)>0):
                for key, value in episode_statistics.items():
                    if isinstance(value, dict):
                        # summary_writer.add_scalars(key, value,(j + 1) * args.num_processes * args.num_steps)
                        for key_v, value_v in value.items():
                             wandb.log({key + "/" + key_v: value_v}, step=(j + 1) * args.num_processes * args.num_steps)
    #
    #             else:
    #                 summary_writer.add_scalar(key, value, (j + 1) * args.num_processes * args.num_steps)
    #             # wandb.log({f'eval/{eval_disp_name}': np.mean(eval_r[eval_disp_name])}, step=total_num_steps)
    #
    #         summary ={'Loss/pi': action_loss,
    #                   'Loss/v': value_loss,
    #                   'Loss/entropy': dist_entropy}
    #         for key, value in summary.items():
    #             summary_writer.add_scalar(key, value, (j + 1) * args.num_processes * args.num_steps)
    #             wandb.log({key: value}, step=(j + 1) * args.num_processes * args.num_steps)
    #
    #
    # # training done. Save and clean up
    # save_obj(log_dict, os.path.join(logdir, 'log_dict.pkl'))
    # envs.close()
    for eval_disp_name in EVAL_ENVS:
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
