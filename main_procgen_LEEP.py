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
from a2c_ppo_acktr.model import Policy, MLPAttnBase, MLPHardAttnBase, MLPHardAttnReinforceBase, ImpalaModel, ImpalaModel_finetune
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate_procgen, evaluate_procgen_maxEnt, evaluate_procgen_LEEP
from a2c_ppo_acktr.utils import save_obj, load_obj
from a2c_ppo_acktr.procgen_wrappers import *
from a2c_ppo_acktr.logger import Logger, maxEnt_Logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from a2c_ppo_acktr.utils import init
import wandb
from evaluation import maxEnt_oracle

EVAL_ENVS = ['train_eval','test_eval']

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                       constant_(x, 0), nn.init.calculate_gain('relu'))
init_2 = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=1)
init_dist = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    gain=0.01)

def main():
    args = get_args()
    import random; random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logdir_ = 'LEEP_' + args.env_name + '_seed_' + str(args.seed) + '_num_env_' + str(args.num_level) + '_entro_' + str(args.entropy_coef) + '_gama_' + str(args.gamma) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    if args.normalize_rew:
        logdir_ = logdir_ + '_normalize_rew'
    if not args.recurrent_policy:
        logdir_ = logdir_ + '_noRNN'
    if args.mask_all:
        logdir_ = logdir_ + '_mask_all'
    if args.mask_size > 0:
        logdir_ = logdir_ + '_mask_' + str(args.mask_size)

    logdir = os.path.join(os.path.expanduser(args.log_dir), logdir_)
    utils.cleanup_log_dir(logdir)

    wandb.init(project=args.env_name + "_PPO_LEEP", entity="ev_zisselman", config=args, name=logdir_, id=logdir_)

    # Ugly but simple logging
    log_dict = {
        'num_steps': args.num_steps,
        'seed': args.seed,
        'recurrent': args.recurrent_policy,
        'train_env': args.env_name,
        'test_env': args.val_env_name,
        'cmd': ' '.join(sys.argv)
    }
    for eval_disp_name in EVAL_ENVS:
        log_dict[eval_disp_name] = []


    argslog = pd.DataFrame(columns=['args', 'value'])
    for key in vars(args):
        log = [key] + [vars(args)[key]]
        argslog.loc[len(argslog)] = log

    print("logdir: " + logdir)
    for key in vars(args):
        print(key, ':', vars(args)[key])

    with open(logdir + '/args.csv', 'w') as f:
        argslog.to_csv(f, index=False)


    # Tensorboard logging
    summary_writer = SummaryWriter(log_dir=logdir)

    summary_writer.add_hparams(vars(args), {})

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.gpu_device) if args.cuda else "cpu")

    print('making envs...')
    #compute maximun reward per seed
    max_reward_seeds = {
        'train_eval': [],
        'test_eval': []
    }

    test_start_level = 100000
    start_train_test = {
        'train_eval': args.start_level,
        'test_eval': test_start_level
    }

    # for eval_disp_name in EVAL_ENVS:
    #     for i in range(args.num_test_level):
    #         envs = make_ProcgenEnvs(num_envs=1,
    #                                 env_name=args.env_name,
    #                                 start_level=start_train_test[eval_disp_name] + i,
    #                                 num_levels=1,
    #                                 distribution_mode=args.distribution_mode,
    #                                 use_generated_assets=args.use_generated_assets,
    #                                 use_backgrounds=args.use_backgrounds,
    #                                 restrict_themes=args.restrict_themes,
    #                                 use_monochrome_assets=args.use_monochrome_assets,
    #                                 rand_seed=args.seed,
    #                                 center_agent=args.center_agent,
    #                                 mask_size=args.mask_size,
    #                                 normalize_rew=args.normalize_rew,
    #                                 mask_all=args.mask_all)
    #
    #         obs = envs.reset()
    #         obs_sum = obs
    #         # plot mazes
    #         # plt.imshow(obs[0].transpose(0, 2).cpu().numpy())
    #         # plt.savefig("test.png")
    #         # plt.show()
    #
    #         action = torch.full((1, 1), 5)
    #         done = torch.full((1, 1), 0)
    #         reward = 0
    #
    #         while not done[0]:
    #             with torch.no_grad():
    #
    #                 action = maxEnt_oracle(obs, action)
    #
    #                 obs, _, done, infos = envs.step(action[0].cpu().numpy())
    #                 # print(action[0])
    #                 # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    #                 # plt.show()
    #
    #                 next_obs_sum = obs_sum + obs
    #                 num_zero_obs_sum = (obs_sum[0] == 0).sum()
    #                 num_zero_next_obs_sum = (next_obs_sum[0] == 0).sum()
    #                 if num_zero_next_obs_sum < num_zero_obs_sum:
    #                     reward += 1
    #
    #                 obs_sum = next_obs_sum
    #
    #         max_reward_seeds[eval_disp_name].append(reward)

    # Train envs
    num_envs = int(args.num_level/args.num_c)
    envs_dic = []
    for i in range(args.num_c):
        envs_dic.append(make_ProcgenEnvs(num_envs=int(args.num_processes/args.num_c),
                                         env_name=args.env_name,
                                         start_level=args.start_level + i*num_envs,
                                         num_levels=(i+1)*num_envs,
                                         distribution_mode=args.distribution_mode,
                                         use_generated_assets=args.use_generated_assets,
                                         use_backgrounds=args.use_backgrounds,
                                         restrict_themes=args.restrict_themes,
                                         use_monochrome_assets=args.use_monochrome_assets,
                                         rand_seed=args.seed,
                                         center_agent=args.center_agent,
                                         mask_size=args.mask_size,
                                         normalize_rew=args.normalize_rew,
                                         mask_all=args.mask_all,
                                         device=device))
    # Test envs
    eval_envs_dic = {}
    eval_envs_dic['train_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                   env_name=args.env_name,
                                                   start_level=args.start_level,
                                                   num_levels=args.num_test_level,
                                                   distribution_mode=args.distribution_mode,
                                                   use_generated_assets=args.use_generated_assets,
                                                   use_backgrounds=args.use_backgrounds,
                                                   restrict_themes=args.restrict_themes,
                                                   use_monochrome_assets=args.use_monochrome_assets,
                                                   rand_seed=args.seed,
                                                   center_agent=args.center_agent,
                                                   mask_size=args.mask_size,
                                                   normalize_rew= args.normalize_rew,
                                                   mask_all=args.mask_all,
                                                   device=device)


    eval_envs_dic['test_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                  env_name=args.env_name,
                                                  start_level=test_start_level,
                                                  num_levels=args.num_test_level,
                                                  distribution_mode=args.distribution_mode,
                                                  use_generated_assets=args.use_generated_assets,
                                                  use_backgrounds=args.use_backgrounds,
                                                  restrict_themes=args.restrict_themes,
                                                  use_monochrome_assets=args.use_monochrome_assets,
                                                  rand_seed=args.seed,
                                                  center_agent=args.center_agent,
                                                  mask_size=args.mask_size,
                                                  normalize_rew=args.normalize_rew,
                                                  mask_all=args.mask_all,
                                                  device=device)
    print('done')

    actor_critic_0 = Policy(
        envs_dic[0].observation_space.shape,
        envs_dic[0].action_space,
        base=ImpalaModel_finetune,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent,'hidden_size': args.recurrent_hidden_size})
        # base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic_0.to(device)

    actor_critic_1 = Policy(
        envs_dic[1].observation_space.shape,
        envs_dic[1].action_space,
        base=ImpalaModel_finetune,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent,'hidden_size': args.recurrent_hidden_size})
        # base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic_1.to(device)

    actor_critic_2 = Policy(
        envs_dic[2].observation_space.shape,
        envs_dic[2].action_space,
        base=ImpalaModel_finetune,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent,'hidden_size': args.recurrent_hidden_size})
        # base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic_2.to(device)

    actor_critic_3 = Policy(
        envs_dic[3].observation_space.shape,
        envs_dic[3].action_space,
        base=ImpalaModel_finetune,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent,'hidden_size': args.recurrent_hidden_size})
        # base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic_3.to(device)

    if args.algo != 'ppo':
        raise print("only PPO is supported")

    # training agent 0
    agent_0 = algo.PPO_LEEP(
        actor_critic_0,
        actor_critic_1,
        actor_critic_2,
        actor_critic_3,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.KL_coef,
        lr=args.lr,
        eps=args.eps,
        num_tasks=int(args.num_processes/args.num_c),
        attention_policy=False,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # training agent 1
    agent_1 = algo.PPO_LEEP(
        actor_critic_1,
        actor_critic_2,
        actor_critic_3,
        actor_critic_0,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.KL_coef,
        lr=args.lr,
        eps=args.eps,
        num_tasks=int(args.num_processes/args.num_c),
        attention_policy=False,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # training agent 2
    agent_2 = algo.PPO_LEEP(
        actor_critic_2,
        actor_critic_3,
        actor_critic_0,
        actor_critic_1,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.KL_coef,
        lr=args.lr,
        eps=args.eps,
        num_tasks=int(args.num_processes/args.num_c),
        attention_policy=False,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # training agent 3
    agent_3 = algo.PPO_LEEP(
        actor_critic_3,
        actor_critic_0,
        actor_critic_1,
        actor_critic_2,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.KL_coef,
        lr=args.lr,
        eps=args.eps,
        num_tasks=int(args.num_processes/args.num_c),
        attention_policy=False,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # rollout storage for agents
    rollouts_0 = RolloutStorage(args.num_steps, int(args.num_processes/args.num_c),
                              envs_dic[0].observation_space.shape, envs_dic[0].observation_space.shape, envs_dic[0].action_space,
                              actor_critic_0.recurrent_hidden_state_size, args.mask_size, device=device)
    rollouts_1 = RolloutStorage(args.num_steps, int(args.num_processes/args.num_c),
                              envs_dic[1].observation_space.shape, envs_dic[1].observation_space.shape, envs_dic[1].action_space,
                              actor_critic_1.recurrent_hidden_state_size, args.mask_size, device=device)
    rollouts_2 = RolloutStorage(args.num_steps, int(args.num_processes/args.num_c),
                              envs_dic[2].observation_space.shape, envs_dic[2].observation_space.shape, envs_dic[2].action_space,
                              actor_critic_2.recurrent_hidden_state_size, args.mask_size, device=device)
    rollouts_3 = RolloutStorage(args.num_steps, int(args.num_processes/args.num_c),
                              envs_dic[3].observation_space.shape, envs_dic[3].observation_space.shape, envs_dic[3].action_space,
                              actor_critic_3.recurrent_hidden_state_size, args.mask_size, device=device)


    # Load previous model
    if (args.continue_from_epoch > 0) and args.save_dir != "":
        save_path = args.save_dir
        actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(args.continue_from_epoch)), map_location=device)
        actor_critic_0.load_state_dict(actor_critic_weighs['state_dict_0'])
        agent_0.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict_0'])
        actor_critic_1.load_state_dict(actor_critic_weighs['state_dict_1'])
        agent_1.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict_1'])
        actor_critic_2.load_state_dict(actor_critic_weighs['state_dict_2'])
        agent_2.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict_2'])
        actor_critic_3.load_state_dict(actor_critic_weighs['state_dict_3'])
        agent_3.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict_3'])

    # Load previous model
    if (args.saved_epoch > 0) and args.save_dir != "":
        save_path = args.save_dir
        actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch)), map_location=device)
        actor_critic_1.load_state_dict(actor_critic_weighs['state_dict_1'], strict=False)

    # logger = maxEnt_Logger(args.num_processes, max_reward_seeds, start_train_test, envs_dic[0].observation_space.shape,
    # actor_critic_0.recurrent_hidden_state_size, device=device)
    logger = Logger(args.num_processes, envs_dic[0].observation_space.shape, envs_dic[0].observation_space.shape, actor_critic_0.recurrent_hidden_state_size, device=device)

    obs_0 = envs_dic[0].reset()
    obs_1 = envs_dic[1].reset()
    obs_2 = envs_dic[2].reset()
    obs_3 = envs_dic[3].reset()
    # rollouts.obs[0].copy_(torch.FloatTensor(obs))
    rollouts_0.obs[0].copy_(obs_0)
    rollouts_1.obs[0].copy_(obs_1)
    rollouts_2.obs[0].copy_(obs_2)
    rollouts_3.obs[0].copy_(obs_3)
    # rollouts.to(device)

    obs_train = eval_envs_dic['train_eval'].reset()
    logger.obs['train_eval'].copy_(obs_train)
    logger.obs_sum['train_eval'].copy_(obs_train)

    obs_test = eval_envs_dic['test_eval'].reset()
    logger.obs['test_eval'].copy_(obs_test)
    logger.obs_sum['test_eval'].copy_(obs_test)

    fig = plt.figure(figsize=(20, 20))
    columns = 1
    rows = 1
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(rollouts_0.obs[0][i].transpose(0,2))
        plt.savefig(logdir + '/fig.png')

    seeds = torch.zeros(int(args.num_processes/args.num_c), 1)
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
    # beta = 1


    # #freeze layers
    # if args.freeze1:
    #     for name, param in actor_critic.base.main[0].named_parameters():
    #         param.requires_grad = False
    # if args.freeze2:
    #     for name, param in actor_critic.base.main[0].named_parameters():
    #         param.requires_grad = False
    #     for name, param in actor_critic.base.main[1].named_parameters():
    #         param.requires_grad = False
    # if args.freeze2_gru:
    #     for name, param in actor_critic.base.main[0].named_parameters():
    #         param.requires_grad = False
    #     for name, param in actor_critic.base.main[1].named_parameters():
    #         param.requires_grad = False
    #     for name, param in actor_critic.base.gru.named_parameters():
    #         param.requires_grad = False
    #     init_(actor_critic.base.main[2].conv)
    #     init_(actor_critic.base.main[2].res1.conv1)
    #     init_(actor_critic.base.main[2].res1.conv2)
    #     init_(actor_critic.base.main[2].res2.conv1)
    #     init_(actor_critic.base.main[2].res1.conv1)
    #     init_(actor_critic.base.main[5])
    #     init_dist(actor_critic.dist.linear)
    #     init_2(actor_critic.base.critic_linear)
    #
    # if args.freeze_all:
    #     for name, param in actor_critic.base.main.named_parameters():
    #         param.requires_grad = False
    # if args.freeze_all_gru:
    #     for name, param in actor_critic.base.main.named_parameters():
    #         param.requires_grad = False
    #     for name, param in actor_critic.base.gru.named_parameters():
    #         param.requires_grad = False
    #     init_dist(actor_critic.dist.linear)
    #     init_2(actor_critic.base.critic_linear)


    for j in range(args.continue_from_epoch, args.continue_from_epoch+num_updates):

        # # plot mazes
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
        actor_critic_0.eval()
        actor_critic_1.eval()
        actor_critic_2.eval()
        actor_critic_3.eval()
        # episode_rewards = []
        # episode_len = []
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value_0, action_0, action_log_prob_0, _, recurrent_hidden_states_0, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic_0.act(
                    rollouts_0.obs[step].to(device), rollouts_0.recurrent_hidden_states[step].to(device),
                    rollouts_0.masks[step].to(device), rollouts_0.attn_masks[step].to(device), rollouts_0.attn_masks1[step].to(device), rollouts_0.attn_masks2[step].to(device),
                    rollouts_0.attn_masks3[step].to(device))

                value_1, action_1, action_log_prob_1, _, recurrent_hidden_states_1, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic_1.act(
                    rollouts_1.obs[step].to(device), rollouts_1.recurrent_hidden_states[step].to(device),
                    rollouts_1.masks[step].to(device), rollouts_1.attn_masks[step].to(device), rollouts_1.attn_masks1[step].to(device), rollouts_1.attn_masks2[step].to(device),
                    rollouts_1.attn_masks3[step].to(device))


                value_2, action_2, action_log_prob_2, _, recurrent_hidden_states_2, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic_2.act(
                    rollouts_2.obs[step].to(device), rollouts_2.recurrent_hidden_states[step].to(device),
                    rollouts_2.masks[step].to(device), rollouts_2.attn_masks[step].to(device), rollouts_2.attn_masks1[step].to(device), rollouts_2.attn_masks2[step].to(device),
                    rollouts_2.attn_masks3[step].to(device))

                value_3, action_3, action_log_prob_3, _, recurrent_hidden_states_3, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic_3.act(
                    rollouts_3.obs[step].to(device), rollouts_3.recurrent_hidden_states[step].to(device),
                    rollouts_3.masks[step].to(device), rollouts_3.attn_masks[step].to(device), rollouts_3.attn_masks1[step].to(device), rollouts_3.attn_masks2[step].to(device),
                    rollouts_3.attn_masks3[step].to(device))

            # Observe reward and next obs
            obs_0, reward_0, done_0, infos_0 = envs_dic[0].step(action_0.squeeze().cpu().numpy())
            obs_1, reward_1, done_1, infos_1 = envs_dic[1].step(action_1.squeeze().cpu().numpy())
            obs_2, reward_2, done_2, infos_2 = envs_dic[2].step(action_2.squeeze().cpu().numpy())
            obs_3, reward_3, done_3, infos_3 = envs_dic[3].step(action_3.squeeze().cpu().numpy())

            for i, info in enumerate(infos_0):
                seeds[i] = info["level_seed"]
                # episode_len_buffer[i] += 1
                # if done[i] == True:
                #     episode_rewards.append(reward[i])
                #     episode_len.append(episode_len_buffer[i])
                #     episode_len_buffer[i] = 0

            # for i in range(len(done)):
            #     if done[i] == 1:
            #         # rollouts.obs_sum[i] = torch.zeros_like(rollouts.obs_sum[i])
            #         rollouts.obs_sum[i].copy_(obs[i].cpu())
            #
            # next_obs_sum =  rollouts.obs_sum + obs.cpu()
            # int_reward = np.zeros_like(reward)
            # for i in range(len(int_reward)):
            #     if done[i] == 0:
            #         num_zero_obs_sum = (rollouts.obs_sum[i][0] == 0).sum()
            #         num_zero_next_obs_sum = (next_obs_sum[i][0] == 0).sum()
            #         if num_zero_next_obs_sum < num_zero_obs_sum:
            #             int_reward[i] = 1

            # reward = (1-beta)*reward + beta*int_reward
            # If done then clean the history of observations.
            masks_0 = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done_0])
            bad_masks_0 = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos_0])
            masks_1 = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done_1])
            bad_masks_1 = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos_1])
            masks_2 = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done_2])
            bad_masks_2 = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos_2])
            masks_3 = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done_3])
            bad_masks_3 = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos_3])

            rollouts_0.insert(obs_0, recurrent_hidden_states_0, action_0,
                            action_log_prob_0, value_0, torch.from_numpy(reward_0).unsqueeze(1), masks_0, bad_masks_0, attn_masks, attn_masks1, attn_masks2, attn_masks3, seeds, infos_0, obs_0)
            rollouts_1.insert(obs_1, recurrent_hidden_states_1, action_1,
                            action_log_prob_1, value_1, torch.from_numpy(reward_1).unsqueeze(1), masks_1, bad_masks_1, attn_masks, attn_masks1, attn_masks2, attn_masks3, seeds, infos_1, obs_0)
            rollouts_2.insert(obs_2, recurrent_hidden_states_2, action_2,
                            action_log_prob_2, value_2, torch.from_numpy(reward_2).unsqueeze(1), masks_2, bad_masks_2, attn_masks, attn_masks1, attn_masks2, attn_masks3, seeds, infos_2, obs_0)
            rollouts_3.insert(obs_3, recurrent_hidden_states_3, action_3,
                            action_log_prob_3, value_3, torch.from_numpy(reward_3).unsqueeze(1), masks_3, bad_masks_3, attn_masks, attn_masks1, attn_masks2, attn_masks3, seeds, infos_3, obs_0)

        # beta = beta*args.beta_decay

        with torch.no_grad():
            next_value_0 = actor_critic_0.get_value(
                rollouts_0.obs[-1].to(device), rollouts_0.recurrent_hidden_states[-1].to(device),
                rollouts_0.masks[-1].to(device), rollouts_0.attn_masks[-1].to(device), rollouts_0.attn_masks1[-1].to(device),
                    rollouts_0.attn_masks2[-1].to(device), rollouts_0.attn_masks3[-1].to(device)).detach()

            next_value_1 = actor_critic_1.get_value(
                rollouts_1.obs[-1].to(device), rollouts_1.recurrent_hidden_states[-1].to(device),
                rollouts_1.masks[-1].to(device), rollouts_1.attn_masks[-1].to(device), rollouts_1.attn_masks1[-1].to(device),
                    rollouts_1.attn_masks2[-1].to(device), rollouts_1.attn_masks3[-1].to(device)).detach()

            next_value_2 = actor_critic_2.get_value(
                rollouts_2.obs[-1].to(device), rollouts_2.recurrent_hidden_states[-1].to(device),
                rollouts_2.masks[-1].to(device), rollouts_2.attn_masks[-1].to(device), rollouts_2.attn_masks1[-1].to(device),
                    rollouts_2.attn_masks2[-1].to(device), rollouts_2.attn_masks3[-1].to(device)).detach()

            next_value_3 = actor_critic_3.get_value(
                rollouts_3.obs[-1].to(device), rollouts_3.recurrent_hidden_states[-1].to(device),
                rollouts_3.masks[-1].to(device), rollouts_3.attn_masks[-1].to(device), rollouts_3.attn_masks1[-1].to(device),
                    rollouts_3.attn_masks2[-1].to(device), rollouts_3.attn_masks3[-1].to(device)).detach()

        actor_critic_0.train()
        rollouts_0.compute_returns(next_value_0, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss_0, action_loss_0, dist_entropy_0, dist_KL_epoch_0 = agent_0.update(rollouts_0)

        rollouts_0.after_update()

        rew_batch_0, done_batch_0 = rollouts_0.fetch_log_data()
        logger.feed_train(rew_batch_0, done_batch_0[1:])

        actor_critic_1.train()
        rollouts_1.compute_returns(next_value_1, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        _, _, _, _ = agent_1.update(rollouts_1)

        rollouts_1.after_update()

        actor_critic_2.train()
        rollouts_2.compute_returns(next_value_2, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        _, _, _, _  = agent_2.update(rollouts_2)

        rollouts_2.after_update()

        actor_critic_3.train()
        rollouts_3.compute_returns(next_value_3, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        _, _, _, _  = agent_3.update(rollouts_3)

        rollouts_3.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == args.continue_from_epoch + num_updates - 1):
            torch.save({'state_dict_0': actor_critic_0.state_dict(), 'optimizer_state_dict_0': agent_0.optimizer.state_dict(),
                        'state_dict_1': actor_critic_1.state_dict(), 'optimizer_state_dict_1': agent_1.optimizer.state_dict(),
                        'state_dict_2': actor_critic_2.state_dict(), 'optimizer_state_dict_2': agent_2.optimizer.state_dict(),
                        'state_dict_3': actor_critic_3.state_dict(), 'optimizer_state_dict_3': agent_3.optimizer.state_dict(),
                        'step': j}, os.path.join(logdir, args.env_name + "-epoch-{}.pt".format(j)))
                        # 'buffer_obs': rollouts.obs,
                        # 'buffer_recurrent_hidden_states': rollouts.recurrent_hidden_states,
                        # 'buffer_rewards': rollouts.rewards,
                        # 'buffer_seeds': rollouts.seeds,
                        # 'buffer_value_preds': rollouts.value_preds,
                        # 'buffer_returns': rollouts.returns,
                        # 'buffer_action_log_probs': rollouts.action_log_probs,
                        # 'buffer_actions': rollouts.actions,
                        # 'buffer_masks': rollouts.masks,
                        # 'buffer_bad_masks': rollouts.bad_masks,
                        # 'buffer_info_batch': rollouts.info_batch,
                        # 'buffer_num_steps': rollouts.num_steps,
                        # 'buffer_step': rollouts.step,
                        # 'buffer_num_processes': rollouts.num_processes}, os.path.join(logdir, args.env_name + "-epoch-{}.pt".format(j)))

        # print some stats
        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            train_statistics = logger.get_train_val_statistics()
            print(
                "Updates {}, num timesteps {}, FPS {}, num training episodes {} \n Last 128 training episodes: mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}, dist_entropy {} , value_loss {}, action_loss {}, KL_loss {}, unique seeds {} \n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        logger.num_episodes, train_statistics['Rewards_mean_episodes'],
                        train_statistics['Rewards_median_episodes'], train_statistics['Rewards_min_episodes'], train_statistics['Rewards_max_episodes'], dist_entropy_0, value_loss_0,
                        action_loss_0, dist_KL_epoch_0, np.unique(rollouts_0.seeds.squeeze().numpy()).size))
        # evaluate agent on evaluation tasks
        if ((args.eval_interval is not None and j % args.eval_interval == 0) or j == args.continue_from_epoch):
            actor_critic_0.eval()
            actor_critic_1.eval()
            actor_critic_2.eval()
            actor_critic_3.eval()
            printout = f'Seed {args.seed} Iter {j} '
            eval_dic_rew = {}
            eval_dic_int_rew = {}
            eval_dic_done = {}
            eval_dic_seeds = {}

            for eval_disp_name in EVAL_ENVS:
                eval_dic_rew[eval_disp_name], eval_dic_int_rew[eval_disp_name], eval_dic_done[eval_disp_name], eval_dic_seeds[eval_disp_name]  = evaluate_procgen_LEEP(actor_critic_0, actor_critic_1, actor_critic_2, actor_critic_3,
                                                                                                                                                                       eval_envs_dic, eval_disp_name,args.num_processes, device, args.num_steps, logger)

                # log_dict[eval_disp_name].append([(j+1) * args.num_processes * args.num_steps, eval_dic_rew[eval_disp_name]])
                # printout += eval_disp_name + ' ' + str(np.mean(eval_dic_rew[eval_disp_name])) + ' '
                # print(printout)

            # if ((args.eval_nondet_interval is not None and j % args.eval_nondet_interval == 0) or j == args.continue_from_epoch):
            eval_test_nondet_rew, _, eval_test_nondet_done, _ = evaluate_procgen_LEEP(actor_critic_0, actor_critic_1, actor_critic_2, actor_critic_3,
                                                                                    eval_envs_dic, 'test_eval', args.num_processes, device, args.num_steps, logger, deterministic=False)


            logger.feed_eval(eval_dic_rew['train_eval'], eval_dic_done['train_eval'],eval_dic_rew['test_eval'], eval_dic_done['test_eval'],  seeds_train, seeds_test,
                             eval_dic_rew['train_eval'], eval_dic_rew['test_eval'], eval_test_nondet_rew, eval_test_nondet_done)

            episode_statistics = logger.get_episode_statistics()
            print(printout)
            print(episode_statistics)


            # summary_writer.add_scalars('eval_mean_rew', {f'{eval_disp_name}': np.mean(eval_dic_rew[eval_disp_name])},
            #                               (j+1) * args.num_processes * args.num_steps)
            # summary_writer.add_scalars('eval_min_rew', {f'{eval_disp_name}': np.min(eval_dic_rew[eval_disp_name])},
            #                               (j+1) * args.num_processes * args.num_steps)
            # summary_writer.add_scalars('eval_max_rew',{f'{eval_disp_name}': np.max(eval_dic_rew[eval_disp_name])},
            #                               (j+1) * args.num_processes * args.num_steps)
            # summary_writer.add_scalars('eval_mean_len', {f'{eval_disp_name}': np.mean(eval_dic_len[eval_disp_name])},
            #                               (j+1) * args.num_processes * args.num_steps)
            # summary_writer.add_scalars('eval_min_len', {f'{eval_disp_name}': np.min(eval_dic_len[eval_disp_name])},
            #                               (j+1) * args.num_processes * args.num_steps)
            # summary_writer.add_scalars('eval_max_len',{f'{eval_disp_name}': np.max(eval_dic_len[eval_disp_name])},
            #                               (j+1) * args.num_processes * args.num_steps)
            #
            # summary_writer.add_scalars('eval_mean_rew', {'train': np.mean(episode_rewards)},
            #                           (j + 1) * args.num_processes * args.num_steps)
            # summary_writer.add_scalars('eval_min_rew', {'train': np.min(episode_rewards)},
            #                           (j + 1) * args.num_processes * args.num_steps)
            # summary_writer.add_scalars('eval_max_rew', {'train': np.max(episode_rewards)},
            #                           (j + 1) * args.num_processes * args.num_steps)
            # summary_writer.add_scalars('eval_mean_len', {'train': np.mean(episode_len)},
            #                           (j + 1) * args.num_processes * args.num_steps)
            # summary_writer.add_scalars('eval_min_len', {'train': np.min(episode_len)},
            #                           (j + 1) * args.num_processes * args.num_steps)
            # summary_writer.add_scalars('eval_max_len', {'train': np.max(episode_len)},
            #                           (j + 1) * args.num_processes * args.num_steps)

            for key, value in episode_statistics.items():
                if isinstance(value, dict):
                    summary_writer.add_scalars(key, value,(j + 1) * args.num_processes * args.num_steps)
                    for key_v, value_v in value.items():
                        wandb.log({key + "/" + key_v: value_v}, step=(j + 1) * args.num_processes * args.num_steps)

                else:
                    summary_writer.add_scalar(key, value, (j + 1) * args.num_processes * args.num_steps)
                # wandb.log({f'eval/{eval_disp_name}': np.mean(eval_r[eval_disp_name])}, step=total_num_steps)

            summary ={'Loss/pi': action_loss_0,
                      'Loss/v': value_loss_0,
                      'Loss/entropy': dist_entropy_0,
                      'Loss/kl': dist_KL_epoch_0}
            for key, value in summary.items():
                summary_writer.add_scalar(key, value, (j + 1) * args.num_processes * args.num_steps)
                wandb.log({key: value}, step=(j + 1) * args.num_processes * args.num_steps)


    # training done. Save and clean up
    save_obj(log_dict, os.path.join(logdir, 'log_dict.pkl'))
    for i in range(args.num_c):
        envs_dic[0].close()
    for eval_disp_name in EVAL_ENVS:
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
