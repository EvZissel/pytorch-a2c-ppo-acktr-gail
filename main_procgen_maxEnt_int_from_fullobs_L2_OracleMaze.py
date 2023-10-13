import copy
import glob
import os
import time
from collections import deque
import sys
from typing import Union

from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
# from a2c_ppo_acktr.envs import make_vec_envs, make_ProcgenEnvs
from a2c_ppo_acktr.envs import make_ProcgenEnvs
from procgen import ProcgenEnv
from a2c_ppo_acktr.model import Policy, MLPAttnBase, MLPHardAttnBase, MLPHardAttnReinforceBase, ImpalaModel
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate_procgen, evaluate_procgen_maxEnt, evaluate_procgen_maxEnt_org, \
    evaluate_procgen_maxEnt_L2, evaluate_procgen_maxEnt_avepool, evaluate_procgen_maxEnt_avepool_original, evaluate_procgen_maxEnt_avepool_original_L2
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

#Here we run all experiments with random policies (Train and Test)
EVAL_ENVS = ['train_eval', 'test_eval_nondet']
# EVAL_ENVS = ['test_eval_nondet']

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                       constant_(x, 0), nn.init.calculate_gain('relu'))
init_2 = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=1)


def main():
    args = get_args()
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logdir_ = args.env_name + '_seed_' + str(args.seed) + '_num_env_' + str(args.num_level) + '_entro_' + str(
        args.entropy_coef) + '_gama_' + str(args.gamma) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
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

    logdir_ = logdir_ + '_L2eval'
    logdir = os.path.join(os.path.expanduser(args.log_dir), logdir_)
    utils.cleanup_log_dir(logdir)

    wandb.init(project=args.env_name + "_PPO_maximum_entropy_L2", entity="ev_zisselman", config=args, name=logdir_,
               id=logdir_)

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
    # Training envs
    # envs = make_ProcgenEnvs(num_envs=args.num_processes,
    #                   env_name=args.env_name,
    #                   start_level=args.start_level,
    #                   num_levels=args.num_level,
    #                   distribution_mode=args.distribution_mode,
    #                   use_generated_assets=True,
    #                   use_backgrounds=False,
    #                   restrict_themes=True,
    #                   use_monochrome_assets=True,
    #                   rand_seed=args.seed,
    #                   mask_size=args.mask_size,
    #                   normalize_rew=args.normalize_rew,
    #                   mask_all=args.mask_all,
    #                   device=device)

    # compute maximun reward per seed

    max_reward_seeds = {
        'train_eval': [],
        'test_eval_nondet': []
    }

    # max_reward_seeds = {
    #     'test_eval_nondet': []
    # }

    test_start_level = args.start_level + args.num_level + 1
    start_train_test = {
        'train_eval': args.start_level,
        'test_eval_nondet': test_start_level
    }

    # start_train_test = {
    #     'test_eval_nondet': test_start_level
    # }

    down_sample_avg = nn.AvgPool2d(args.kernel_size, stride=args.stride)
    # Calculate exact max reward per seed
    for eval_disp_name in EVAL_ENVS:
        for i in range(args.num_test_level):
            envs = make_ProcgenEnvs(num_envs=1,
                                    env_name=args.env_name,
                                    start_level=start_train_test[eval_disp_name] + i,
                                    num_levels=1,
                                    distribution_mode=args.distribution_mode,
                                    use_generated_assets=True,
                                    use_backgrounds=False,
                                    restrict_themes=True,
                                    use_monochrome_assets=True,
                                    rand_seed=args.seed,
                                    center_agent=args.center_agent,
                                    mask_size=args.mask_size,
                                    normalize_rew=args.normalize_rew,
                                    mask_all=args.mask_all)

            obs = envs.reset()
            obs_ds = down_sample_avg(obs[0])

            # # plot mazes
            # if (i == 73):
            #     plt.imshow(obs[0].transpose(0, 2).cpu().numpy())
            #     # plt.savefig("test.png")
            #     plt.show()

            action = torch.full((1, 1), 5)
            done = torch.full((1, 1), 0)
            int_reward = 0.0
            step = 0
            obs_vec_ds = []
            obs_vec_ds.append(obs_ds)

            while not done[0]:
                with torch.no_grad():

                    action = maxEnt_oracle(obs, action)

                    obs, _, done, infos = envs.step(action[0].cpu().numpy())
                    # if (i == 73) and done[0] == 1:
                    #     print("action: {}".format(action[0]))
                    #     plt.imshow(obs[0].transpose(0,2).cpu().numpy())
                    #     plt.show()

                    obs_ds = down_sample_avg(obs)

                    step += 1
                    if done[0] == 1:
                        break
                    if step > args.num_buffer:
                        old_obs = torch.stack(obs_vec_ds[step - args.num_buffer:])
                    else:
                        old_obs = torch.stack(obs_vec_ds)
                    neighbor_size_i = min(args.neighbor_size, step) - 1
                    int_reward += (old_obs - obs_ds).flatten(start_dim=1).norm(p=args.p_norm,dim=1).sort().values[neighbor_size_i]

                    obs_vec_ds.append(obs_ds.squeeze())
                    # print("int_reward: {}".format(int_reward))


            max_reward_seeds[eval_disp_name].append(int_reward)


    envs = make_ProcgenEnvs(num_envs=args.num_processes,
                            env_name=args.env_name,
                            start_level=args.start_level,
                            num_levels=args.num_level,
                            distribution_mode=args.distribution_mode,
                            use_generated_assets=True,
                            use_backgrounds=False,
                            restrict_themes=True,
                            use_monochrome_assets=True,
                            rand_seed=args.seed,
                            center_agent=args.center_agent,
                            mask_size=args.mask_size,
                            normalize_rew=args.normalize_rew,
                            mask_all=args.mask_all,
                            device=device)

    envs_full_obs = make_ProcgenEnvs(num_envs=args.num_processes,
                                     env_name=args.env_name,
                                     start_level=args.start_level,
                                     num_levels=args.num_level,
                                     distribution_mode=args.distribution_mode,
                                     use_generated_assets=True,
                                     use_backgrounds=False,
                                     restrict_themes=True,
                                     use_monochrome_assets=True,
                                     rand_seed=args.seed,
                                     center_agent=False,
                                     mask_size=args.mask_size,
                                     normalize_rew=args.normalize_rew,
                                     mask_all=args.mask_all,
                                     device=device)
    # Test envs
    eval_envs_dic = {}
    eval_envs_dic['train_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                   env_name=args.env_name,
                                                   start_level=args.start_level,
                                                   num_levels=args.num_test_level,
                                                   distribution_mode=args.distribution_mode,
                                                   use_generated_assets=True,
                                                   use_backgrounds=False,
                                                   restrict_themes=True,
                                                   use_monochrome_assets=True,
                                                   rand_seed=args.seed,
                                                   center_agent=args.center_agent,
                                                   mask_size=args.mask_size,
                                                   normalize_rew=args.normalize_rew,
                                                   mask_all=args.mask_all,
                                                   device=device)

    eval_envs_dic['test_eval_nondet'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                  env_name=args.env_name,
                                                  start_level=test_start_level,
                                                  num_levels=args.num_test_level,
                                                  distribution_mode=args.distribution_mode,
                                                  use_generated_assets=True,
                                                  use_backgrounds=False,
                                                  restrict_themes=True,
                                                  use_monochrome_assets=True,
                                                  rand_seed=args.seed,
                                                  center_agent=args.center_agent,
                                                  mask_size=args.mask_size,
                                                  normalize_rew=args.normalize_rew,
                                                  mask_all=args.mask_all,
                                                  device=device)
    # eval_envs_dic_nondet = {}
    # eval_envs_dic_nondet['test_eval_nondet'] =  make_ProcgenEnvs(num_envs=args.num_processes,
    #                                                  env_name=args.env_name,
    #                                                  start_level=test_start_level,
    #                                                  num_levels=args.num_test_level,
    #                                                  distribution_mode=args.distribution_mode,
    #                                                  use_generated_assets=args.use_generated_assets,
    #                                                  use_backgrounds=args.use_backgrounds,
    #                                                  restrict_themes=args.restrict_themes,
    #                                                  use_monochrome_assets=args.use_monochrome_assets,
    #                                                  rand_seed=args.seed,
    #                                                  center_agent=True,
    #                                                  mask_size=args.mask_size,
    #                                                  normalize_rew=args.normalize_rew,
    #                                                  mask_all=args.mask_all,
    #                                                  device=device)

    # Test envs full observation
    eval_envs_dic_full_obs = {}
    eval_envs_dic_full_obs['train_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                            env_name=args.env_name,
                                                            start_level=args.start_level,
                                                            num_levels=args.num_test_level,
                                                            distribution_mode=args.distribution_mode,
                                                            use_generated_assets=True,
                                                            use_backgrounds=False,
                                                            restrict_themes=True,
                                                            use_monochrome_assets=True,
                                                            rand_seed=args.seed,
                                                            center_agent=False,
                                                            mask_size=args.mask_size,
                                                            normalize_rew=args.normalize_rew,
                                                            mask_all=args.mask_all,
                                                            device=device)

    eval_envs_dic_full_obs['test_eval_nondet'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                           env_name=args.env_name,
                                                           start_level=test_start_level,
                                                           num_levels=args.num_test_level,
                                                           distribution_mode=args.distribution_mode,
                                                           use_generated_assets=True,
                                                           use_backgrounds=False,
                                                           restrict_themes=True,
                                                           use_monochrome_assets=True,
                                                           rand_seed=args.seed,
                                                           center_agent=False,
                                                           mask_size=args.mask_size,
                                                           normalize_rew=args.normalize_rew,
                                                           mask_all=args.mask_all,
                                                           device=device)
    print('done')

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent,
                     'hidden_size': args.recurrent_hidden_size, 'gray_scale': args.gray_scale},
        epsilon_RPO=args.epsilon_RPO)
    # base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic.to(device)

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
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size, args.mask_size, device=device)

    # Load previous model
    if (args.continue_from_epoch > 0) and args.save_dir != "":
        save_path = args.save_dir
        actor_critic_weighs = torch.load(
            os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(args.continue_from_epoch)),
            map_location=device)
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

    logger = maxEnt_Logger(args.num_processes, max_reward_seeds, start_train_test, envs.observation_space.shape,
                           envs.observation_space.shape, actor_critic.recurrent_hidden_state_size, device=device)

    obs = envs.reset()
    obs_full = envs_full_obs.reset()
    obs_ds = down_sample_avg(obs_full)

    # We sum the difference between two consecutive observation
    # rollouts.obs[0].copy_(torch.FloatTensor(obs))
    rollouts.obs[0].copy_(obs)
    rollouts.obs_ds[0].copy_(obs_ds)
    rollouts.obs_full.copy_(obs_full)
    rollouts.obs_sum.copy_(torch.zeros_like(obs_full))
    rollouts.obs0.copy_(obs_full)
    # rollouts.to(device)

    # we use diff_obs as obs without the trail
    # rollouts.clean_obs.copy_((obs[:,1,:,:] != 1).unsqueeze(1)*obs)

    obs_train = eval_envs_dic['train_eval'].reset()
    logger.obs['train_eval'].copy_(obs_train)
    obs_train_ds = down_sample_avg(obs_train)
    for i in range(args.num_processes):
        logger.obs_vec_ds['train_eval'][i].append(obs_train_ds[i])
    obs_train_full = eval_envs_dic_full_obs['train_eval'].reset()
    logger.obs_full['train_eval'].copy_(obs_train_full)
    logger.obs_sum['train_eval'].copy_(torch.zeros_like(obs_train_full))
    logger.obs0['train_eval'].copy_(obs_train_full)
    # logger.clean_obs['train_eval'].copy_((obs_train[:,1,:,:] != 1).unsqueeze(1)*obs_train)

    obs_test_nondet = eval_envs_dic['test_eval_nondet'].reset()
    logger.obs['test_eval_nondet'].copy_(obs_test_nondet)
    obs_test_ds = down_sample_avg(obs_test_nondet)
    for i in range(args.num_processes):
        logger.obs_vec_ds['test_eval_nondet'][i].append(obs_test_ds[i])
    obs_test_full = eval_envs_dic_full_obs['test_eval_nondet'].reset()
    logger.obs_full['test_eval_nondet'].copy_(obs_test_full)
    logger.obs_sum['test_eval_nondet'].copy_(torch.zeros_like(obs_test_full))
    logger.obs0['test_eval_nondet'].copy_(obs_test_full)

    # logger.clean_obs['test_eval'].copy_((obs_test[:, 1, :, :] != 1).unsqueeze(1)*obs_test)
    # obs_test_nondet = eval_envs_dic_nondet['test_eval_nondet'].reset()
    # logger.obs['test_eval_nondet'].copy_(obs_test_nondet)
    # for i in range(args.num_processes):
    #     logger.obs_vec['test_eval_nondet'][i].append(obs_train[i])
    # logger.obs_sum['test_eval_nondet'].copy_(obs_test_nondet)
    # logger.obs0['test_eval_nondet'].copy_(obs_test_nondet)
    # logger.clean_obs['test_eval_nondet'].copy_((obs_test_nondet[:, 1, :, :] != 1).unsqueeze(1) * obs_test_nondet)
    # plot mazes

    fig = plt.figure(figsize=(20, 20))
    columns = 5
    rows = 5
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(rollouts.obs[0][i].transpose(0, 2))
        plt.savefig(logdir + '/fig.png')

    seeds = torch.zeros(args.num_processes, 1)
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    # save_copy = True
    save_image_every = int(num_updates / 10)
    # episode_len_buffer = []
    # for _ in range(args.num_processes):
    #     episode_len_buffer.append(0)
    # eval_test_nondet_rew = np.zeros((args.num_steps, args.num_processes))
    # eval_test_nondet_done = np.zeros((args.num_steps, args.num_processes))

    for j in range(args.continue_from_epoch, args.continue_from_epoch + num_updates):

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
        actor_critic.eval()
        # episode_rewards = []
        # episode_len = []
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, _, recurrent_hidden_states, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic.act(
                    rollouts.obs[step].to(device), rollouts.recurrent_hidden_states[step].to(device),
                    rollouts.masks[step].to(device), rollouts.attn_masks[step].to(device),
                    rollouts.attn_masks1[step].to(device), rollouts.attn_masks2[step].to(device),
                    rollouts.attn_masks3[step].to(device))

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action.squeeze().cpu().numpy())
            obs_full, reward_full, done_full, infos_full = envs_full_obs.step(action.squeeze().cpu().numpy())
            # if max(reward) < 10 and max(reward) >0:
            #     print(reward)
            int_reward = np.zeros_like(reward)
            obs_ds = down_sample_avg(obs_full)
            # next_obs_diff = 1 * ((obs_full - rollouts.obs_full.to(device)).abs() > 1e-5)
            # next_obs_sum = rollouts.obs_sum.to(device) + next_obs_diff
            # next_obs_sum = (1 * (down_sample_avg(next_obs_sum).abs() > 1e-5)).sum(1)
            # obs_sum = (1 * (down_sample_avg(rollouts.obs_sum.to(device)).abs() > 1e-5)).sum(1)

            diff_all = obs_ds.unsqueeze(0) - rollouts.obs_ds.to(device)

            for i in range(len(done)):
                if done[i] == 1:
                    rollouts.obs_sum[i] = torch.zeros_like(rollouts.obs_full[i])
                    rollouts.obs_full[i].copy_(obs_full[i])
                    # rollouts.obs_sum[i].copy_(obs[i].cpu())
                    # rollouts.obs0[i].copy_(obs_full[i].cpu())
                    rollouts.step_env[i] = 0
                    # rollouts.diff_obs[step][i].copy_(torch.zeros_like(obs[i]).cpu())
                else:
                    actual_step_env = int(max(0, rollouts.step_env[i] - args.num_buffer))

                    episode_start = int(step + 1 - rollouts.step_env[i] + actual_step_env)
                    diff = diff_all[max(0, episode_start):step+1][:, i, :, :]
                    if episode_start < 0:
                        if not len(diff):
                            diff = diff_all[args.num_steps + episode_start:args.num_steps][:, i, :, :]
                        else:
                            diff = torch.cat((diff, diff_all[args.num_steps + episode_start:args.num_steps][:, i, :, :].to(device)), dim=0)
                    # diff = down_sample_avg(diff)
                    if args.p_norm == 0:
                        diff = (1.0 * (diff.abs() > 1e-5)).sum(1)
                    neighbor_size = args.neighbor_size
                    if len(diff) < args.neighbor_size:
                        neighbor_size = len(diff)
                    int_reward[i] = diff.flatten(start_dim=1).norm(p=args.p_norm, dim=1).sort().values[int(neighbor_size-1)]


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
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, torch.from_numpy(int_reward).unsqueeze(1), masks, bad_masks,
                            attn_masks, attn_masks1, attn_masks2, attn_masks3, seeds, infos, obs_full)
            rollouts.obs_ds[rollouts.step].copy_(obs_ds)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1].to(device), rollouts.recurrent_hidden_states[-1].to(device),
                rollouts.masks[-1].to(device), rollouts.attn_masks[-1].to(device), rollouts.attn_masks1[-1].to(device),
                rollouts.attn_masks2[-1].to(device), rollouts.attn_masks3[-1].to(device)).detach()

        actor_critic.train()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, _ = agent.update(rollouts)

        rollouts.after_update()

        rew_batch, done_batch = rollouts.fetch_log_data()
        logger.feed_train(rew_batch, done_batch[1:])

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == args.continue_from_epoch + num_updates - 1):
            torch.save({'state_dict': actor_critic.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(),
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
                "min/max reward {:.1f}/{:.1f}, dist_entropy {} , value_loss {}, action_loss {}, unique seeds {}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            logger.num_episodes, train_statistics['Rewards_mean_episodes'],
                            train_statistics['Rewards_median_episodes'], train_statistics['Rewards_min_episodes'],
                            train_statistics['Rewards_max_episodes'], dist_entropy, value_loss,
                            action_loss, np.unique(rollouts.seeds.squeeze().numpy()).size))
        # evaluate agent on evaluation tasks
        if ((args.eval_interval is not None and j % args.eval_interval == 0) or j == args.continue_from_epoch):
            actor_critic.eval()
            printout = f'Seed {args.seed} Iter {j} '
            eval_dic_rew = {}
            eval_dic_int_rew = {}
            eval_dic_done = {}
            eval_dic_seeds = {}

            for eval_disp_name in EVAL_ENVS:
                eval_dic_rew[eval_disp_name], eval_dic_int_rew[eval_disp_name], eval_dic_done[eval_disp_name], \
                eval_dic_seeds[eval_disp_name] = evaluate_procgen_maxEnt_avepool_original_L2(actor_critic, eval_envs_dic,
                                                                                          eval_envs_dic_full_obs,
                                                                                          eval_disp_name,
                                                                                          args.num_processes, device,
                                                                                          args.num_steps, logger, args.num_buffer,
                                                                                          kernel_size=args.kernel_size,
                                                                                          stride=args.stride, deterministic=False, p_norm=args.p_norm, neighbor_size=args.neighbor_size)
                # log_dict[eval_disp_name].append([(j+1) * args.num_processes * args.num_steps, eval_dic_rew[eval_disp_name]])
                # printout += eval_disp_name + ' ' + str(np.mean(eval_dic_rew[eval_disp_name])) + ' '
                # print(printout)
                # wandb.log({"mun_maxEnt/"+eval_disp_name: np.mean(num_zero_obs_end[eval_disp_name])}, step=(j + 1) * args.num_processes * args.num_steps)
                # wandb.log({"mun_maxEnt_oracle/"+eval_disp_name: np.mean(num_zero_obs_end_oracle[eval_disp_name])}, step=(j + 1) * args.num_processes * args.num_steps)
                # wandb.log({"mun_maxEnt_vs_oracle/"+eval_disp_name: np.mean(num_zero_obs_end[eval_disp_name])/np.mean(num_zero_obs_end_oracle[eval_disp_name])}, step=(j + 1) * args.num_processes * args.num_steps)

            # # if ((args.eval_nondet_interval is not None and j % args.eval_nondet_interval == 0) or j == args.continue_from_epoch):
            # eval_test_nondet_rew, eval_test_nondet_int_rew, eval_test_nondet_done, eval_test_nondet_seeds = evaluate_procgen_maxEnt_L2(actor_critic, eval_envs_dic_nondet, 'test_eval_nondet',
            #                                       args.num_processes, device, args.num_steps, logger, args.eps_diff_NN, args.eps_NN, args.num_buffer, args.reset_cont, deterministic=False)
            #     # wandb.log({"mun_maxEnt/nondet": np.mean(num_zero_obs_end_nondet)}, step=(j + 1) * args.num_processes * args.num_steps)

            logger.feed_eval_test_nondet(eval_dic_int_rew['train_eval'], eval_dic_done['train_eval'], eval_dic_rew['train_eval'],
                                         eval_dic_int_rew['test_eval_nondet'], eval_dic_done['test_eval_nondet'], eval_dic_rew['test_eval_nondet'],
                                         eval_dic_seeds['train_eval'], eval_dic_seeds['test_eval_nondet'])
            episode_statistics = logger.get_episode_statistics_test_nondet()
            print(printout)
            print(episode_statistics)

            # # reinitialize the last layers of networks + GRU unit
            # if args.reinitialization and (j % 500 == 0):
            #     print('initialize weights j = {}'.format(j))
            #     init_2(actor_critic.base.critic_linear)
            #     init_(actor_critic.base.main[5])
            #     for name, param in actor_critic.base.gru.named_parameters():
            #         if 'bias' in name:
            #             nn.init.constant_(param, 0)
            #         elif 'weight' in name:
            #             nn.init.orthogonal_(param)

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
                    summary_writer.add_scalars(key, value, (j + 1) * args.num_processes * args.num_steps)
                    for key_v, value_v in value.items():
                        wandb.log({key + "/" + key_v: value_v}, step=(j + 1) * args.num_processes * args.num_steps)

                else:
                    summary_writer.add_scalar(key, value, (j + 1) * args.num_processes * args.num_steps)
                # wandb.log({f'eval/{eval_disp_name}': np.mean(eval_r[eval_disp_name])}, step=total_num_steps)

            summary = {'Loss/pi': action_loss,
                       'Loss/v': value_loss,
                       'Loss/entropy': dist_entropy}
            for key, value in summary.items():
                summary_writer.add_scalar(key, value, (j + 1) * args.num_processes * args.num_steps)
                wandb.log({key: value}, step=(j + 1) * args.num_processes * args.num_steps)

    # training done. Save and clean up
    save_obj(log_dict, os.path.join(logdir, 'log_dict.pkl'))
    envs.close()
    for eval_disp_name in EVAL_ENVS:
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
