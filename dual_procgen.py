import copy
import glob
import os
import time
from collections import deque
import sys

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
# from a2c_ppo_acktr.envs import make_vec_envs, make_ProcgenEnvs
from a2c_ppo_acktr.envs import make_ProcgenEnvs
from procgen import ProcgenEnv
from a2c_ppo_acktr.model import Policy, ImpalaHardAttnReinforce
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate_procgen
from a2c_ppo_acktr.utils import save_obj, load_obj
from a2c_ppo_acktr.procgen_wrappers import *
from a2c_ppo_acktr.logger import Logger
import pandas as pd
import matplotlib.pyplot as plt

EVAL_ENVS = ['train_eval', 'test_eval', 'partial_train_eval']


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

    logdir = args.env_name + '_seed_' + str(args.seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    if args.normalize_rew:
        logdir = logdir + '_normalize_rew'
    logdir = os.path.join(os.path.expanduser(args.log_dir), logdir)
    utils.cleanup_log_dir(logdir)

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

    # print arguments
    print("logdir: " + logdir)
    for key in vars(args):
        print(key, ':', vars(args)[key])

    with open(logdir + '/args.csv', 'w') as f:
        argslog.to_csv(f, index=False)

    # Tensorboard logging
    summary_writer = SummaryWriter(log_dir=logdir)

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.gpu_device) if args.cuda else "cpu")

    print('making envs...')
    # Training envs
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
                            mask_size=args.mask_size,
                            normalize_rew=args.normalize_rew)

    # Validation envs
    val_start_level = args.start_level + args.num_level + 1
    val_envs = make_ProcgenEnvs(num_envs=args.num_processes,
                                                  env_name=args.env_name,
                                                  start_level=val_start_level,
                                                  num_levels=args.num_level,
                                                  distribution_mode=args.distribution_mode,
                                                  use_generated_assets=True,
                                                  use_backgrounds=False,
                                                  restrict_themes=True,
                                                  use_monochrome_assets=True,
                                                  rand_seed=args.seed,
                                                  mask_size=args.mask_size,
                                                  normalize_rew=args.normalize_rew)

    # Test envs
    eval_envs_dic = {}
    eval_envs_dic['train_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                   env_name=args.env_name,
                                                   start_level=args.start_level,
                                                   num_levels=args.num_level,
                                                   distribution_mode=args.distribution_mode,
                                                   use_generated_assets=True,
                                                   use_backgrounds=False,
                                                   restrict_themes=True,
                                                   use_monochrome_assets=True,
                                                   rand_seed=args.seed,
                                                   mask_size=args.mask_size,
                                                   normalize_rew=args.normalize_rew)

    test_start_level = args.start_level + val_start_level + args.num_level + 1
    eval_envs_dic['test_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                  env_name=args.env_name,
                                                  start_level=test_start_level,
                                                  num_levels=args.num_level,
                                                  distribution_mode=args.distribution_mode,
                                                  use_generated_assets=True,
                                                  use_backgrounds=False,
                                                  restrict_themes=True,
                                                  use_monochrome_assets=True,
                                                  rand_seed=args.seed,
                                                  mask_size=args.mask_size,
                                                  normalize_rew=args.normalize_rew)

    partial_start_level = args.start_level + int(0.5*val_start_level)
    eval_envs_dic['partial_train_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                  env_name=args.env_name,
                                                  start_level=partial_start_level,
                                                  num_levels=args.num_level,
                                                  distribution_mode=args.distribution_mode,
                                                  use_generated_assets=True,
                                                  use_backgrounds=False,
                                                  restrict_themes=True,
                                                  use_monochrome_assets=True,
                                                  rand_seed=args.seed,
                                                  mask_size=args.mask_size,
                                                  normalize_rew=args.normalize_rew)

    print('done')

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base=ImpalaHardAttnReinforce,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent,
                     'att_size': [8,8],
                     'obs_size': envs.observation_space.shape})
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

    # validation agent
    val_agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        value_loss_coef=0.0,  # we don't learn the value function with the validation agent
        entropy_coef=0.0,  # we don't implement entropy for reinforce update
        lr=args.val_lr,
        eps=args.eps,
        num_tasks=args.num_processes,
        attention_policy=True,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)


    # Load previous model
    if (args.continue_from_epoch > 0) and args.save_dir != "":
        save_path = args.save_dir
        actor_critic_weighs = torch.load(
            os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(args.continue_from_epoch)))
        actor_critic.load_state_dict(actor_critic_weighs['state_dict'])
        agent.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict'])

    # rollout storage for agent
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size, device)

    # rollout storage for validation agent
    val_rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                  val_envs.observation_space.shape, val_envs.action_space,
                                  actor_critic.recurrent_hidden_state_size, device)

    logger = Logger(args.num_processes)

    obs = envs.reset()
    rollouts.obs[0].copy_(torch.FloatTensor(obs))
    # rollouts.to(device)

    val_obs = val_envs.reset()
    val_rollouts.obs[0].copy_(torch.FloatTensor(val_obs))
    # val_rollouts.to(device)

    seeds = torch.zeros(args.num_processes, 1)
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    # save_copy = True
    save_image_every = num_updates
    # episode_len_buffer = []
    # for _ in range(args.num_processes):
    #     episode_len_buffer.append(0)
    for j in range(args.continue_from_epoch, args.continue_from_epoch + num_updates):

        # plot mazes
        if j % save_image_every == 0:
            fig = plt.figure(figsize=(20, 20))
            columns = 5
            rows = 5
            for i in range(1, columns * rows + 1):
                fig.add_subplot(rows, columns, i)
                plt.imshow(rollouts.obs[0][i].transpose(0, 2))
            summary_writer.add_images('samples_step_{}'.format(j), rollouts.obs[0][0:25])
            plt.show()

        # policy rollouts
        actor_critic.eval()
        # episode_rewards = []
        # episode_len = []
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, attn_masks = actor_critic.act(
                    rollouts.obs[step].to(device), rollouts.recurrent_hidden_states[step].to(device),
                    rollouts.masks[step].to(device), rollouts.attn_masks[step].to(device),device)

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
            rollouts.insert(torch.from_numpy(obs), recurrent_hidden_states, action,
                            action_log_prob, value, torch.from_numpy(reward).unsqueeze(1), masks, bad_masks, attn_masks,
                            seeds, infos)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1].to(device), rollouts.recurrent_hidden_states[-1].to(device),
                rollouts.masks[-1].to(device), rollouts.attn_masks[-1].to(device), device).detach()

        actor_critic.train()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts,device=device)

        rollouts.after_update()

        rew_batch, done_batch = rollouts.fetch_log_data()
        logger.feed_train(rew_batch, done_batch[1:])

        # validation rollouts
        for val_iter in range(args.val_agent_steps):    # we allow several PPO steps for each validation update
            actor_critic.eval()
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states, attn_masks = actor_critic.act(
                        val_rollouts.obs[step].to(device), val_rollouts.recurrent_hidden_states[step].to(device),
                        val_rollouts.masks[step].to(device), val_rollouts.attn_masks[step].to(device), deterministic=True,
                        attention_act=True, device=device)

                # Observe reward and next obs
                obs, reward, done, infos = val_envs.step(action.squeeze().cpu().numpy())

                for i, info in enumerate(infos):
                    seeds[i] = info["level_seed"]

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                val_rollouts.insert(torch.from_numpy(obs), recurrent_hidden_states, action,
                                    action_log_prob, value, torch.from_numpy(reward).unsqueeze(1), masks, bad_masks, attn_masks,seeds, infos)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    val_rollouts.obs[-1].to(device), val_rollouts.recurrent_hidden_states[-1].to(device),
                    val_rollouts.masks[-1].to(device), val_rollouts.attn_masks[-1].to(device),device=device).detach()

            actor_critic.train()
            val_rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                         args.gae_lambda, args.use_proper_time_limits)
            val_value_loss, val_action_loss, val_dist_entropy = val_agent.update(val_rollouts, attention_update=True,device=device)
            val_rollouts.after_update()

            rew_batch, done_batch = val_rollouts.fetch_log_data()
            logger.feed_val(rew_batch, done_batch[1:])

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == args.continue_from_epoch + num_updates - 1):
            torch.save({'state_dict': actor_critic.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(),
                        'step': j}, os.path.join(logdir, args.env_name + "-epoch-{}.pt".format(j)))

        # print some stats
        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            train_statistics = logger.get_train_val_statistics()
            print(
                "Updates {}, num timesteps {}, FPS {}, num training episodes {} \n Last 128 training episodes: mean/median reward {:.1f}/{:.1f}, mean/median reward val {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}, min/max reward val {:.1f}/{:.1f}, dist_entropy {} , value_loss {}, action_loss {}, unique seeds {}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            logger.num_episodes, train_statistics['Rewards_mean_episodes'],
                            train_statistics['Rewards_median_episodes'], train_statistics['Rewards_mean_episodes_val'],
                            train_statistics['Rewards_median_episodes_val'], train_statistics['Rewards_min_episodes'],
                            train_statistics['Rewards_max_episodes'],  train_statistics['Rewards_min_episodes_val'],
                            train_statistics['Rewards_max_episodes_val'], dist_entropy, value_loss,
                            action_loss, np.unique(rollouts.seeds.squeeze().numpy()).size))

        # evaluate agent on evaluation tasks
        if (args.eval_interval is not None and j % args.eval_interval == 0):
            actor_critic.eval()
            printout = f'Seed {args.seed} Iter {j} '
            eval_dic_rew = {}
            eval_dic_done = {}
            for eval_disp_name in EVAL_ENVS:
                eval_dic_rew[eval_disp_name], eval_dic_done[eval_disp_name] = evaluate_procgen(actor_critic,
                                                                                               eval_envs_dic,
                                                                                               eval_disp_name,
                                                                                               args.num_processes,
                                                                                               device, args.num_steps,
                                                                                               logger)

                # log_dict[eval_disp_name].append([(j+1) * args.num_processes * args.num_steps, eval_dic_rew[eval_disp_name]])
                # printout += eval_disp_name + ' ' + str(np.mean(eval_dic_rew[eval_disp_name])) + ' '
                # print(printout)

            logger.feed_eval(eval_dic_rew['train_eval'], eval_dic_done['train_eval'], eval_dic_rew['test_eval'],
                             eval_dic_done['test_eval'],eval_dic_rew['partial_train_eval'], eval_dic_done['partial_train_eval'])
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
                    summary_writer.add_scalars(key, value, (j + 1) * args.num_processes * args.num_steps)
                else:
                    summary_writer.add_scalar(key, value, (j + 1) * args.num_processes * args.num_steps)

            summary = {'Loss/pi': action_loss,
                       'Loss/v': value_loss,
                       'Loss/entropy': dist_entropy}
            for key, value in summary.items():
                summary_writer.add_scalar(key, value, (j + 1) * args.num_processes * args.num_steps)

    # training done. Save and clean up
    save_obj(log_dict, os.path.join(logdir, 'log_dict.pkl'))
    envs.close()
    for eval_disp_name in EVAL_ENVS:
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
