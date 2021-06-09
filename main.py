import copy
import glob
import os
import time
from collections import deque
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from a2c_ppo_acktr.utils import save_obj, load_obj

# EVAL_ENVS = {'three_arms': 'h_bandit-randchoose-v0',
#              'five_arms': 'h_bandit-randchoose-v2',
#              'many_arms': 'h_bandit-randchoose-v1'}

# EVAL_ENVS = {'five_arms': 'h_bandit-randchoose-v2',
#              'ten_arms': 'h_bandit-randchoose-v3',
#              'many_arms': 'h_bandit-randchoose-v1'}

EVAL_ENVS = {'five_arms': 'h_bandit-randchoose-v6',
             'ten_arms': 'h_bandit-randchoose-v5',
             'many_arms': 'h_bandit-randchoose-v1'}

# EVAL_ENVS = {'many_arms': 'h_bandit-randchoose-v1'}

def main():
    args = get_args()
    import random; random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logdir = args.env_name + '_' + args.algo + '_num_arms_' + str(args.num_processes) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    if args.use_privacy:
        logdir = logdir + '_privacy'
    elif args.use_noisygrad:
        logdir = logdir + '_noisygrad'
    elif args.use_pcgrad:
        logdir = logdir + '_pcgrad'
    elif args.use_testgrad:
        logdir = logdir + '_testgrad'
    elif args.use_median_grad:
        logdir = logdir + '_mediangrad'
    logdir = os.path.join('runs', logdir)
    logdir = os.path.join(os.path.expanduser(args.log_dir), logdir)
    utils.cleanup_log_dir(logdir)

    # Ugly but simple logging
    log_dict = {
        'task_steps': args.task_steps,
        'grad_noise_ratio': args.grad_noise_ratio,
        'max_task_grad_norm': args.max_task_grad_norm,
        'use_noisygrad': args.use_noisygrad,
        'use_pcgrad': args.use_pcgrad,
        'use_testgrad': args.use_testgrad,
        'use_testgrad_median': args.use_testgrad_median,
        'testgrad_quantile': args.testgrad_quantile,
        'median_grad': args.use_median_grad,
        'use_meanvargrad': args.use_meanvargrad,
        'meanvar_beta': args.meanvar_beta,
        'no_special_grad_for_critic': args.no_special_grad_for_critic,
        'use_privacy': args.use_privacy,
        'seed': args.seed,
        'recurrent': args.recurrent_policy,
        'obs_recurrent': args.obs_recurrent,
        'cmd': ' '.join(sys.argv[1:])
    }
    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        log_dict[eval_disp_name] = []

    summary_writer = SummaryWriter()
    summary_writer.add_hparams({'task_steps': args.task_steps,
                                'grad_noise_ratio': args.grad_noise_ratio,
                                'max_task_grad_norm': args.max_task_grad_norm,
                                'use_noisygrad': args.use_noisygrad,
                                'use_pcgrad': args.use_pcgrad,
                                'use_testgrad': args.use_testgrad,
                                'use_testgrad_median': args.use_testgrad_median,
                                'testgrad_quantile': args.testgrad_quantile,
                                'median_grad': args.use_median_grad,
                                'use_meanvargrad': args.use_meanvargrad,
                                'meanvar_beta': args.meanvar_beta,
                                'no_special_grad_for_critic': args.no_special_grad_for_critic,
                                'use_privacy': args.use_privacy,
                                'seed': args.seed,
                                'recurrent': args.recurrent_policy,
                                'obs_recurrent': args.obs_recurrent,
                                'cmd': ' '.join(sys.argv[1:])}, {})

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, steps=args.task_steps,
                         free_exploration=args.free_exploration, recurrent=args.recurrent_policy,
                         obs_recurrent=args.obs_recurrent, multi_task=True)
    eval_envs_dic = {}
    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        eval_envs_dic[eval_disp_name] = make_vec_envs(eval_env_name, args.seed + args.num_processes, args.num_processes,
                                                      None, logdir, device, True, steps=args.task_steps,
                                                      recurrent=args.recurrent_policy,
                                                      obs_recurrent=args.obs_recurrent, multi_task=True,
                                                      free_exploration=args.free_exploration)
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic.to(device)

    if (args.continue_from_epoch > 0) and args.save_dir != "":
        save_path = os.path.join(args.save_dir, args.algo)
        actor_critic_, loaded_obs_rms_ = torch.load(os.path.join(save_path,
                                                                   args.env_name +
                                                                   "-epoch-{}.pt".format(args.continue_from_epoch)))
        actor_critic.load_state_dict(actor_critic_.state_dict())

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            num_tasks=args.num_processes,
            use_pcgrad=args.use_pcgrad,
            use_noisygrad=args.use_noisygrad,
            use_testgrad=args.use_testgrad,
            use_graddrop=args.use_graddrop,
            use_testgrad_median=args.use_testgrad_median,
            testgrad_quantile=args.testgrad_quantile,
            use_privacy=args.use_privacy,
            use_median_grad=args.use_median_grad,
            use_meanvargrad=args.use_meanvargrad,
            meanvar_beta=args.meanvar_beta,
            max_task_grad_norm=args.max_task_grad_norm,
            grad_noise_ratio=args.grad_noise_ratio,
            testgrad_alpha=args.testgrad_alpha,
            testgrad_beta=args.testgrad_beta,
            no_special_grad_for_critic=args.no_special_grad_for_critic,
            weight_decay=args.weight_decay)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(args.continue_from_epoch, args.continue_from_epoch+num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            actor_critic.eval()
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            actor_critic.train()

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    for k, v in info['episode'].items():
                        summary_writer.add_scalar(f'training/{k}', v, j * args.num_processes * args.num_steps + args.num_processes * step)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        actor_critic.eval()
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        actor_critic.train()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(j)))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            actor_critic.eval()
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            # obs_rms = 0
            eval_r = {}
            printout = f'Seed {args.seed} Iter {j} '
            for eval_disp_name, eval_env_name in EVAL_ENVS.items():
                # print(eval_disp_name)
                eval_r[eval_disp_name] = evaluate(actor_critic, obs_rms, eval_envs_dic, eval_disp_name, args.seed,
                                                  args.num_processes, logdir, device, steps=args.task_steps,
                                                  recurrent=args.recurrent_policy, obs_recurrent=args.obs_recurrent,
                                                  multi_task=True, free_exploration=args.free_exploration)
                summary_writer.add_scalar(f'eval/{eval_disp_name}', eval_r[eval_disp_name], (j+1) * args.num_processes * args.num_steps)
                log_dict[eval_disp_name].append([(j+1) * args.num_processes * args.num_steps, eval_r[eval_disp_name]])
                printout += eval_disp_name + ' ' + str(eval_r[eval_disp_name]) + ' '

            summary_writer.add_scalars('eval_combined', eval_r, (j+1) * args.num_processes * args.num_steps)
            print(printout)
            actor_critic.train()

    save_obj(log_dict, os.path.join(logdir, 'log_dict.pkl'))
    envs.close()
    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        eval_envs_dic[eval_disp_name].close()



if __name__ == "__main__":
    main()
