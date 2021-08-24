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
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from a2c_ppo_acktr.utils import save_obj, load_obj
import pandas as pd


# EVAL_ENVS = {'five_arms': ['h_bandit-randchoose-v6', 5],
#              'ten_arms': ['h_bandit-randchoose-v5', 10],
#              'many_arms': ['h_bandit-randchoose-v1', 100]}

# EVAL_ENVS = {'ten_arms': ['h_bandit-obs-randchoose-v5', 10],
#              'many_arms': ['h_bandit-obs-randchoose-v1', 100]}

# EVAL_ENVS = {'train_eval': ['h_bandit-obs-randchoose-v8', 25],
#              'test_eval': ['h_bandit-obs-randchoose-v1', 100]}


def main():
    args = get_args()
    import random; random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    EVAL_ENVS = {'train_eval': [args.env_name, args.num_processes],
                 'test_eval': ['h_bandit-obs-randchoose-v1', 100]}

    logdir = args.env_name + '_' + args.algo + '_seed_' + str(args.seed) + '_num_arms_' + str(args.num_processes) + '_entro_' + str(args.entropy_coef) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    # if args.use_privacy:
    #     logdir = logdir + '_privacy'
    # elif args.use_noisygrad:
    #     logdir = logdir + '_noisygrad'
    # elif args.use_pcgrad:
    #     logdir = logdir + '_pcgrad'
    # elif args.use_testgrad:
    #     logdir = logdir + '_testgrad'
    # elif args.use_median_grad:
    #     logdir = logdir + '_mediangrad'

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

    argslog = pd.DataFrame(columns=['args', 'value'])
    for key in vars(args):
        log = [key] + [vars(args)[key]]
        argslog.loc[len(argslog)] = log

    print("logdir: " + logdir)
    for key in vars(args):
        print(key, ':', vars(args)[key])

    with open(logdir + '/args.csv', 'w') as f:
        argslog.to_csv(f, index=False)

    summary_writer = SummaryWriter(log_dir=logdir)
    # summary_writer.add_hparams({'task_steps': args.task_steps,
    #                             'grad_noise_ratio': args.grad_noise_ratio,
    #                             'max_task_grad_norm': args.max_task_grad_norm,
    #                             'use_noisygrad': args.use_noisygrad,
    #                             'use_pcgrad': args.use_pcgrad,
    #                             'use_testgrad': args.use_testgrad,
    #                             'use_testgrad_median': args.use_testgrad_median,
    #                             'testgrad_quantile': args.testgrad_quantile,
    #                             'median_grad': args.use_median_grad,
    #                             'use_meanvargrad': args.use_meanvargrad,
    #                             'meanvar_beta': args.meanvar_beta,
    #                             'no_special_grad_for_critic': args.no_special_grad_for_critic,
    #                             'use_privacy': args.use_privacy,
    #                             'seed': args.seed,
    #                             'recurrent': args.recurrent_policy,
    #                             'obs_recurrent': args.obs_recurrent,
    #                             'cmd': ' '.join(sys.argv[1:])}, {})
    summary_writer.add_hparams(vars(args), {})

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.gpu_device) if args.cuda else "cpu")

    print('making envs...')
    monitor_dir_train = os.path.join(logdir, 'monitor_train')
    utils.cleanup_log_dir(monitor_dir_train)
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, monitor_dir_train, device, False, steps=args.task_steps,
                         free_exploration=args.free_exploration, recurrent=args.recurrent_policy,
                         obs_recurrent=args.obs_recurrent, multi_task=True, normalize=not args.no_normalize, rotate=args.rotate)

    eval_envs_dic = {}
    monitor_dir_test = os.path.join(logdir, 'monitor_test')
    utils.cleanup_log_dir(monitor_dir_test)
    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        eval_envs_dic[eval_disp_name] = make_vec_envs(eval_env_name[0], args.seed, args.num_processes,
                                                      None, monitor_dir_test, device, True, steps=args.task_steps,
                                                      recurrent=args.recurrent_policy,
                                                      obs_recurrent=args.obs_recurrent, multi_task=True,
                                                      free_exploration=args.free_exploration, normalize=not args.no_normalize, rotate=args.rotate)

    prev_eval_r = {}
    print('done')

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic.to(device)

    # if (args.continue_from_epoch > 0) and args.save_dir != "":
    #     save_path = os.path.join(args.save_dir, args.algo)
    #     actor_critic_, loaded_obs_rms_ = torch.load(os.path.join(save_path,
    #                                                                args.env_name +
    #                                                                "-epoch-{}.pt".format(args.continue_from_epoch)))
    #     actor_critic.load_state_dict(actor_critic_.state_dict())


    if args.algo != 'ppo':
        raise "only PPO is supported"
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


    # Load previous model
    if (args.continue_from_epoch > 0) and args.save_dir != "":
        save_path = args.save_dir
        actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(args.continue_from_epoch)))
        actor_critic.load_state_dict(actor_critic_weighs['state_dict'])
        agent.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict'])


    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=25)
    episode_len = deque(maxlen=25)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    save_copy = True
    for j in range(args.continue_from_epoch, args.continue_from_epoch+num_updates):

        for step in range(args.num_steps):
            # Sample actions
            actor_critic.eval()
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, attn_masks = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], rollouts.attn_masks[step])
            actor_critic.train()


            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action.cpu())

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_len.append(info['episode']['l'])
                    # for k, v in info['episode'].items():
                    #     summary_writer.add_scalar(f'training/{k}', v, j * args.num_processes * args.num_steps + args.num_processes * step)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks, attn_masks)

        actor_critic.eval()
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1], rollouts.attn_masks[-1]).detach()
        actor_critic.train()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        # if save_copy:
        #     prev_weights = copy.deepcopy(actor_critic.state_dict())
        #     prev_opt_state = copy.deepcopy(agent.optimizer.state_dict())
        #     save_copy = False

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # # save for every interval-th episode or for the last epoch
        # if (j % args.save_interval == 0
        #         or j == num_updates - 1) and args.save_dir != "":
        #     save_path = os.path.join(args.save_dir, args.algo)
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass
        #
        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
        #     ], os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(j)))

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == args.continue_from_epoch + num_updates - 1):
            torch.save({'state_dict': actor_critic.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(),
                        'step': j, 'obs_rms': getattr(utils.get_vec_normalize(envs), 'obs_rms', None)}, os.path.join(logdir, args.env_name + "-epoch-{}.pt".format(j)))

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
        revert = False
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            actor_critic.eval()
            obs_rms = None if args.no_normalize else utils.get_vec_normalize(envs).obs_rms
            eval_r = {}
            printout = f'Seed {args.seed} Iter {j} '
            for eval_disp_name, eval_env_name in EVAL_ENVS.items():
                eval_r[eval_disp_name] = evaluate(actor_critic, obs_rms, eval_envs_dic, eval_disp_name, args.seed,
                                                  args.num_processes, eval_env_name[1], logdir, device, steps=args.task_steps,
                                                  recurrent=args.recurrent_policy, obs_recurrent=args.obs_recurrent,
                                                  multi_task=True, free_exploration=args.free_exploration)
                # if eval_disp_name in prev_eval_r:
                #     diff = np.array(eval_r[eval_disp_name]) - np.array(prev_eval_r[eval_disp_name])
                #     if eval_disp_name == 'many_arms':
                #         if np.sum(diff > 0) - np.sum(diff < 0) < args.val_improvement_threshold:
                #             print('no update')
                #             revert = True

                summary_writer.add_scalar(f'eval/{eval_disp_name}', np.mean(eval_r[eval_disp_name]),
                                          (j+1) * args.num_processes * args.num_steps)
                log_dict[eval_disp_name].append([(j+1) * args.num_processes * args.num_steps, eval_r[eval_disp_name]])
                printout += eval_disp_name + ' ' + str(np.mean(eval_r[eval_disp_name])) + ' '
            # summary_writer.add_scalars('eval_combined', eval_r, (j+1) * args.num_processes * args.num_steps)
            # if revert:
            #     actor_critic.load_state_dict(prev_weights)
            #     agent.optimizer.load_state_dict(prev_opt_state)
            # else:
            #     print(printout)
            #     prev_eval_r = eval_r.copy()
            # save_copy = True
            print(printout)
            actor_critic.train()

    save_obj(log_dict, os.path.join(logdir, 'log_dict.pkl'))
    envs.close()
    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
