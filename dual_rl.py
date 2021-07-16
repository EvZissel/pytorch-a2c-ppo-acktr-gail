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
from a2c_ppo_acktr.model import Policy, MLPAttnBase, MLPHardAttnBase, MLPHardAttnReinforceBase
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from a2c_ppo_acktr.utils import save_obj, load_obj


EVAL_ENVS = {'partial_train_eval': ['h_bandit-obs-randchoose-v5', 10],
             'test_eval': ['h_bandit-obs-randchoose-v1', 100]}

# Train dual RL using a REINFORCE update for the validation agent



def main():
    args = get_args()
    import random; random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logdir = args.env_name + '_' + args.val_env_name + '_' + str(args.seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(os.path.expanduser(args.log_dir), logdir)
    utils.cleanup_log_dir(logdir)

    # Ugly but simple logging
    log_dict = {
        'task_steps': args.task_steps,
        'seed': args.seed,
        'recurrent': args.recurrent_policy,
        'obs_recurrent': args.obs_recurrent,
        'train_env': args.env_name,
        'test_env': args.val_env_name,
        'cmd': ' '.join(sys.argv[1:])
    }
    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        log_dict[eval_disp_name] = []

    # Tensorboard logging
    summary_writer = SummaryWriter(log_dir=logdir)
    summary_writer.add_hparams({'task_steps': args.task_steps,
                                'seed': args.seed,
                                'recurrent': args.recurrent_policy,
                                'obs_recurrent': args.obs_recurrent,
                                'train_env': args.env_name,
                                'test_env': args.val_env_name,
                                'cmd': ' '.join(sys.argv[1:])}, {})

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    print('making envs...')
    # Training envs
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, steps=args.task_steps,
                         free_exploration=args.free_exploration, recurrent=args.recurrent_policy,
                         obs_recurrent=args.obs_recurrent, multi_task=True, normalize=not args.no_normalize)
    # Validation envs
    val_envs = make_vec_envs(args.val_env_name, args.seed, args.num_processes,
                             args.gamma, args.log_dir, device, False, steps=args.task_steps,
                             free_exploration=args.free_exploration, recurrent=args.recurrent_policy,
                             obs_recurrent=args.obs_recurrent, multi_task=True, normalize=not args.no_normalize)
    # Test envs
    eval_envs_dic = {}
    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        eval_envs_dic[eval_disp_name] = make_vec_envs(eval_env_name[0], args.seed, args.num_processes,
                                                      None, logdir, device, True, steps=args.task_steps,
                                                      recurrent=args.recurrent_policy,
                                                      obs_recurrent=args.obs_recurrent, multi_task=True,
                                                      free_exploration=args.free_exploration, normalize=not args.no_normalize)
    print('done')

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base=MLPHardAttnReinforceBase,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
    actor_critic.to(device)

    # Load previous model
    if (args.continue_from_epoch > 0) and args.save_dir != "":
        save_path = os.path.join(args.save_dir, args.algo)
        actor_critic_, loaded_obs_rms_ = torch.load(os.path.join(save_path,
                                                                 args.env_name +
                                                                 "-epoch-{}.pt".format(args.continue_from_epoch)))
        actor_critic.load_state_dict(actor_critic_.state_dict())

    if args.algo != 'ppo':
        raise print("only PPO is supported")

    # We have two agents that train different parts of the same neural network on different train/validation tasks
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

    # rollout storage for agent
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    # rollout storage for validation agent
    val_rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                  val_envs.observation_space.shape, val_envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    val_obs = val_envs.reset()
    val_rollouts.obs[0].copy_(val_obs)
    val_rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    val_episode_rewards = deque(maxlen=10)

    start = time.time()
    seeds = torch.zeros(args.num_processes, 1)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    # save_copy = True
    for j in range(args.continue_from_epoch, args.continue_from_epoch+num_updates):

        # policy rollouts
        actor_critic.eval()
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, attn_masks = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], rollouts.attn_masks[step])

            # Observe reward and next obs
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
                            action_log_prob, value, reward, masks, bad_masks, attn_masks,seeds)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1], rollouts.attn_masks[-1]).detach()

        actor_critic.train()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # validation rollouts
        for val_iter in range(args.val_agent_steps):    # we allow several PPO steps for each validation update
            actor_critic.eval()
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states, attn_masks = actor_critic.act(
                        val_rollouts.obs[step], val_rollouts.recurrent_hidden_states[step],
                        val_rollouts.masks[step], val_rollouts.attn_masks[step], deterministic=True,
                        attention_act=True)

                # Observe reward and next obs
                obs, reward, done, infos = val_envs.step(action)

                for info in infos:
                    if 'episode' in info.keys():
                        val_episode_rewards.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                val_rollouts.insert(obs, recurrent_hidden_states, action,
                                    action_log_prob, value, reward, masks, bad_masks, attn_masks,seeds)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    val_rollouts.obs[-1], val_rollouts.recurrent_hidden_states[-1],
                    val_rollouts.masks[-1], val_rollouts.attn_masks[-1]).detach()

            actor_critic.train()
            val_rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                         args.gae_lambda, args.use_proper_time_limits)
            val_value_loss, val_action_loss, val_dist_entropy = val_agent.update(val_rollouts, attention_update=True)
            val_rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1):
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(logdir, args.env_name + "-epoch-{}.pt".format(j)))

        # print some stats
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}; validation episodes: mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(val_episode_rewards),
                        np.median(val_episode_rewards), np.min(val_episode_rewards),
                        np.max(val_episode_rewards), dist_entropy, value_loss,
                        action_loss))
            print(actor_critic.base.input_attention)

        # evaluate agent on evaluation tasks
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

                summary_writer.add_scalar(f'eval/{eval_disp_name}', np.mean(eval_r[eval_disp_name]),
                                          (j+1) * args.num_processes * args.num_steps)
                log_dict[eval_disp_name].append([(j+1) * args.num_processes * args.num_steps, eval_r[eval_disp_name]])
                printout += eval_disp_name + ' ' + str(np.mean(eval_r[eval_disp_name])) + ' '
            # summary_writer.add_scalars('eval_combined', eval_r, (j+1) * args.num_processes * args.num_steps)
            print(printout)

    # training done. Save and clean up
    save_obj(log_dict, os.path.join(logdir, 'log_dict.pkl'))
    envs.close()
    val_envs.close()
    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
