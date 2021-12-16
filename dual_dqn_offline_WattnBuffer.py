#!/usr/bin/env python3

"""
Usage:

$ . ~/env/bin/activate

Example pong command (~900k ts solve):
    python dual_dqn_offline.py \
        --env "PongNoFrameskip-v4" --CnnDQN --learning_rate 0.00001 \
        --target_update_rate 0.1 --replay_size 100000 --start_train_ts 10000 \
        --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 30000 --max_ts 1400000 \
        --batch_size 32 --gamma 0.99 --log_every 10000
 """

import argparse
import math
import math
import random
from copy import deepcopy
import os
import time
from collections import deque

import numpy as np
import torch
import torch.optim as optim
from helpers_dqn import ReplayBuffer, make_atari, make_gym_env, wrap_deepmind, wrap_pytorch, ReplayBufferBandit, AttnReplayBufferBandit
from models_dqn import DQN, CnnDQN, DQN_Attention, DQN_Attention_Wvalue
from a2c_ppo_acktr.envs import make_vec_envs
from torch.utils.tensorboard import SummaryWriter
from a2c_ppo_acktr import utils
import pandas as pd


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("Using GPU: GPU requested and available.")
    dtype = torch.cuda.FloatTensor
    dtypelong = torch.cuda.LongTensor

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    print("NOT Using GPU: GPU not requested or not available.")
    dtype = torch.FloatTensor
    dtypelong = torch.LongTensor


class Agent:
    def __init__(self, env, q_network, target_q_network):
        self.env = env
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.num_actions = env.action_space.n

    def act(self, state, hidden_state, masks, attn_masks):
        # state = torch.tensor(np.float32(state)).type(dtype).unsqueeze(0)
        q_value, value, rnn_hxs, attn_log_probs, attn_masks = self.q_network.forward(state, hidden_state, masks, attn_masks)
        return q_value.max(1)[1].unsqueeze(-1), value, rnn_hxs, attn_log_probs, attn_masks

    def get_value(self, state, hidden_state, masks, attn_masks):
        _, value, _, _, _ = self.q_network.forward(state, hidden_state, masks, attn_masks)
        return value


def compute_td_loss(agent, num_mini_batch, mini_batch_size, replay_buffer, optimizer, gamma, loss_var_coeff):
    num_processes = replay_buffer.rewards.size(1)
    num_steps = replay_buffer.rewards.size(0)
    num_steps_per_batch = int(num_steps/num_mini_batch)

    all_losses =[]
    start_ind_array = [i for i in range(0, num_steps, num_steps_per_batch)]
    # random.shuffle(start_ind_array)
    start_ind_array = random.choices(start_ind_array, k=mini_batch_size)

    for start_ind in start_ind_array:
        data_sampler = replay_buffer.sampler(num_processes, start_ind, num_steps_per_batch)

        losses = []
        recurrent_hidden = torch.zeros(1, agent.q_network.recurrent_hidden_state_size).type(dtypelong)
        # attn_masks = torch.ones(1, replay_buffer.obs.size(2)).type(dtypelong)
        for states, actions, rewards, masks in data_sampler:

            # double q-learning
            with torch.no_grad():
                probs = torch.sigmoid(agent.q_network.input_attention)
                probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
                sampled_mask = probs.sample() / (1 - probs.probs)
                attn_masks = sampled_mask.repeat([states.shape[0], 1]).type(dtype)
                online_q_values, _, _, _, attn_masks  = agent.q_network(states, recurrent_hidden, masks, attn_masks, reuse_masks=True)
                _, max_indicies = torch.max(online_q_values, dim=1)
                target_q_values, _, _, _, attn_masks = agent.target_q_network(states, recurrent_hidden, masks, attn_masks, reuse_masks=True)
                next_q_value = target_q_values.gather(1, max_indicies.unsqueeze(1))

                next_q_value = next_q_value * masks
                expected_q_value = (rewards + gamma * next_q_value[1:, :]).squeeze(1)

            # Normal DDQN update
            q_values, _, _, _, attn_masks = agent.q_network(states, recurrent_hidden, masks, attn_masks, reuse_masks=True)
            q_value = q_values[:-1, :].gather(1, actions).squeeze(1)

            losses.append((q_value - expected_q_value.data).pow(2))


        losses = torch.stack(losses, 1)
        loss = losses.mean(0)
        all_losses.append(loss)

    all_losses = torch.stack(all_losses).mean(0)
    total_loss = all_losses.mean() + loss_var_coeff * all_losses.var()
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss, all_losses.mean(), all_losses.var()


def Atten_update(agent, replay_buffer, optimizer):
    advantages = replay_buffer.returns[:-1] - replay_buffer.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    num_processes = replay_buffer.rewards.size(1)
    num_steps = replay_buffer.rewards.size(0)
    # num_steps_per_batch = int(num_steps/num_mini_batch)

    agent.q_network.train()
    data_sampler = replay_buffer.sampler(num_processes, advantages, start_ind=0, num_steps_per_mini_batch=num_steps)
    recurrent_hidden = torch.zeros(1, agent.q_network.recurrent_hidden_state_size).type(dtype)
    losses_attn = []

    for obs_batch, actions_batch, rewards_batch, value_preds_batch, return_batch, masks_batch, attn_masks_batch, adv_targ in data_sampler:

        _, values, recurrent_hidden, attn_log_probs, _ = agent.q_network.forward(obs_batch, recurrent_hidden, masks_batch, attn_masks_batch)
        # attn_logprob = agent.q_network.attn_log_probs(attn_masks_batch[:-1,:])
        attn_logprob = attn_log_probs[:-1,:]
        loss_attn = attn_logprob * adv_targ.detach()
        value_loss = 0.5 * (return_batch - values[:-1,:]).pow(2).mean()

        losses_attn.append(loss_attn + 0.5*value_loss)

    total_loss = torch.stack(losses_attn).mean()
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss


def evaluate(agent, eval_envs_dic ,env_name, eval_locations_dic, num_processes, num_tasks, **kwargs):

    eval_envs = eval_envs_dic[env_name]
    locations = eval_locations_dic[env_name]
    eval_episode_rewards = []

    for iter in range(0, num_tasks, num_processes):
        for i in range(num_processes):
            eval_envs.set_task_id(task_id=iter+i, task_location=locations[i], indices=i)

        obs = eval_envs.reset()
        recurrent_hidden = torch.zeros(num_processes, agent.q_network.recurrent_hidden_state_size).type(dtype)
        masks = torch.zeros(num_processes, 1).type(dtype)
        attn_masks = torch.ones(num_processes, obs.size(1)).type(dtypelong)

        for t in range(kwargs["steps"]):
            with torch.no_grad():
                actions, value, recurrent_hidden, attn_log_probs, attn_masks = agent.act(obs, recurrent_hidden, masks=masks, attn_masks=attn_masks)

            # Observe reward and next obs
            obs, _, done, infos = eval_envs.step(actions.cpu())

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done]).type(dtype)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])


    return eval_episode_rewards


def get_epsilon(epsilon_start, epsilon_final, epsilon_decay, frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1.0 * frame_idx / epsilon_decay
    )


def soft_update(q_network, target_q_network, tau):
    for t_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        if t_param is param:
            continue
        new_param = tau * param.data + (1.0 - tau) * t_param.data
        t_param.data.copy_(new_param)


def hard_update(q_network, target_q_network):
    for t_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        if t_param is param:
            continue
        new_param = param.data
        t_param.data.copy_(new_param)


def main_dqn(params):
    EVAL_ENVS = {'train_eval': [params.env, params.num_processes],
                 'valid_eval': [params.val_env, params.num_processes],
                 'test_eval': ['h_bandit-obs-randchoose-v1', 100]}

    random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)

    device = "cpu"
    if USE_CUDA:
        device = "cuda"

    logdir = 'dual_offline_' +  params.env + '_' + str(params.seed) + '_num_arms_' + str(params.num_processes)+ '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    if params.rotate:
        logdir = logdir + '_rotate'

    logdir = os.path.join('dual_dqn_runs_offline', logdir)
    logdir = os.path.join(os.path.expanduser(params.log_dir), logdir)
    utils.cleanup_log_dir(logdir)
    summary_writer = SummaryWriter(log_dir=logdir)
    summary_writer.add_hparams(vars(params), {})

    print("logdir: " + logdir)
    for key in vars(params):
        print(key, ':', vars(params)[key])

    argslog = pd.DataFrame(columns=['args', 'value'])
    for key in vars(params):
        log = [key] + [vars(params)[key]]
        argslog.loc[len(argslog)] = log

    with open(logdir + '/args.csv', 'w') as f:
        argslog.to_csv(f, index=False)


    print('making envs...')
    eval_envs_dic = {}
    eval_locations_dic = {}
    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        eval_locations_dic[eval_disp_name] = np.random.randint(0, 6, size=params.num_processes)

    envs = make_vec_envs(params.env, params.seed, params.num_processes, eval_locations_dic['train_eval'],
                         params.gamma, None, device, False, steps=params.task_steps,
                         free_exploration=params.free_exploration, recurrent=params.recurrent_policy,
                         obs_recurrent=params.obs_recurrent, multi_task=True, normalize=not params.no_normalize, rotate=params.rotate)

    # Validation envs
    val_envs = make_vec_envs(params.val_env, params.seed, params.num_processes, eval_locations_dic['valid_eval'],
                             params.gamma, None, device, False, steps=params.task_steps,
                             free_exploration=params.free_exploration, recurrent=params.recurrent_policy,
                             obs_recurrent=params.obs_recurrent, multi_task=True, normalize=not params.no_normalize, rotate=params.rotate)

    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        eval_envs_dic[eval_disp_name] = make_vec_envs(eval_env_name[0], params.seed, params.num_processes, eval_locations_dic[eval_disp_name],
                                                      None, None, device, True, steps=params.task_steps,
                                                      recurrent=params.recurrent_policy,
                                                      obs_recurrent=params.obs_recurrent, multi_task=True,
                                                      free_exploration=params.free_exploration, normalize=not params.no_normalize, rotate=params.rotate)

    q_network = DQN_Attention_Wvalue(envs.observation_space.shape, envs.action_space.n, recurrent=True)
    target_q_network = deepcopy(q_network)

    q_network = q_network.to(device)
    target_q_network = target_q_network.to(device)

    agent = Agent(envs, q_network, target_q_network)

    attention_parameters = []
    non_attention_parameters = []
    for name, p in q_network.named_parameters():
        if ('attention' in name) or ('value' in name) :
            attention_parameters.append(p)
        else:
            non_attention_parameters.append(p)

    optimizer = optim.Adam(non_attention_parameters, lr=params.learning_rate)
    optimizer_val = optim.Adam(attention_parameters, lr=params.attn_learning_rate)
    replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)
    Attn_replay_buffer = AttnReplayBufferBandit(params.attn_steps, params.num_processes, envs.observation_space.shape, envs.action_space)

    # Load previous model
    if (params.continue_from_epoch > 0) and params.save_dir != "":
        save_path = params.save_dir
        q_network_weighs = torch.load(os.path.join(save_path, params.env + "-epoch-{}.pt".format(params.continue_from_epoch)), map_location=device)
        agent.q_network.load_state_dict(q_network_weighs['state_dict'])
        hard_update(agent.q_network, agent.target_q_network)
        optimizer.load_state_dict(q_network_weighs['optimizer_state_dict'])
        optimizer_val.load_state_dict(q_network_weighs['val_optimizer_state_dict'])


    obs = envs.reset()
    replay_buffer.obs[0].copy_(obs)
    replay_buffer.to(device)

    episode_rewards = deque(maxlen=25)
    episode_len = deque(maxlen=25)

    losses = []

    # Collect data
    # recurrent_hidden_states = torch.zeros(params.num_processes, agent.q_network.recurrent_hidden_state_size).type(dtypelong)
    for step in range(params.num_steps):

        # actions, recurrent_hidden_states = agent.act(replay_buffer.obs[step], recurrent_hidden_states, epsilon, replay_buffer.masks[step])
        actions = torch.tensor(np.random.randint(agent.num_actions, size=params.num_processes)).type(dtypelong).unsqueeze(-1)

        next_obs, reward, done, infos = envs.step(actions.cpu())


        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                episode_len.append(info['episode']['l'])

        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])

        replay_buffer.insert(next_obs, actions, reward, masks)

    # Training
    num_updates = int(
        params.max_ts  // params.num_processes // params.task_steps // params.mini_batch_size)
    val_obs = val_envs.reset()
    Attn_replay_buffer.obs[0].copy_(val_obs)
    Attn_replay_buffer.to(device)
    # mean_epoch = 0
    for ts in range(num_updates):
        # Update the q-network & the target network
        loss, mean_loss, var_loss = compute_td_loss(
            agent, params.num_mini_batch, params.mini_batch_size, replay_buffer, optimizer, params.gamma, params.loss_var_coeff
        )
        losses.append(loss.data)

        # Attention update
        recurrent_hidden_states = torch.zeros(params.num_processes, agent.q_network.recurrent_hidden_state_size).type(dtype)
        # masks = torch.zeros(params.num_processes, 1).type(dtype)
        # cumulative_reward = torch.zeros(params.num_processes, 1).type(dtype)
        # attn_masks = torch.ones(params.num_processes, obs.size(1)).type(dtypelong)

        for step in range(params.attn_steps):
            actions, value, recurrent_hidden_states, attn_log_probs, attn_masks = agent.act(Attn_replay_buffer.obs[step], recurrent_hidden_states, Attn_replay_buffer.masks[step], Attn_replay_buffer.attn_masks[step])

            next_val_obs, reward, done, infos = val_envs.step(actions.cpu())

            # for info in infos:
            #     if 'episode' in info.keys():
            #         episode_rewards.append(info['episode']['r'])
            #         episode_len.append(info['episode']['l'])

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done]).type(dtype)

            # val_obs = next_val_obs


            Attn_replay_buffer.insert(next_val_obs, actions, reward, attn_log_probs, value, masks, attn_masks)

        # with torch.no_grad():
        #     next_value = agent.get_value(Attn_replay_buffer.obs[-1], recurrent_hidden_states, Attn_replay_buffer.masks[-1], Attn_replay_buffer.attn_masks[-1]).detach()
        #     Attn_replay_buffer.compute_returns(next_value, use_gae=params.use_gae, gamma=params.gamma, gae_lambda=params.gae_lambda)

        # atten_loss, value_loss = Atten_update(agent, Attn_replay_buffer, optimizer_val)
        # atten_loss = Atten_update(agent, Attn_replay_buffer, optimizer_val)


        if ts % params.target_network_update_f == 0:
            hard_update(agent.q_network, agent.target_q_network)

        if (ts % params.save_interval == 0 or ts == params.continue_from_epoch + num_updates- 1):
            torch.save(
                {'state_dict': q_network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_optimizer_state_dict': optimizer_val.state_dict(),
                    'step': ts, 'obs_rms': getattr(utils.get_vec_normalize(envs), 'obs_rms', None)},
                os.path.join(logdir, params.env + "-epoch-{}.pt".format(ts)))

        if ts % params.log_every == 0:
            total_num_steps = (ts+1)*params.num_processes*params.task_steps*params.mini_batch_size
            out_str = "Iter {}, Timestep {}, Attention {} ".format(ts, total_num_steps, torch.sigmoid(agent.q_network.input_attention).data)
            if len(episode_rewards) > 1:
                # out_str += ", Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".format(len(episode_rewards), np.mean(episode_rewards),
                #         np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards))
                # summary_writer.add_scalar(f'eval/train episode reward', np.mean(episode_rewards), ts)
                # summary_writer.add_scalar(f'eval/train episode len', np.mean(episode_len), ts)

                eval_r = {}
                for eval_disp_name, eval_env_name in EVAL_ENVS.items():
                    eval_r[eval_disp_name] = evaluate(agent, eval_envs_dic, eval_disp_name, eval_locations_dic,
                                                      params.num_processes,
                                                      eval_env_name[1],
                                                      steps=params.task_steps,
                                                      recurrent=params.recurrent_policy, obs_recurrent=params.obs_recurrent,
                                                      multi_task=True, free_exploration=params.free_exploration)

                    summary_writer.add_scalar(f'eval/{eval_disp_name}', np.mean(eval_r[eval_disp_name]), total_num_steps)
            if len(losses) > 0:
                out_str += ", TD Loss: {}".format(losses[-1])
                summary_writer.add_scalar(f'losses/TD_loss', losses[-1], total_num_steps)
                summary_writer.add_scalar(f'losses/mean_loss', mean_loss, total_num_steps)
                summary_writer.add_scalar(f'losses/var_loss', var_loss, total_num_steps)
                # summary_writer.add_scalar(f'losses/attention_loss', atten_loss, total_num_steps)
                # summary_writer.add_scalar(f'losses/mean_reward',value_loss, total_num_steps)
            print(out_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument('--val-env', type=str, default=None)
    parser.add_argument("--num-processes", type=int, default=25, help='how many envs to use (default: 25)')
    parser.add_argument("--num-steps", type=int, default=5, help='number of forward steps in (default: 5)')
    parser.add_argument("--seed", type=int, default=1, help='random seed (default: 1)')
    parser.add_argument("--task_steps", type=int, default=20, help='number of steps in each task')
    parser.add_argument("--free_exploration", type=int, default=0, help='number of steps in each task without reward')
    parser.add_argument("--recurrent-policy", action='store_true', default=False, help='use a recurrent policy')
    parser.add_argument("--obs_recurrent", action='store_true', default=False, help='use a recurrent policy and observations input')
    parser.add_argument("--no_normalize", action='store_true', default=False, help='no normalize inputs')
    parser.add_argument("--rotate", action='store_true', default=False, help='rotate observations')
    parser.add_argument("--continue_from_epoch", type=int, default=0, help='load previous training (from model save dir) and continue')
    parser.add_argument("--log-dir", default='/tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument("--save-dir", default='./trained_models/', help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument("--num-mini-batch", type=int, default=32, help='number of mini-batches (default: 32)')
    parser.add_argument("--mini-batch-size", type=int, default=32, help='size of mini-batches (default: 32)')
    # parser.add_argument("--CnnDQN", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--attn_learning_rate", type=float, default=0.00001)
    # parser.add_argument("--target_update_rate", type=float, default=0.1)
    parser.add_argument("--attn-steps", type=int, default=100)
    parser.add_argument("--save-interval",type=int,default=1000, help='save interval, one save per n updates (default: 1000)')
    parser.add_argument("--max_ts", type=int, default=1400000)
    # parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--use_gae", action='store_true', default=False, help='rotate observations')
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--target_network_update_f", type=int, default=10000)
    parser.add_argument("--loss_var_coeff", type=float, default=0.0)
    main_dqn(parser.parse_args())
