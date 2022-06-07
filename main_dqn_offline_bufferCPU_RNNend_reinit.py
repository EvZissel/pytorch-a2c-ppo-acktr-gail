#!/usr/bin/env python3

"""
Usage:

$ . ~/env/bin/activate

Example pong command (~900k ts solve):
    python main_dqn_offline.py \
        --env "PongNoFrameskip-v4" --CnnDQN --learning_rate 0.00001 \
        --target_update_rate 0.1 --replay_size 100000 --start_train_ts 10000 \
        --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 30000 --max_ts 1400000 \
        --batch_size 32 --gamma 0.99 --log_every 10000
 """

import argparse
import math
import random
from copy import deepcopy
import os
import time
from collections import deque

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from helpers_dqn import ReplayBuffer, make_atari, make_gym_env, wrap_deepmind, wrap_pytorch, ReplayBufferBandit
from models_dqn import DQN, CnnDQN, DQN_softAttn, DQN_softAttn_L2grad, DQN_RNNLast
from a2c_ppo_acktr.envs import make_vec_envs
from torch.utils.tensorboard import SummaryWriter
from a2c_ppo_acktr import utils
import pandas as pd
import torch.nn as nn
from grad_tools.grad_coherence import GradCoherenceDqn
import itertools
from collections import OrderedDict
import wandb

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


def _flatten_grad(grads):
    flatten_grad = torch.cat([g.flatten() for g in grads])
    return flatten_grad

class Agent:
    def __init__(self, env, q_network, target_q_network):
        self.env = env
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.num_actions = env.action_space.n

    def act(self, state, hidden_state, epsilon, masks):
        """DQN action - max q-value w/ epsilon greedy exploration."""
        # state = torch.tensor(np.float32(state)).type(dtype).unsqueeze(0)

        q_value, _, rnn_hxs = self.q_network.forward(state, hidden_state, masks)
        if random.random() > epsilon:
            return q_value.max(1)[1].unsqueeze(-1), rnn_hxs
        return torch.tensor(np.random.randint(self.env.action_space.n, size=q_value.size()[0])).type(dtypelong).unsqueeze(-1), rnn_hxs


def compute_td_loss(agent, num_mini_batch, mini_batch_size, replay_buffer, optimizer, gamma, loss_var_coeff, device = "CPU", train=True, same_ind=False, start_ind_array=None):
    num_processes = replay_buffer.rewards.size(1)
    num_steps = replay_buffer.rewards.size(0)
    num_steps_per_batch = int(num_steps/num_mini_batch)


    if not same_ind:
        start_ind_array = [i for i in range(0, num_steps, num_steps_per_batch)]
        start_ind_array = np.random.choice(start_ind_array, size=mini_batch_size, replace=False)

    all_losses = []
    # all_grad_W2 = []
    # all_grad_b2 = []
    grad_L2_states_all = 0
    out1_states_all = 0
    # grad_L2_states_mean_all = 0
    # grad_L2_states_sum_all = 0
    # grad_L2_states_sum_squre_all = 0
    # states_all = 0
    for i in range(num_processes):
        all_losses.append(0)
        # all_grad_W2.append(0)
        # all_grad_b2.append(0)

    recurrent_hidden = torch.zeros(1, agent.q_network.recurrent_hidden_state_size).type(dtypelong)
    for start_ind in start_ind_array:
        data_sampler = replay_buffer.sampler(num_processes, start_ind, num_steps_per_batch)

        losses = []
        # grad_W2 = []
        # grad_b2 = []
        grad_L2_states = 0
        out1_states = 0
        for states, actions, rewards, done in data_sampler:
            # double q-learning
            with torch.no_grad():
                    # recurrent_hidden.detach()
                online_q_values, _, _ = agent.q_network(states.to(device), recurrent_hidden, done.to(device))
                _, max_indicies = torch.max(online_q_values, dim=1)
                target_q_values, _, _ = agent.target_q_network(states.to(device), recurrent_hidden, done.to(device))
                next_q_value = target_q_values.gather(1, max_indicies.unsqueeze(1))

                next_q_value = next_q_value * done.to(device)
                expected_q_value = (rewards.to(device) + gamma * next_q_value[1:, :]).squeeze(1)

            # Normal DDQN update
            q_values, out_1, _ = agent.q_network(states.to(device), recurrent_hidden, done.to(device))
            q_value = q_values[:-1, :].gather(1, actions.to(device)).squeeze(1)
            out_1 = out_1[:-1, :].unsqueeze(1)

            td_err = (q_value - expected_q_value.data).unsqueeze(1).unsqueeze(1)
            e = torch.zeros(num_steps_per_batch, actions.size(0), 1, device=device)
            e[torch.arange(e.size(0)).unsqueeze(1), actions.to(device)] = 1.
            # grad_b2 = td_err*e
            # e_actions = torch.sparse_coo_tensor(torch.cat((actions, torch.tensor([0, 1, 2, 3, 4, 5], device=actions.device).unsqueeze(1)), dim=1).t(),(q_value - expected_q_value.data), (6, 6)).to_dense().unsqueeze(2)
            # batch_grad_b2 = e_actions.mean(1)
            # e_actions = e_actions.t().unsqueeze(2)
            # out_1 = out_1.unsqueeze(1)
            # batch_grad_W2 = torch.matmul(e_actions, out_1).mean(0)
            # grad_w2 = (e*out_1)
            grad_L2_states_b = td_err*torch.cat(((e*out_1), e), dim=2)
            grad_L2_states_b = torch.flatten(grad_L2_states_b, start_dim=1).unsqueeze(1)

            one_loss = 0.5*(q_value - expected_q_value.data).pow(2).mean()
            losses.append(one_loss)
            # grad_W2.append(batch_grad_W2)
            # grad_b2.append(batch_grad_b2)
            # states_all_b += torch.sigmoid(agent.q_network.input_attention) *states
            if (len(losses)==1):
                grad_L2_states = grad_L2_states_b
                out1_states = out_1
                # states_all = torch.sigmoid(agent.q_network.input_attention) *states
            else:
                grad_L2_states = torch.cat((grad_L2_states, grad_L2_states_b), dim=1)
                out1_states = torch.cat((out1_states, out_1), dim=1)
                # states_all = torch.cat((states_all, torch.sigmoid(agent.q_network.input_attention) *states), dim=0)

            # grads = torch.autograd.grad(one_loss,
            #                             updated_train_params.values(),
            #                             create_graph=create_graph)

        # loss = torch.stack(losses)
        # loss = losses.mean(0)
        # all_losses.append(loss)
        for i in range(num_processes):
            all_losses[i] += losses[i]/mini_batch_size
            # all_grad_W2[i] += grad_W2[i]/mini_batch_size
            # all_grad_b2[i] += grad_b2[i]/mini_batch_size

        # grad_L2_states_sum = grad_L2_states.sum(1)
        # grad_L2_states_mean = grad_L2_states.mean(1)
        # grad_L2_states_sum_squre = (grad_L2_states**2).sum(1)
        if start_ind == start_ind_array[0] :
            grad_L2_states_all = grad_L2_states
            out1_states_all = out1_states
            # grad_L2_states_mean_all = grad_L2_states_mean
            # grad_L2_states_sum_all = grad_L2_states_sum
            # grad_L2_states_sum_squre_all = grad_L2_states_sum_squre
            # states_all = states_all_b
        else:
            grad_L2_states_all = torch.cat((grad_L2_states_all, grad_L2_states), dim=0)
            out1_states_all = torch.cat((out1_states_all, out1_states), dim=0)
            # grad_L2_states_mean_all = torch.cat((grad_L2_states_mean_all, grad_L2_states_mean), dim=0)
            # grad_L2_states_sum_all = torch.cat((grad_L2_states_sum_all, grad_L2_states_sum), dim=0)
            # grad_L2_states_sum_squre_all = torch.cat((grad_L2_states_sum_squre_all, grad_L2_states_sum_squre), dim=0)
            # states_all = torch.cat((states_all, states_all_b), dim=0)


    total_loss = torch.stack(all_losses)
    # total_grad_W2 = torch.stack(all_grad_W2).mean(0)
    # total_grad_b2 = torch.stack(all_grad_b2).mean(0)

    # total_grad_L2 = _flatten_grad((total_grad_W2, total_grad_b2))
    # total_grad_L2 = []
    # for i in range(num_processes):
    #     total_grad_L2.append(_flatten_grad(torch.cat((all_grad_W2[i],torch.unsqueeze(all_grad_b2[i],1)), dim=1)))
    # total_grad_L2_state = []
    # for i in range(grad_L2_states_all.shape[0]):
    #     total_grad_L2_state.append(_flatten_grad(grad_L2_states_all[i]))

    total_loss = total_loss.mean() + loss_var_coeff * total_loss.var()
    # optimizer.zero_grad()
    # total_loss.backward()
    # optimizer.step()

    # optimizer.zero_grad()
    # grads = torch.autograd.grad(total_loss,
    #                             updated_train_params.values(),
    #                             create_graph=create_graph)

    optimizer.zero_grad()
    # mean_flat_grad, grads, shapes, coherence = optimizer.plot_backward(all_losses)
    total_loss.backward()

    if train:
        optimizer.step()
    # grads = None
    # else:
    #     grads = torch.autograd.grad(total_loss,
    #                                 updated_train_params.values(),
    #                                 create_graph=create_graph)
    return total_loss.detach(), grad_L2_states_all.detach(), out1_states_all.detach(), start_ind_array


def evaluate(agent, eval_envs_dic ,env_name, eval_locations_dic, num_processes, num_tasks, **kwargs):

    eval_envs = eval_envs_dic[env_name]
    locations = eval_locations_dic[env_name]
    eval_episode_rewards = []

    for iter in range(0, num_tasks, num_processes):
        for i in range(num_processes):
            eval_envs.set_task_id(task_id=iter+i, task_location=locations[i], indices=i)

        obs = eval_envs.reset()
        recurrent_hidden = torch.zeros(num_processes, agent.q_network.recurrent_hidden_state_size).type(dtype)
        masks = torch.ones(num_processes, 1).type(dtype)

        for t in range(kwargs["steps"]):
            with torch.no_grad():
                actions, recurrent_hidden = agent.act(obs, recurrent_hidden, epsilon=-1, masks=masks)

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
                 'test_eval' : ['h_bandit-obs-randchoose-v1', 100]}

    random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)

    device = "cpu"
    if USE_CUDA:
        device = "cuda"

    logdir_ = 'offline_Bcpu' +  params.env + '_' + str(params.seed) + '_num_arms_' + str(params.num_processes) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    if params.rotate:
        logdir_ = logdir_ + '_rotate'
    if params.zero_ind:
        logdir_ = logdir_ + '_Zero_ind'
    if params.reinitialization_last:
        logdir_ = logdir_ + '_reinitLast'
    if params.reinitialization_last_GRU:
        logdir_ = logdir_ + '_reinitLastGRU'

    logdir = os.path.join('dqn_runs_offline', logdir_)
    logdir = os.path.join(os.path.expanduser(params.log_dir), logdir)
    utils.cleanup_log_dir(logdir)
    summary_writer = SummaryWriter(log_dir=logdir)
    summary_writer.add_hparams(vars(params), {})

    wandb.init(project="main_dqn_offline_bufferCPU_RNNend_reinit", entity="ev_zisselman",config=params,name=logdir_,id=logdir_)

    print("logdir: " + logdir)
    for key in vars(params):
        print(key, ':', vars(params)[key])

    argslog = pd.DataFrame(columns=['args', 'value'])
    for key in vars(params):
        log = [key] + [vars(params)[key]]
        argslog.loc[len(argslog)] = log

    with open(logdir + '/args.csv', 'w') as f:
        argslog.to_csv(f, index=False)

    logdir_grad = os.path.join(logdir, 'grads')
    utils.cleanup_log_dir(logdir_grad)

    print('making envs...')
    eval_envs_dic = {}
    eval_locations_dic = {}
    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        eval_locations_dic[eval_disp_name] = np.random.randint(0, 6, size=params.num_processes)

    envs = make_vec_envs(params.env, params.seed, params.num_processes, eval_locations_dic['train_eval'],
                         params.gamma, None, device, False, steps=params.task_steps,
                         free_exploration=params.free_exploration, recurrent=params.recurrent_policy,
                         obs_recurrent=params.obs_recurrent, multi_task=True, normalize=not params.no_normalize, rotate=params.rotate)

    # val_envs = make_vec_envs(params.val_env, params.seed, params.num_processes, eval_locations_dic['valid_eval'],
    #                      params.gamma, None, device, False, steps=params.task_steps,
    #                      free_exploration=params.free_exploration, recurrent=params.recurrent_policy,
    #                      obs_recurrent=params.obs_recurrent, multi_task=True, normalize=not params.no_normalize, rotate=params.rotate)

    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        eval_envs_dic[eval_disp_name] = make_vec_envs(eval_env_name[0], params.seed, params.num_processes, eval_locations_dic[eval_disp_name],
                                                      None, None, device, True, steps=params.task_steps,
                                                      recurrent=params.recurrent_policy,
                                                      obs_recurrent=params.obs_recurrent, multi_task=True,
                                                      free_exploration=params.free_exploration, normalize=not params.no_normalize, rotate=params.rotate)

    q_network = DQN_RNNLast(envs.observation_space.shape, envs.action_space.n, params.zero_ind, recurrent=True, hidden_size=params.hidden_size)
    target_q_network = deepcopy(q_network)
    if params.target_hard:
        target_q_network.target = True

    q_network = q_network.to(device)
    target_q_network = target_q_network.to(device)

    agent = Agent(envs, q_network, target_q_network)
    attention_parameters = []
    non_attention_parameters = []
    for name, p in q_network.named_parameters():
        if 'attention' in name:
            attention_parameters.append(p)
        else:
            non_attention_parameters.append(p)

    optimizer = optim.Adam(non_attention_parameters, lr=params.learning_rate, weight_decay=params.weight_decay)
    # optimizer = GradCoherenceDqn(optimizer)
    # optimizer_val = optim.Adam(attention_parameters, lr=params.learning_rate_val, weight_decay=params.weight_decay)
    # scheduler = MultiStepLR(optimizer_val, milestones=[30, 80], gamma=0.1)

    replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)
    # grad_replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)
    # val_replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)

    # Load previous model
    if (params.continue_from_epoch > 0) and params.save_dir != "":
        save_path = params.save_dir
        q_network_weighs = torch.load(os.path.join(save_path, params.env + "-epoch-{}.pt".format(params.continue_from_epoch)), map_location=device)
        agent.q_network.load_state_dict(q_network_weighs['state_dict'])
        agent.target_q_network.load_state_dict(q_network_weighs['target_state_dict'])
        # hard_update(agent.q_network, agent.target_q_network)
        optimizer.load_state_dict(q_network_weighs['optimizer_state_dict'])


    obs = envs.reset()
    replay_buffer.obs[0].copy_(obs)
    # replay_buffer.to(device)

    # val_obs = val_envs.reset()
    # val_replay_buffer.obs[0].copy_(val_obs)
    # # val_replay_buffer.to(device)

    episode_rewards = deque(maxlen=25)
    episode_len = deque(maxlen=25)
    # val_episode_rewards = deque(maxlen=25)
    # val_episode_len = deque(maxlen=25)

    losses = []
    # # grad_losses = []
    # val_losses = []

    # Collect train data
    # recurrent_hidden_states = torch.zeros(params.num_processes, agent.q_network.recurrent_hidden_state_size).type(dtypelong)
    for step in range(params.num_steps):

        # actions, recurrent_hidden_states = agent.act(replay_buffer.obs[step], recurrent_hidden_states, epsilon, replay_buffer.masks[step])
        # actions = torch.tensor(np.random.randint(agent.num_actions, size=params.num_processes)).type(dtypelong).unsqueeze(-1)
        actions = torch.tensor(np.random.randint(agent.num_actions) * np.ones(params.num_processes)).type(dtypelong).unsqueeze(-1).cpu()

        next_obs, reward, done, infos = envs.step(actions.cpu())


        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                episode_len.append(info['episode']['l'])

        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])

        replay_buffer.insert(next_obs, actions, reward, masks)

    # # Collect validation data
    # for step in range(params.num_steps):
    #
    #     # val_actions = torch.tensor(np.random.randint(agent.num_actions, size=params.num_processes)).type(dtypelong).unsqueeze(-1)
    #     val_actions = torch.tensor(np.random.randint(agent.num_actions) * np.ones(params.num_processes)).type(dtypelong).unsqueeze(-1).cpu()
    #
    #     val_next_obs, val_reward, val_done, val_infos = val_envs.step(val_actions.cpu())
    #
    #
    #     for info in val_infos:
    #         if 'episode' in info.keys():
    #             val_episode_rewards.append(info['episode']['r'])
    #             val_episode_len.append(info['episode']['l'])
    #
    #     val_masks = torch.FloatTensor(
    #         [[0.0] if val_done_ else [1.0] for val_done_ in val_done])
    #
    #     val_replay_buffer.insert(val_next_obs, val_actions, val_reward, val_masks)

    # val_replay_buffer.copy(replay_buffer)
    # # replay_buffer.obs[:,:,:-2] = obs[:,:-2].repeat([replay_buffer.obs.size(0),1,1])
    # val_replay_buffer.obs[:,:,:-2] = val_obs[:,:-2].repeat([replay_buffer.obs.size(0),1,1])

    # Training
    num_updates = int(
        params.max_ts  // params.num_processes // params.task_steps // params.mini_batch_size)

    min_corr = torch.tensor(0.9)
    epsilon = 1e-6

    # alpha = params.loss_corr_coeff_update
    # loss_corr_coeff = params.loss_corr_coeff
    # # last_Corr = 0
    # grads_sum10 = []
    # grads_sum100 = []
    # val_grads_sum10 = []
    # val_grads_sum100 = []
    # mean_grads_sum10 = deque(maxlen=10)
    # mean_grads_sum100 = deque(maxlen=100)
    # val_mean_grads_sum10 = deque(maxlen=10)
    # val_mean_grads_sum100 = deque(maxlen=100)
    # for i in range(params.num_processes):
    #     grads_sum10.append(deque(maxlen=10))
    #     grads_sum100.append(deque(maxlen=100))
    #     val_grads_sum10.append(deque(maxlen=10))
    #     val_grads_sum100.append(deque(maxlen=100))

    # # compute validation gradient for start_ind_array
    # loss, grads_L2, out1, mean_flat_grad, grads, shapes, coherence, start_ind_array = compute_td_loss(
    #     agent, params.num_mini_batch, params.mini_batch_size_val, replay_buffer, optimizer, params.gamma, params.loss_var_coeff, device, train=False,
    # )

    for ts in range(params.continue_from_epoch, params.continue_from_epoch+num_updates):
        # Update the q-network & the target network

        #### Update Theta #####
        loss, grads_L2, out1, _ = compute_td_loss(
            agent, params.num_mini_batch, params.mini_batch_size, replay_buffer, optimizer, params.gamma, params.loss_var_coeff, device, train=True,
        )
        losses.append(loss.data)

        # # compute validation gradient
        # loss, grads_L2, out1, mean_flat_grad, grads, shapes, coherence, _ = compute_td_loss(
        #     agent, params.num_mini_batch, params.mini_batch_size_val, replay_buffer, optimizer, params.gamma, params.loss_var_coeff, device, train=False, same_ind=True, start_ind_array=start_ind_array
        # )
        # # losses.append(loss.data)
        #
        # val_loss, val_grads_L2, val_out1, val_mean_flat_grad, val_grads, val_shapes, val_coherence, _ = compute_td_loss(
        #     agent, params.num_mini_batch, params.mini_batch_size_val, val_replay_buffer, optimizer, params.gamma, params.loss_var_coeff, device, train=False, same_ind=True, start_ind_array=start_ind_array
        # )
        # val_losses.append(val_loss.data)

        # # compute validation gradient batch 2
        # loss_b2, grads_L2_b2, out1_b2, mean_flat_grad_b2, grads_b2, shapes_b2, coherence_b2, start_ind_array_b2 = compute_td_loss(
        #     agent, params.num_mini_batch, params.mini_batch_size_val2, replay_buffer, optimizer, params.gamma, params.loss_var_coeff, train=False,
        # )
        # losses.append(loss.data)
        #
        # val_loss_b2, val_grads_L2_b2, val_out1_b2, val_mean_flat_grad_b2, val_grads_b2, val_shapes_b2, val_coherence_b2, _ = compute_td_loss(
        #     agent, params.num_mini_batch, params.mini_batch_size_val2, val_replay_buffer, optimizer, params.gamma, params.loss_var_coeff, train=False,
        # )
        # val_losses.append(val_loss.data)


        # grads_L2 = grads_L2.resize(grads_L2.size(0) * grads_L2.size(1), grads_L2.size(2))
        # val_grads_L2 = val_grads_L2.resize(val_grads_L2.size(0) * val_grads_L2.size(1), val_grads_L2.size(2))
        # grads_L2 = torch.flatten(grads_L2, start_dim=0, end_dim=1)
        # val_grads_L2 = torch.flatten(val_grads_L2, start_dim=0, end_dim=1)

        # out1 = out1.resize(out1.size(0) * out1.size(1), out1.size(2))
        # val_out1 = val_out1.resize(val_out1.size(0) * val_out1.size(1), val_out1.size(2))
        # out1 = torch.flatten(out1, start_dim=0, end_dim=1)
        # val_out1 = torch.flatten(val_out1, start_dim=0, end_dim=1)

        # train_grads_L2_Corr = grads_L2.mean(0)
        # val_grads_L2_Corr = val_grads_L2.mean(0)
        # Corr_dqn_L2_grad_mean = (train_grads_L2_Corr * val_grads_L2_Corr).sum() / (train_grads_L2_Corr.norm(2) * val_grads_L2_Corr.norm(2))
        # print("correlation mean L2: {}".format(Corr_dqn_L2_grad_mean))
        # print("correlation mean denominator  L2: {}".format(train_grads_L2_Corr.norm(2) * val_grads_L2_Corr.norm(2)))
        # mean_grad = optimizer._unflatten_grad(mean_flat_grad, shapes)

        # train_out1_Corr = out1.mean(0)
        # val_out1_Corr = val_out1.mean(0)
        # Corr_dqn_out1_mean = (train_out1_Corr * val_out1_Corr).sum() / (train_out1_Corr.norm(2) * val_out1_Corr.norm(2))

        # mean_grads_sum10.append(mean_flat_grad)
        # mean_grads_sum100.append(mean_flat_grad)
        # val_mean_grads_sum10.append(val_mean_flat_grad)
        # val_mean_grads_sum100.append(val_mean_flat_grad)
        # for i in range(params.num_processes):
        #     grads_sum10[i].append(grads[i])
        #     grads_sum100[i].append(grads[i])
        #     val_grads_sum10[i].append(val_grads[i])
        #     val_grads_sum100[i].append(val_grads[i])

        # if ts==0 or (train_grads_L2_Corr.norm(2) * val_grads_L2_Corr.norm(2) > epsilon):
        #     last_Corr = Corr_dqn_L2_grad_mean
        # print("last correlation mean: {}".format(last_Corr))
        # trajectory_len = grads_L2.size(0)/params.mini_batch_size
        # grads_L2_mean_trajectory = grads_L2[0:int(trajectory_len),:,:].mean(0).unsqueeze(0)
        # grads_L2_sum_mean_trajectory = grads_sum_L2[0:int(trajectory_len),:].mean(0).unsqueeze(0)
        # grads_L2_sum_squre_mean_trajectory = grads_sum_squre_L2[0:int(trajectory_len),:].mean(0).unsqueeze(0)
        # val_grads_L2_mean_trajectory = val_grads_L2[0:int(trajectory_len),:,:].mean(0).unsqueeze(0)
        # val_grads_L2_sum_mean_trajectory = val_grads_sum_L2[0:int(trajectory_len),:].mean(0).unsqueeze(0)
        # val_grads_L2_sum_squre_mean_trajectory = val_grads_sum_squre_L2[0:int(trajectory_len),:].mean(0).unsqueeze(0)
        # for i in range(1,params.mini_batch_size):
        #     grads_L2_mean_trajectory = torch.cat((grads_L2_mean_trajectory, grads_L2[int(i*trajectory_len):int((i+1)*trajectory_len),:,:].mean(0).unsqueeze(0)), dim=0)
        #     grads_L2_sum_mean_trajectory = torch.cat((grads_L2_sum_mean_trajectory, grads_sum_L2[int(i*trajectory_len):int((i+1)*trajectory_len),:].mean(0).unsqueeze(0)), dim=0)
        #     grads_L2_sum_squre_mean_trajectory = torch.cat((grads_L2_sum_squre_mean_trajectory, grads_sum_squre_L2[int(i*trajectory_len):int((i+1)*trajectory_len),:].mean(0).unsqueeze(0)), dim=0)
        #     val_grads_L2_mean_trajectory = torch.cat((val_grads_L2_mean_trajectory, val_grads_L2[int(i*trajectory_len):int((i+1)*trajectory_len),:,:].mean(0).unsqueeze(0)), dim=0)
        #     val_grads_L2_sum_mean_trajectory = torch.cat((val_grads_L2_sum_mean_trajectory, val_grads_sum_L2[int(i*trajectory_len):int((i+1)*trajectory_len),:].mean(0).unsqueeze(0)), dim=0)
        #     val_grads_L2_sum_squre_mean_trajectory = torch.cat((val_grads_L2_sum_squre_mean_trajectory, val_grads_sum_squre_L2[int(i*trajectory_len):int((i+1)*trajectory_len),:].mean(0).unsqueeze(0)), dim=0)

        # for i in range(states_all.size(0)):
        #     Corr_dqn_state += ((states_all[i] * val_states_all[i]).sum() / (
        #             states_all[i].norm(2) * val_states_all[i].norm(2))) / states_all.size(0)

        # grads_L2_mean_trajectory = grads_L2_mean_trajectory.resize(params.mini_batch_size*params.num_processes,grads_L2_mean_trajectory.size(2))

        # grads_L2_sum = grads_L2.sum(0)
        # grads_L2_sum_squre = (grads_L2**2).sum()
        #
        # Coherence_train = ((grads_L2 * (grads_L2_sum.unsqueeze(0))).sum())/(grads_L2_sum_squre + epsilon)
        # print("Coherence train: {}".format(Coherence_train))
        # print("Coherence train denominator: {}".format(grads_L2_sum_squre))
        #
        # val_grads_L2_sum = val_grads_L2.sum(0)
        # val_grads_L2_sum_squre = (val_grads_L2**2).sum()
        #
        # Coherence_val = ((val_grads_L2 * (val_grads_L2_sum.unsqueeze(0))).sum()) / (val_grads_L2_sum_squre + epsilon)
        # print("Coherence val: {}".format(Coherence_val))
        # print("Coherence val denominator: {}".format(val_grads_L2_sum_squre))
        # val_mean_grad = optimizer._unflatten_grad(val_mean_flat_grad, shapes)
        #
        # mean_grad_sum10_sum = optimizer._unflatten_grad(sum(mean_grads_sum10), shapes)
        # mean_grad_sum100_sum = optimizer._unflatten_grad(sum(mean_grads_sum100), shapes)
        #
        # val_mean_grad_sum10_sum = optimizer._unflatten_grad(sum(val_mean_grads_sum10), val_shapes)
        # val_mean_grad_sum100_sum = optimizer._unflatten_grad(sum(val_mean_grads_sum100), val_shapes)
        #
        # cosine_layers = []
        # cosine_layers_sum10 = []
        # cosine_layers_sum100 = []
        # for i in range(len(mean_grad)):
        #     mean_grad_i = optimizer._flatten_grad(mean_grad[i])
        #     val_mean_grad_i = optimizer._flatten_grad(val_mean_grad[i])
        #     cosine_layers.append((mean_grad_i * val_mean_grad_i).sum() / (mean_grad_i.norm(2) * val_mean_grad_i.norm(2)))
        #
        #     mean_grad_i_10 = optimizer._flatten_grad(mean_grad_sum10_sum[i])
        #     val_mean_grad_i_10 = optimizer._flatten_grad(val_mean_grad_sum10_sum[i])
        #     cosine_layers_sum10.append((mean_grad_i_10 * val_mean_grad_i_10).sum() / (mean_grad_i_10.norm(2) * val_mean_grad_i_10.norm(2)))
        #
        #     mean_grad_i_100 = optimizer._flatten_grad(mean_grad_sum100_sum[i])
        #     val_mean_grad_i_100 = optimizer._flatten_grad(val_mean_grad_sum100_sum[i])
        #     cosine_layers_sum100.append((mean_grad_i_100 * val_mean_grad_i_100).sum() / (mean_grad_i_100.norm(2) * val_mean_grad_i_100.norm(2)))
        #
        # coherence_sum10 = []
        # coherence_sum100 = []
        # val_coherence_sum10 = []
        # val_coherence_sum100 = []
        # g10_sum = []
        # g100_sum = []
        # g10_sum_squared = []
        # g100_sum_squared = []
        # val_g10_sum = []
        # val_g100_sum = []
        # val_g10_sum_squared = []
        # val_g100_sum_squared = []
        # for i in range(len(mean_grad)):
        #     g10_sum.append(0)
        #     g100_sum.append(0)
        #     g10_sum_squared.append(0)
        #     g100_sum_squared.append(0)
        #     coherence_sum10.append(0)
        #     coherence_sum100.append(0)
        #
        #     val_g10_sum.append(0)
        #     val_g100_sum.append(0)
        #     val_g10_sum_squared.append(0)
        #     val_g100_sum_squared.append(0)
        #     val_coherence_sum10.append(0)
        #     val_coherence_sum100.append(0)
        # for i in range(len(grads_sum10)):
        #     g_10 = optimizer._unflatten_grad(sum(grads_sum10[i]), shapes)
        #     g_100 = optimizer._unflatten_grad(sum(grads_sum100[i]), shapes)
        #     val_g_10 = optimizer._unflatten_grad(sum(val_grads_sum10[i]), val_shapes)
        #     val_g_100 = optimizer._unflatten_grad(sum(val_grads_sum100[i]), val_shapes)
        #
        #
        #     for i in range(len(g_10)):
        #         g10_sum[i] += g_10[i]
        #         g100_sum[i] += g_100[i]
        #         g10_sum_squared[i] += (g_10[i]**2).sum()
        #         g100_sum_squared[i] += (g_100[i]**2).sum()
        #
        #         val_g10_sum[i] += val_g_10[i]
        #         val_g100_sum[i] += val_g_100[i]
        #         val_g10_sum_squared[i] += (val_g_10[i]**2).sum()
        #         val_g100_sum_squared[i] += (val_g_100[i]**2).sum()
        #
        # for i in range(len(grads_sum10)):
        #     g_10 = optimizer._unflatten_grad(sum(grads_sum10[i]), shapes)
        #     g_100 = optimizer._unflatten_grad(sum(grads_sum100[i]), shapes)
        #     val_g_10 = optimizer._unflatten_grad(sum(val_grads_sum10[i]), val_shapes)
        #     val_g_100 = optimizer._unflatten_grad(sum(val_grads_sum100[i]), val_shapes)
        #
        #     for i in range(len(g_10)):
        #         coherence_sum10[i] += ((g_10[i]*g10_sum[i]).sum())/g10_sum_squared[i]
        #         coherence_sum100[i] += ((g_100[i]*g10_sum[i]).sum())/g100_sum_squared[i]
        #
        #         val_coherence_sum10[i] += ((val_g_10[i]*val_g10_sum[i]).sum())/val_g10_sum_squared[i]
        #         val_coherence_sum100[i] += ((val_g_100[i]*val_g10_sum[i]).sum())/val_g100_sum_squared[i]

        # ##### batch 2 #####
        # grads_L2_b2 = grads_L2_b2.resize(grads_L2_b2.size(0) * grads_L2_b2.size(1), grads_L2_b2.size(2))
        # val_grads_L2_b2 = val_grads_L2_b2.resize(val_grads_L2_b2.size(0) * val_grads_L2_b2.size(1), val_grads_L2_b2.size(2))
        #
        # out1_b2 = out1_b2.resize(out1_b2.size(0) * out1_b2.size(1), out1_b2.size(2))
        # val_out1_b2 = val_out1_b2.resize(val_out1_b2.size(0) * val_out1_b2.size(1), val_out1_b2.size(2))
        #
        # train_grads_L2_Corr_b2 = grads_L2_b2.mean(0)
        # val_grads_L2_Corr_b2 = val_grads_L2_b2.mean(0)
        # Corr_dqn_L2_grad_mean_b2 = (train_grads_L2_Corr_b2 * val_grads_L2_Corr_b2).sum() / (train_grads_L2_Corr_b2.norm(2) * val_grads_L2_Corr_b2.norm(2))
        # mean_grad_b2 = optimizer._unflatten_grad(mean_flat_grad_b2, shapes_b2)
        #
        # train_out1_Corr_b2 = out1_b2.mean(0)
        # val_out1_Corr_b2 = val_out1_b2.mean(0)
        # Corr_dqn_out1_mean_b2 = (train_out1_Corr_b2 * val_out1_Corr_b2).sum() / (train_out1_Corr_b2.norm(2) * val_out1_Corr_b2.norm(2))
        #
        # val_mean_grad_b2 = optimizer._unflatten_grad(val_mean_flat_grad_b2, shapes)
        # cosine_layers_b2 = []
        # for i in range(len(mean_grad_b2)):
        #     mean_grad_i_b2 = optimizer._flatten_grad(mean_grad_b2[i])
        #     val_mean_grad_i_b2 = optimizer._flatten_grad(val_mean_grad_b2[i])
        #     cosine_layers_b2.append((mean_grad_i_b2 * val_mean_grad_i_b2).sum() / (mean_grad_i_b2.norm(2) * val_mean_grad_i_b2.norm(2)))


        total_num_steps = (ts + 1) * params.num_processes * params.task_steps * params.mini_batch_size

        if ts % params.target_network_update_f == 0:
            hard_update(agent.q_network, agent.target_q_network)

        if (ts % params.save_interval == 0 or ts == params.continue_from_epoch + num_updates- 1):
            torch.save(
                {'state_dict': q_network.state_dict(),'target_state_dict': target_q_network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'step': ts, 'obs_rms': getattr(utils.get_vec_normalize(envs), 'obs_rms', None)},
                os.path.join(logdir, params.env + "-epoch-{}.pt".format(ts)))

        if ts % params.log_every == 0:
            out_str = "Iter {}, Timestep {}, attention {}".format(ts, total_num_steps, torch.sigmoid(q_network.input_attention).data)
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
                    wandb.log({f'eval/{eval_disp_name}': np.mean(eval_r[eval_disp_name])}, step=total_num_steps)
            if len(losses) > 0:
                out_str += ", TD Loss: {}".format(losses[-1])
                summary_writer.add_scalar(f'losses/TD_loss', losses[-1], total_num_steps)
                wandb.log({f'losses/TD_loss': losses[-1]}, step=total_num_steps)
                # summary_writer.add_scalar(f'losses/val_TD_loss', val_losses[-1], total_num_steps)

            print(out_str)

            # summary_writer.add_scalar(f'Correlation_grad_vec/Corr_dqn_L2_grad', Corr_dqn_L2_grad_mean, total_num_steps)
            # summary_writer.add_scalar(f'Correlation_grad_vec/Last_Corr_dqn_out1_grad', Corr_dqn_out1_mean, total_num_steps)
            # summary_writer.add_scalar(f'Correlation_grad_vec/Coherence_train_L2_grad', Coherence_train, total_num_steps)
            # summary_writer.add_scalar(f'Correlation_grad_vec/Coherence_val_L2_grad', Coherence_val, total_num_steps)

            # summary_writer.add_scalar(f'Correlation_grad_vec/Corr_dqn_L2_grad_b2', Corr_dqn_L2_grad_mean_b2, total_num_steps)
            # summary_writer.add_scalar(f'Correlation_grad_vec/Last_Corr_dqn_out1_grad_b2', Corr_dqn_out1_mean_b2, total_num_steps)

            # wandb.log({f'Correlation_grad_vec/Corr_dqn_L2_grad': Corr_dqn_L2_grad_mean})

            # for i in range(len(coherence)):
            #     summary_writer.add_scalar(f'Coherence/{i}', coherence[i], total_num_steps)
            #     summary_writer.add_scalar(f'Coherence_val/{i}', val_coherence[i], total_num_steps)
            #     summary_writer.add_scalar(f'Correlation/{i}', cosine_layers[i], total_num_steps)
            #
            #     summary_writer.add_scalar(f'Coherence_sum10/{i}', coherence_sum10[i], total_num_steps)
            #     summary_writer.add_scalar(f'Coherence_val_sum10/{i}', val_coherence_sum10[i], total_num_steps)
            #     summary_writer.add_scalar(f'Correlation_sum10/{i}', cosine_layers_sum10[i], total_num_steps)
            #
            #     summary_writer.add_scalar(f'Coherence_sum100/{i}', coherence_sum100[i], total_num_steps)
            #     summary_writer.add_scalar(f'Coherence_val_sum100/{i}', val_coherence_sum100[i], total_num_steps)
            #     summary_writer.add_scalar(f'Correlation_sum100/{i}', cosine_layers_sum100[i], total_num_steps)

                # summary_writer.add_scalar(f'Coherence_b2/{i}', coherence_b2[i], total_num_steps)
                # summary_writer.add_scalar(f'Coherence_val_b2/{i}', val_coherence_b2[i], total_num_steps)
                # summary_writer.add_scalar(f'Correlation_b2/{i}', cosine_layers_b2[i], total_num_steps)

            if params.reinitialization_last and (ts % 1000) == 0:
                nn.init.orthogonal_(agent.q_network.layer_last.weight)
                nn.init.zeros_(agent.q_network.layer_last.bias)

                nn.init.orthogonal_(agent.target_q_network.layer_last.weight)
                nn.init.zeros_(agent.target_q_network.layer_last.bias)

            if params.reinitialization_last_GRU and (ts % 1000) == 0:
                nn.init.orthogonal_(agent.q_network.layer_last.weight)
                nn.init.zeros_(agent.q_network.layer_last.bias)

                nn.init.orthogonal_(agent.target_q_network.layer_last.weight)
                nn.init.zeros_(agent.target_q_network.layer_last.bias)

                for name, param in agent.q_network.gru.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
                for name, param in agent.target_q_network.gru.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)

    wandb.finish()



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
    parser.add_argument("--mini-batch-size-val", type=int, default=32, help='size of mini-batches (default: 32)')
    parser.add_argument("--mini-batch-size-val2", type=int, default=32, help='size of mini-batches (default: 32)')
    # parser.add_argument("--CnnDQN", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--learning_rate_val", type=float, default=0.1)
    parser.add_argument("--num_val_updates", type=int, default=1)
    # parser.add_argument("--target_update_rate", type=float, default=0.1)
    # parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--save-interval",type=int,default=1000, help='save interval, one save per n updates (default: 1000)')
    # parser.add_argument("--save-grad",type=int,default=1000, help='save gradient, one save per n updates (default: 1000)')
    parser.add_argument("--max-grad-sum",type=int,default=1000, help='max gradient sum, one save per n updates (default: 1000)')
    parser.add_argument("--max_ts", type=int, default=1400000)
    # parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--target_network_update_f", type=int, default=10000)
    parser.add_argument("--loss_var_coeff", type=float, default=0.0)
    parser.add_argument("--loss_corr_coeff", type=float, default=0.0)
    parser.add_argument("--loss_corr_coeff_update", type=float, default=1.0)
    parser.add_argument("--zero_ind", action='store_true', default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--target_hard", action='store_true', default=False)
    parser.add_argument("--update_target", action='store_true', default=False)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--max_attn_grad_norm", type=float, default=20.0)
    parser.add_argument('--reinitialization_last', action='store_true', default=False)
    parser.add_argument('--reinitialization_last_GRU', action='store_true', default=False)
    main_dqn(parser.parse_args())
