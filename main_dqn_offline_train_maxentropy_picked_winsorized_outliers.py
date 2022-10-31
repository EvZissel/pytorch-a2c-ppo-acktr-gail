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
from models_dqn import DQN, CnnDQN, DQN_softAttn, DQN_softAttn_L2grad, DQN_RNNLast, DQN_RNNLast_analytic
from a2c_ppo_acktr.envs import make_vec_envs
from torch.utils.tensorboard import SummaryWriter
from a2c_ppo_acktr import utils
from grad_tools.grad_plot import GradPlotDqn
import pandas as pd
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

        q_value, _, _, _, rnn_hxs = self.q_network.forward(state, hidden_state, masks)
        if random.random() > epsilon:
            return q_value.max(1)[1].unsqueeze(-1), rnn_hxs
        return torch.tensor(np.random.randint(self.env.action_space.n, size=q_value.size()[0])).type(dtypelong).unsqueeze(-1), rnn_hxs


# def compute_td_gard(agent, num_mini_batch, mini_batch_size, replay_buffer, optimizer, gamma, device = "CPU", same_ind=False, start_ind_array=None):
#     num_processes = replay_buffer.rewards.size(1)
#     num_steps = replay_buffer.rewards.size(0)
#     num_steps_per_batch = int(num_steps/num_mini_batch)
#
#
#     if not same_ind:
#         start_ind_array = [i for i in range(0, num_steps, num_steps_per_batch)]
#         start_ind_array = np.random.choice(start_ind_array, size=mini_batch_size, replace=False)
#
#     all_losses = []
#     grads = []
#     shapes = []
#     grad_L2_states_all = 0
#     out1_states_all = 0
#     out2_states_all = 0
#     out3_states_all = 0
#     states_all_all = 0
#     # states_all = 0
#     for i in range(num_processes):
#         all_losses.append(0)
#         # all_grad_W2.append(0)
#         # all_grad_b2.append(0)
#
#     recurrent_hidden = torch.zeros(1, agent.q_network.recurrent_hidden_state_size).type(dtype)
#     for start_ind in start_ind_array:
#         data_sampler =  replay_buffer.sampler(num_processes, start_ind, num_steps_per_batch)
#
#         losses = []
#         # grad_W2 = []
#         # grad_b2 = []
#         grad_L2_states = 0
#         out1_states = 0
#         out2_states = 0
#         out3_states = 0
#         states_all = 0
#         # states_all_b = 0
#
#         for states, actions, rewards, done in data_sampler:
#
#             # double q-learning
#             with torch.no_grad():
#                     # recurrent_hidden.detach()
#                 online_q_values, _, _, _, _ = agent.q_network(states.to(device), recurrent_hidden, done.to(device))
#                 _, max_indicies = torch.max(online_q_values, dim=1)
#                 target_q_values, _, _, _, _  = agent.target_q_network(states.to(device), recurrent_hidden, done.to(device))
#                 next_q_value = target_q_values.gather(1, max_indicies.unsqueeze(1))
#
#                 next_q_value = next_q_value * done.to(device)
#                 expected_q_value = (rewards.to(device) + gamma * next_q_value[1:, :]).squeeze(1)
#
#             # Normal DDQN update
#             # with torch.backends.cudnn.flags(enabled=False):
#             q_values, out_3, out_2, out_1, _ = agent.q_network(states.to(device), recurrent_hidden, done.to(device))
#             q_value = q_values[:-1, :].gather(1, actions.to(device)).squeeze(1)
#             out_1 = out_1[:-1, :].unsqueeze(1)
#             out_2 = out_2[:-1, :].unsqueeze(1)
#             out_3 = out_3[:-1, :].unsqueeze(1)
#
#             td_err = (q_value - expected_q_value.data).unsqueeze(1).unsqueeze(1)
#             e = torch.zeros(num_steps_per_batch, actions.size(0), 1, device=device)
#             e[torch.arange(e.size(0)).unsqueeze(1), actions.to(device)] = 1.
#             # grad_b2 = td_err*e
#             # e_actions = torch.sparse_coo_tensor(torch.cat((actions, torch.tensor([0, 1, 2, 3, 4, 5], device=actions.device).unsqueeze(1)), dim=1).t(),(q_value - expected_q_value.data), (6, 6)).to_dense().unsqueeze(2)
#             # batch_grad_b2 = e_actions.mean(1)
#             # e_actions = e_actions.t().unsqueeze(2)
#             # out_1 = out_1.unsqueeze(1)
#             # batch_grad_W2 = torch.matmul(e_actions, out_1).mean(0)
#             # grad_w2 = (e*out_1)
#             grad_L2_states_b = td_err*torch.cat(((e*out_1), e), dim=2)
#             grad_L2_states_b = torch.flatten(grad_L2_states_b, start_dim=1).unsqueeze(1)
#
#             one_loss = 0.5*(q_value - expected_q_value.data).pow(2).mean()
#             losses.append(one_loss)
#             # grad_W2.append(batch_grad_W2)
#             # grad_b2.append(batch_grad_b2)
#             # states_all_b += torch.sigmoid(agent.q_network.input_attention) *states
#             if (len(losses)==1):
#                 grad_L2_states = grad_L2_states_b
#                 out1_states = out_1
#                 out2_states = out_2
#                 out3_states = out_3
#                 states_all = states[:-1, :].unsqueeze(1)
#             else:
#                 grad_L2_states = torch.cat((grad_L2_states, grad_L2_states_b), dim=1)
#                 out1_states = torch.cat((out1_states, out_1), dim=1)
#                 out2_states = torch.cat((out2_states, out_1), dim=1)
#                 out3_states = torch.cat((out3_states, out_1), dim=1)
#                 states_all = torch.cat((states_all, states[:-1, :].unsqueeze(1)), dim=1)
#
#             # grads = torch.autograd.grad(one_loss,
#             #                             updated_train_params.values(),
#             #                             create_graph=create_graph)
#
#         # loss = torch.stack(losses)
#         # loss = losses.mean(0)
#         # all_losses.append(loss)
#         for i in range(num_processes):
#             all_losses[i] += losses[i]/mini_batch_size
#             # all_grad_W2[i] += grad_W2[i]/mini_batch_size
#             # all_grad_b2[i] += grad_b2[i]/mini_batch_size
#
#         if start_ind == start_ind_array[0] :
#             grad_L2_states_all = grad_L2_states.mean(1)
#             out1_states_all = out1_states
#             out2_states_all = out2_states
#             out3_states_all = out3_states
#             states_all_all = states_all
#         else:
#             grad_L2_states_all = torch.cat((grad_L2_states_all, grad_L2_states.mean(1)), dim=0)
#             out1_states_all = torch.cat((out1_states_all, out1_states), dim=0)
#             out2_states_all = torch.cat((out2_states_all, out1_states), dim=0)
#             out3_states_all = torch.cat((out3_states_all, out1_states), dim=0)
#             states_all_all = torch.cat((states_all_all, states_all), dim=0)
#
#     # all_losses = random.choices(all_losses, k=k)
#     total_loss = torch.stack(all_losses).mean()
#
#     # total_loss = total_loss.mean() + loss_var_coeff * total_loss.var()
#     optimizer.zero_grad()
#     # grads, shapes = optimizer.plot_backward(all_losses)
#     # total_loss.backward()
#     grads, shapes, has_grads = optimizer._pack_grad([total_loss])
#     mean_flat_grad = optimizer._merge_grad(grads, has_grads)
#     mean_grad = optimizer._unflatten_grad(mean_flat_grad, shapes[0])
#     # optimizer._set_grad(mean_grad)
#     # for i in
#     #     grads
#     # optimizer.step()
#
#     # out_1_grad =0
#
#     # if train:
#     #     out1_states_all = out1_states_all[:,0,:]
#     # else:
#     #     out1_states_all = out1_states_all[:,2,:]
#
#     # return total_loss, grad_L2_states_all, out1_states_all, start_ind_array, out_1_grad, grads, shapes[0]
#     return total_loss, grad_L2_states_all, out1_states_all, out2_states_all, out3_states_all, start_ind_array, mean_grad


def compute_td_loss(agent, num_mini_batch, mini_batch_size, replay_buffer, optimizer, gamma, loss_var_coeff, k=0, min_corr = -1.0, device = "CPU", train=True, compute_analytic=False, same_ind=False, start_ind_array=None):
    num_processes = replay_buffer.rewards.size(1)
    num_steps = replay_buffer.rewards.size(0)
    num_steps_per_batch = int(num_steps/num_mini_batch)


    if not same_ind:
        start_ind_array = [i for i in range(0, num_steps, num_steps_per_batch)]
        start_ind_array = np.random.choice(start_ind_array, size=mini_batch_size, replace=False)

    all_losses = []
    # grads = []
    # shapes = []
    grad_L2_states_all = 0
    out1_states_all = 0
    out2_states_all = 0
    out3_states_all = 0
    states_all_all = 0
    # states_all = 0

    # cosine_grad = []
    # cosine_grad_layers = []
    # cosine_grad_activation = []


    for i in range(num_processes):
        all_losses.append(0)

    recurrent_hidden = torch.zeros(1, agent.q_network.recurrent_hidden_state_size).type(dtype)
    for start_ind in start_ind_array:
        data_sampler = replay_buffer.sampler(num_processes, start_ind, num_steps_per_batch)

        losses = []
        grad_L2_states = 0
        out1_states = 0
        out2_states = 0
        out3_states = 0
        states_all = 0
        # states_all_b = 0

        for states, actions, rewards, done in data_sampler:

            # double q-learning
            with torch.no_grad():
                    # recurrent_hidden.detach()
                online_q_values, _, _, _, _  = agent.q_network(states.to(device), recurrent_hidden, done.to(device))
                _, max_indicies = torch.max(online_q_values, dim=1)
                target_q_values, _, _, _, _ = agent.target_q_network(states.to(device), recurrent_hidden, done.to(device))
                next_q_value = target_q_values.gather(1, max_indicies.unsqueeze(1))

                next_q_value = next_q_value * done.to(device)
                expected_q_value = (rewards.to(device) + gamma * next_q_value[1:, :]).squeeze(1)

            # Normal DDQN update
            # with torch.backends.cudnn.flags(enabled=False):
            q_values, out_3, out_2, out_1, _ = agent.q_network(states.to(device), recurrent_hidden, done.to(device))
            q_value = q_values[:-1, :].gather(1, actions.to(device)).squeeze(1)
            out_1 = out_1[:-1, :].unsqueeze(1)
            out_2 = out_2[:-1, :].unsqueeze(1)
            out_3 = out_3[:-1, :].unsqueeze(1)

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
                out2_states = out_2
                out3_states = out_3
                states_all = states[:-1, :].unsqueeze(1)
            else:
                grad_L2_states = torch.cat((grad_L2_states, grad_L2_states_b), dim=1)
                out1_states = torch.cat((out1_states, out_1), dim=1)
                out2_states = torch.cat((out2_states, out_2), dim=1)
                out3_states = torch.cat((out3_states, out_3), dim=1)
                states_all = torch.cat((states_all, states[:-1, :].unsqueeze(1)), dim=1)

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

        if start_ind == start_ind_array[0] :
            grad_L2_states_all = grad_L2_states.mean(1)
            out1_states_all = out1_states
            out2_states_all = out2_states
            out3_states_all = out3_states
            states_all_all = states_all
        else:
            grad_L2_states_all = torch.cat((grad_L2_states_all, grad_L2_states.mean(1)), dim=0)
            out1_states_all = torch.cat((out1_states_all, out1_states), dim=0)
            out2_states_all = torch.cat((out2_states_all, out1_states), dim=0)
            out3_states_all = torch.cat((out3_states_all, out1_states), dim=0)
            states_all_all = torch.cat((states_all_all, states_all), dim=0)

    # all_losses = random.choices(all_losses, k=k)
    total_loss = torch.stack(all_losses).mean()


    # all_losses = all_losses[int(5*k):int(5*k)+5]
    # total_grad_W2 = torch.stack(all_grad_W2).mean(0)
    # total_grad_b2 = torch.stack(all_grad_b2).mean(0)

    # total_grad_L2 = _flatten_grad((total_grad_W2, total_grad_b2))
    # total_grad_L2 = []
    # for i in range(num_processes):
    #     total_grad_L2.append(_flatten_grad(torch.cat((all_grad_W2[i],torch.unsqueeze(all_grad_b2[i],1)), dim=1)))
    # total_grad_L2_state = []
    # for i in range(grad_L2_states_all.shape[0]):
    #     total_grad_L2_state.append(_flatten_grad(grad_L2_states_all[i]))

    # total_loss = total_loss.mean() + loss_var_coeff * total_loss.var()
    optimizer.zero_grad()
    grads, shapes = optimizer.plot_backward(all_losses)
    # total_loss.backward()
    # grads, shapes, has_grads = optimizer._pack_grad([total_loss])
    # mean_flat_grad = optimizer._merge_grad(grads, has_grads)

    Wh1 = []
    Wh2 = []

    bh1 = []
    bh2 = []

    L1w = []
    L1b = []

    L2w = []
    L2b = []

    LLw = []
    LLb = []

    for grad in grads:
        grad = optimizer._unflatten_grad(grad, shapes)
        Wh1.append(grad[0])
        Wh2.append(grad[1])

        bh1.append(grad[2])
        bh2.append(grad[3])

        L1w.append(grad[4])
        L1b.append(grad[5])

        L2w.append(grad[6])
        L2b.append(grad[7])

        LLw.append(grad[8])
        LLb.append(grad[9])

    for i in range(k):
        cosine_Wh1_WOi_mean = []
        mean_Wh1 = sum(Wh1) / len(Wh1)
        for j in range(len(Wh1)):

            mean_Wh1_WOi = Wh1[:j] + Wh1[j+1:]
            mean_grad_WOi = sum(mean_Wh1_WOi) / len(mean_Wh1_WOi)

            cosine_Wh1_WOi_mean.append((mean_grad_WOi * mean_Wh1).sum() / (mean_grad_WOi.norm(2) * mean_Wh1.norm(2)))

        smallest_val = torch.sort(torch.tensor(cosine_Wh1_WOi_mean))[0][0]
        smallest_ind = torch.sort(torch.tensor(cosine_Wh1_WOi_mean))[1][0]
        if smallest_val < min_corr:
            Wh1 = Wh1[:smallest_ind] + Wh1[smallest_ind + 1:]

        cosine_Wh2_WOi_mean = []
        mean_Wh2 = sum(Wh2) / len(Wh2)
        for j in range(len(Wh2)):
            mean_Wh2_WOi = Wh2[:j] + Wh2[j + 1:]
            mean_grad_WOi = sum(mean_Wh2_WOi) / len(mean_Wh2_WOi)

            cosine_Wh2_WOi_mean.append((mean_grad_WOi * mean_Wh2).sum() / (mean_grad_WOi.norm(2) * mean_Wh2.norm(2)))

        smallest_val = torch.sort(torch.tensor(cosine_Wh2_WOi_mean))[0][0]
        smallest_ind = torch.sort(torch.tensor(cosine_Wh2_WOi_mean))[1][0]
        if smallest_val < min_corr:
            Wh2 = Wh2[:smallest_ind] + Wh2[smallest_ind + 1:]

        cosine_bh1_WOi_mean = []
        mean_bh1 = sum(bh1) / len(bh1)
        for j in range(len(bh1)):
            mean_bh1_WOi = bh1[:j] + bh1[j + 1:]
            mean_grad_WOi = sum(mean_bh1_WOi) / len(mean_bh1_WOi)

            cosine_bh1_WOi_mean.append((mean_grad_WOi * mean_bh1).sum() / (mean_grad_WOi.norm(2) * mean_bh1.norm(2)))

        smallest_val = torch.sort(torch.tensor(cosine_bh1_WOi_mean))[0][0]
        smallest_ind = torch.sort(torch.tensor(cosine_bh1_WOi_mean))[1][0]
        if smallest_val < min_corr:
            bh1 = bh1[:smallest_ind] + bh1[smallest_ind + 1:]

        cosine_bh2_WOi_mean = []
        mean_bh2 = sum(bh2) / len(bh2)
        for j in range(len(bh2)):
            mean_bh2_WOi = bh2[:j] + bh2[j + 1:]
            mean_grad_WOi = sum(mean_bh2_WOi) / len(mean_bh2_WOi)

            cosine_bh2_WOi_mean.append((mean_grad_WOi * mean_bh2).sum() / (mean_grad_WOi.norm(2) * mean_bh2.norm(2)))

        smallest_val = torch.sort(torch.tensor(cosine_bh2_WOi_mean))[0][0]
        smallest_ind = torch.sort(torch.tensor(cosine_bh2_WOi_mean))[1][0]
        if smallest_val < min_corr:
            bh2 = bh2[:smallest_ind] + bh2[smallest_ind + 1:]

        cosine_L1w_WOi_mean = []
        mean_L1w = sum(L1w) / len(L1w)
        for j in range(len(L1w)):
            mean_L1w_WOi = L1w[:j] + L1w[j + 1:]
            mean_grad_WOi = sum(mean_L1w_WOi) / len(mean_L1w_WOi)

            cosine_L1w_WOi_mean.append((mean_grad_WOi * mean_L1w).sum() / (mean_grad_WOi.norm(2) * mean_L1w.norm(2)))

        smallest_val = torch.sort(torch.tensor(cosine_L1w_WOi_mean))[0][0]
        smallest_ind = torch.sort(torch.tensor(cosine_L1w_WOi_mean))[1][0]
        if smallest_val < min_corr:
            L1w = L1w[:smallest_ind] + L1w[smallest_ind + 1:]

        cosine_L1b_WOi_mean = []
        mean_L1b = sum(L1b) / len(L1b)
        for j in range(len(L1b)):
            mean_L1b_WOi = L1b[:j] + L1b[j + 1:]
            mean_grad_WOi = sum(mean_L1b_WOi) / len(mean_L1b_WOi)

            cosine_L1b_WOi_mean.append((mean_grad_WOi * mean_L1b).sum() / (mean_grad_WOi.norm(2) * mean_L1b.norm(2)))

        smallest_val = torch.sort(torch.tensor(cosine_L1b_WOi_mean))[0][0]
        smallest_ind = torch.sort(torch.tensor(cosine_L1b_WOi_mean))[1][0]
        if smallest_val < min_corr:
            L1b = L1b[:smallest_ind] + L1b[smallest_ind + 1:]

        cosine_L2w_WOi_mean = []
        mean_L2w = sum(L2w) / len(L2w)
        for j in range(len(L2w)):
            mean_L2w_WOi = L2w[:j] + L2w[j + 1:]
            mean_grad_WOi = sum(mean_L2w_WOi) / len(mean_L2w_WOi)

            cosine_L2w_WOi_mean.append((mean_grad_WOi * mean_L2w).sum() / (mean_grad_WOi.norm(2) * mean_L2w.norm(2)))

        smallest_val = torch.sort(torch.tensor(cosine_L2w_WOi_mean))[0][0]
        smallest_ind = torch.sort(torch.tensor(cosine_L2w_WOi_mean))[1][0]
        if smallest_val < min_corr:
            L2w = L2w[:smallest_ind] + L2w[smallest_ind + 1:]

        cosine_L2b_WOi_mean = []
        mean_L2b = sum(L2b) / len(L2b)
        for j in range(len(L2b)):
            mean_L2b_WOi = L2b[:j] + L2b[j + 1:]
            mean_grad_WOi = sum(mean_L2b_WOi) / len(mean_L2b_WOi)

            cosine_L2b_WOi_mean.append((mean_grad_WOi * mean_L2b).sum() / (mean_grad_WOi.norm(2) * mean_L2b.norm(2)))

        smallest_val = torch.sort(torch.tensor(cosine_L2b_WOi_mean))[0][0]
        smallest_ind = torch.sort(torch.tensor(cosine_L2b_WOi_mean))[1][0]
        if smallest_val < min_corr:
            L2b = L2b[:smallest_ind] + L2b[smallest_ind + 1:]

        cosine_LLw_WOi_mean = []
        mean_LLw = sum(LLw) / len(LLw)
        for j in range(len(LLw)):
            mean_LLw_WOi = LLw[:j] + LLw[j + 1:]
            mean_grad_WOi = sum(mean_LLw_WOi) / len(mean_LLw_WOi)

            cosine_LLw_WOi_mean.append((mean_grad_WOi * mean_LLw).sum() / (mean_grad_WOi.norm(2) * mean_LLw.norm(2)))

        smallest_val = torch.sort(torch.tensor(cosine_LLw_WOi_mean))[0][0]
        smallest_ind = torch.sort(torch.tensor(cosine_LLw_WOi_mean))[1][0]
        if smallest_val < min_corr:
            LLw = LLw[:smallest_ind] + LLw[smallest_ind + 1:]

        cosine_LLb_WOi_mean = []
        mean_LLb = sum(LLb) / len(LLb)
        for j in range(len(LLb)):
            mean_LLb_WOi = LLb[:j] + LLb[j + 1:]
            mean_grad_WOi = sum(mean_LLb_WOi) / len(mean_LLb_WOi)

            cosine_LLb_WOi_mean.append((mean_grad_WOi * mean_LLb).sum() / (mean_grad_WOi.norm(2) * mean_LLb.norm(2)))

        smallest_val = torch.sort(torch.tensor(cosine_LLb_WOi_mean))[0][0]
        smallest_ind = torch.sort(torch.tensor(cosine_LLb_WOi_mean))[1][0]
        if smallest_val < min_corr:
            LLb = LLb[:smallest_ind] + LLb[smallest_ind + 1:]

    # for grad in grads:
    #     grad = optimizer._unflatten_grad(grad, shapes)
    #     cosine_Wh1_i = (grad[0] * mean_grad_400[0]).sum() / (grad[0].norm(2) * mean_grad_400[0].norm(2))
    #     cosine_Wh2_i = (grad[1] * mean_grad_400[1]).sum() / (grad[1].norm(2) * mean_grad_400[1].norm(2))
    #     cosine_bh1_i = (grad[2] * mean_grad_400[2]).sum() / (grad[2].norm(2) * mean_grad_400[2].norm(2))
    #     cosine_bh2_i = (grad[3] * mean_grad_400[3]).sum() / (grad[3].norm(2) * mean_grad_400[3].norm(2))
    #     cosine_L1w_i = (grad[4] * mean_grad_400[4]).sum() / (grad[4].norm(2) * mean_grad_400[4].norm(2))
    #     cosine_L1b_i = (grad[5] * mean_grad_400[5]).sum() / (grad[5].norm(2) * mean_grad_400[5].norm(2))
    #     cosine_L2w_i = (grad[6] * mean_grad_400[6]).sum() / (grad[6].norm(2) * mean_grad_400[6].norm(2))
    #     cosine_L2b_i = (grad[7] * mean_grad_400[7]).sum() / (grad[7].norm(2) * mean_grad_400[7].norm(2))
    #     cosine_LLw_i = (grad[8] * mean_grad_400[8]).sum() / (grad[8].norm(2) * mean_grad_400[8].norm(2))
    #     cosine_LLb_i = (grad[9] * mean_grad_400[9]).sum() / (grad[9].norm(2) * mean_grad_400[9].norm(2))
    #
    #     cosine_grad_layers[0].append(cosine_Wh1_i)
    #     cosine_grad_layers[1].append(cosine_Wh2_i)
    #     cosine_grad_layers[2].append(cosine_bh1_i)
    #     cosine_grad_layers[3].append(cosine_bh2_i)
    #     cosine_grad_layers[4].append(cosine_L1w_i)
    #     cosine_grad_layers[5].append(cosine_L1b_i)
    #     cosine_grad_layers[6].append(cosine_L2w_i)
    #     cosine_grad_layers[7].append(cosine_L2b_i)
    #     cosine_grad_layers[8].append(cosine_LLw_i)
    #     cosine_grad_layers[9].append(cosine_LLb_i)
    #
    #     # cosine_grad.append(cosine_Wh1_i + cosine_Wh2_i + cosine_bh1_i + cosine_bh2_i + cosine_L1w_i + cosine_L1b_i + cosine_L2w_i + cosine_L2b_i + cosine_LLw_i + cosine_LLb_i)
    #     cosine_grad.append(cosine_L1w_i)

    # idx = torch.stack(cosine_grad).sort(descending=True)[1][:k]
    # grads_5 = [grads[i] for i in idx]
    # mean_grad = sum(grads_5) / len(grads_5)
    # mean_grad = optimizer._unflatten_grad(mean_grad, shapes)
    # print("top 5 cosine grade: {}".format(cosine_grad[:k]))

    # mean_grad = optimizer._unflatten_grad(grads[0], shapes)
    # for i in range(len(mean_grad)):
    #     idx = torch.stack(cosine_grad_layers[i]).sort(descending=True)[1][:k]
    #     grads_5 = [optimizer._unflatten_grad(grads[j], shapes)[i] for j in idx]
    #     mean_grad[i] = sum(grads_5) / len(grads_5)
    #     print([cosine_grad_layers[i][j].item() for j in idx])

    mean_grad = optimizer._unflatten_grad(grads[0], shapes)
    for grad in mean_grad:
        grad.zero_()

    if (len(Wh1) > 1):
        mean_grad[0] = sum(Wh1) / len(Wh1)
    print("Wh1 len {}".format(len(Wh1)))
    if (len(Wh2) > 1):
        mean_grad[1] = sum(Wh2) / len(Wh2)
    print("Wh2 len {}".format(len(Wh2)))
    if (len(bh1) > 1):
        mean_grad[2] = sum(bh1) / len(bh1)
    print("bh1 len {}".format(len(bh1)))
    if (len(bh2) > 1):
        mean_grad[3] = sum(bh2) / len(bh2)
    print("bh2 len {}".format(len(bh2)))
    if (len(L1w) > 1):
        mean_grad[4] = sum(L1w) / len(L1w)
    print("L1w len {}".format(len(L1w)))
    if (len(L1b) > 1):
        mean_grad[5] = sum(L1b) / len(L1b)
    print("L1b len {}".format(len(L1b)))
    if (len(L2w) > 1):
        mean_grad[6] = sum(L2w) / len(L2w)
    print("L2w len {}".format(len(L2w)))
    if (len(L2b) > 1):
        mean_grad[7] = sum(L2b) / len(L2b)
    print("L2b len {}".format(len(L2b)))
    if (len(LLw) > 1):
        mean_grad[8] = sum(LLw) / len(LLw)
    print("LLw len {}".format(len(LLw)))
    if (len(LLb) > 1):
        mean_grad[9] = sum(LLb) / len(LLb)
    print("LLb len {}".format(len(LLb)))

    # idx = torch.nonzero(torch.stack(cosine_grad) > min_corr)
    # if len(idx) > 0:
    #     grads_k = [grads[i] for i in idx]
    #     mean_grad = sum(grads_k) / len(grads_k)
    #     mean_grad = optimizer._unflatten_grad(mean_grad, shapes)
    #     print("top {} cosine grade: {}".format(min_corr, [cosine_grad[i] for i in idx]))


    # mean_grad = optimizer._unflatten_grad(grads[0], shapes)
    # for grad in mean_grad:
    #     grad.zero_()
    # for i in range(len(mean_grad)):
    #     idx = torch.nonzero(torch.stack(cosine_grad_layers[i]) > min_corr)
    #     if len(idx) > 0:
    #         grads_k = [optimizer._unflatten_grad(grads[j], shapes)[i] for j in idx]
    #         mean_grad[i] = sum(grads_k) / len(grads_k)
    #         print("Layer {}: {}".format(i,[cosine_grad_layers[i][j].item() for j in idx]))

    optimizer._set_grad(mean_grad)
    optimizer.step()

    # optimizer.zero_grad()
    # atten_grads_loss = torch.autograd.grad(total_loss,
    #                             agent.q_network.input_attention,
    #                             retain_graph=True)
    #
    # if train:
    #     total_loss.backward(retain_graph=True)
    #     optimizer.step()

    # out_1_grad = torch.zeros((out1_states_all.size()[-1],agent.q_network.input_attention.size()[0])).type(dtype)
    out_1_grad =0

    # if compute_analytic:
    #     out1_states_all_flat = torch.flatten(out1_states_all, start_dim=0, end_dim=1)
    #     out1_states = out1_states_all_flat.mean(0)
    #     # states_all_all_flat = torch.flatten(states_all_all, start_dim=0, end_dim=1)
    #     for i in range(out1_states.size()[0]):
    #         loss_i = out1_states[i]
    #         # loss_i.backward()
    #         loss_i_grads = torch.autograd.grad(loss_i,
    #                                             agent.q_network.input_attention,
    #                                             retain_graph=True)
    #         out_1_grad[i,:] = loss_i_grads[0]

    # grads = None
    # else:
    #     grads = torch.autograd.grad(total_loss,
    #                                 updated_train_params.values(),
    #                                 create_graph=create_graph)
    if train:
        out1_states_all = out1_states_all[:,0,:]
    else:
        out1_states_all = out1_states_all[:,2,:]

    # return total_loss, grad_L2_states_all, out1_states_all, start_ind_array, out_1_grad, grads, shapes[0]
    return total_loss, grad_L2_states_all, out1_states_all, start_ind_array, out_1_grad


def evaluate(agent, eval_envs_dic ,env_name, eval_locations_dic, num_processes, num_tasks, **kwargs):

    eval_envs = eval_envs_dic[env_name]
    locations = eval_locations_dic[env_name]
    eval_episode_rewards = []
    num_uniform = 0

    for iter in range(0, num_tasks, num_processes):
        eval_actions = []
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

            eval_actions.append(actions)
            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        eval_actions = torch.stack(eval_actions).squeeze()
        for i in range(num_processes):
            if len(torch.unique(eval_actions[:,i])) == len(eval_actions[:,i]):
                num_uniform += 1

    return eval_episode_rewards, eval_actions, num_uniform


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
    if not params.debug:
        logdir_ = 'offline_train_picked_winsorized_outliers' +  params.env + '_' + str(params.seed) + '_num_arms_' + str(params.num_processes) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        if params.rotate:
            logdir_ = logdir_ + '_rotate'
        if params.zero_ind:
            logdir_ = logdir_ + '_Zero_ind'

        logdir = os.path.join('dqn_runs_offline', logdir_)
        logdir = os.path.join(os.path.expanduser(params.log_dir), logdir)
        utils.cleanup_log_dir(logdir)
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_hparams(vars(params), {})

        wandb.init(project="main_dqn_offline_maximum_entropy_train_saveGrad", entity="ev_zisselman", config=params, name=logdir_, id=logdir_)

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
                         obs_recurrent=params.obs_recurrent, multi_task=True, normalize=not params.no_normalize, rotate=params.rotate, obs_rand_loc=params.obs_rand_loc)

    # envs_400 = make_vec_envs(params.env_400, params.seed, 416,  np.random.randint(0, 6, size=416),
    #                      params.gamma, None, device, False, steps=params.task_steps,
    #                      free_exploration=params.free_exploration, recurrent=params.recurrent_policy,
    #                      obs_recurrent=params.obs_recurrent, multi_task=True, normalize=not params.no_normalize, rotate=params.rotate, obs_rand_loc=params.obs_rand_loc)

    # val_envs = make_vec_envs(params.val_env, params.seed, params.num_processes, eval_locations_dic['valid_eval'],
    #                      params.gamma, None, device, False, steps=params.task_steps,
    #                      free_exploration=params.free_exploration, recurrent=params.recurrent_policy,
    #                      obs_recurrent=params.obs_recurrent, multi_task=True, normalize=not params.no_normalize, rotate=params.rotate, obs_rand_loc=params.obs_rand_loc)

    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        eval_envs_dic[eval_disp_name] = make_vec_envs(eval_env_name[0], params.seed, params.num_processes, eval_locations_dic[eval_disp_name],
                                                      None, None, device, True, steps=params.task_steps,
                                                      recurrent=params.recurrent_policy,
                                                      obs_recurrent=params.obs_recurrent, multi_task=True,
                                                      free_exploration=params.free_exploration, normalize=not params.no_normalize, rotate=params.rotate, obs_rand_loc=params.obs_rand_loc)

    q_network = DQN_RNNLast(envs.observation_space.shape, envs.action_space.n, params.zero_ind, recurrent=True, hidden_size=params.hidden_size)
    target_q_network = deepcopy(q_network)
    if params.target_hard:
        target_q_network.target = True

    q_network = q_network.to(device)
    target_q_network = target_q_network.to(device)

    agent = Agent(envs, q_network, target_q_network)

    # attention_parameters = []
    # non_attention_parameters = []
    # for name, p in q_network.named_parameters():
    #     if 'attention' in name:
    #         attention_parameters.append(p)
    #     else:
    #         non_attention_parameters.append(p)

    # last_layer_param = []
    # for name, p in q_network.named_parameters():
    #     if 'layer_last' in name:
    #         last_layer_param.append(p)
    #     elif 'hidden_last' in name:
    #         last_layer_param.append(p)
    #     else:
    #         p.requires_grad = False

    # train_param = []
    # for name, p in q_network.named_parameters():
    #     if 'layer_1' in name:
    #         p.requires_grad = False
    #     elif 'layer_2' in name:
    #         p.requires_grad = False
    #     else:
    #         train_param.append(p)


    # optimizer = optim.Adam(non_attention_parameters, lr=params.learning_rate, weight_decay=params.weight_decay)
    # optimizer = optim.Adam(last_layer_param, lr=params.learning_rate, weight_decay=params.weight_decay)
    optimizer = optim.Adam(q_network.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    optimizer = GradPlotDqn(optimizer)
    # optimizer_val = optim.Adam(attention_parameters, lr=params.learning_rate_val, weight_decay=params.weight_decay)
    # scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)
    # grad_replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)
    # val_replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)
    # replay_buffer_400 = ReplayBufferBandit(params.num_steps, 416, envs.observation_space.shape, envs.action_space)

    # Load previous model
    if (params.continue_from_epoch > 0) and params.save_dir != "":
        save_path = params.save_dir
        q_network_weighs = torch.load(os.path.join(save_path, params.load_env + "-epoch-{}.pt".format(params.continue_from_epoch)), map_location=device)
        agent.q_network.load_state_dict(q_network_weighs['state_dict'], strict=False)
        agent.target_q_network.load_state_dict(q_network_weighs['target_state_dict'], strict=False)
        # hard_update(agent.q_network, agent.target_q_network)
        optimizer.load_state_dict(q_network_weighs['optimizer_state_dict']) #load weights only


    # Load pretrained
    if (params.saved_epoch > 0) and params.save_dir != "":
        save_path = params.save_dir
        q_network_weighs = torch.load(os.path.join(save_path, params.load_env + "-epoch-{}.pt".format(params.saved_epoch)), map_location=device)
        agent.q_network.load_state_dict(q_network_weighs['state_dict'], strict=False)
        agent.target_q_network.load_state_dict(q_network_weighs['target_state_dict'], strict=False)


    obs = envs.reset()
    replay_buffer.obs[0].copy_(obs)
    # replay_buffer.to(device)

    # val_obs = val_envs.reset()
    # val_replay_buffer.obs[0].copy_(val_obs)
    # # val_replay_buffer.to(device)

    episode_rewards = deque(maxlen=25)
    episode_len = deque(maxlen=25)
    val_episode_rewards = deque(maxlen=25)
    val_episode_len = deque(maxlen=25)

    losses = []
    # grad_losses = []
    val_losses = []

    # Collect train data
    # recurrent_hidden_states = torch.zeros(params.num_processes, agent.q_network.recurrent_hidden_state_size).type(dtypelong)
    for step in range(params.num_steps):

        # actions, recurrent_hidden_states = agent.act(replay_buffer.obs[step], recurrent_hidden_states, epsilon, replay_buffer.masks[step])
        actions = torch.tensor(np.random.randint(agent.num_actions, size=params.num_processes)).type(dtypelong).unsqueeze(-1)
        # actions = torch.tensor(np.random.randint(agent.num_actions) * np.ones(params.num_processes)).type(dtypelong).unsqueeze(-1)

        next_obs, reward, done, infos = envs.step(actions.cpu())


        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                episode_len.append(info['episode']['l'])

        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])

        replay_buffer.insert(next_obs, actions, reward, masks)


    # obs_400 = envs_400.reset()
    # replay_buffer_400.obs[0].copy_(obs_400)
    #
    # for step in range(params.num_steps):
    #
    #     # actions, recurrent_hidden_states = agent.act(replay_buffer.obs[step], recurrent_hidden_states, epsilon, replay_buffer.masks[step])
    #     actions = torch.tensor(np.random.randint(agent.num_actions, size=416)).type(dtypelong).unsqueeze(-1)
    #     # actions = torch.tensor(np.random.randint(agent.num_actions) * np.ones(params.num_processes)).type(dtypelong).unsqueeze(-1)
    #
    #     next_obs, reward, done, infos = envs_400.step(actions.cpu())
    #
    #
    #     for info in infos:
    #         if 'episode' in info.keys():
    #             episode_rewards.append(info['episode']['r'])
    #             episode_len.append(info['episode']['l'])
    #
    #     masks = torch.FloatTensor(
    #         [[0.0] if done_ else [1.0] for done_ in done])
    #
    #     replay_buffer_400.insert(next_obs, actions, reward, masks)

    # # Collect validation data
    # for step in range(params.num_steps):
    #
    #     val_actions = torch.tensor(np.random.randint(agent.num_actions, size=params.num_processes)).type(dtypelong).unsqueeze(-1)
    #     # val_actions = torch.tensor(np.random.randint(agent.num_actions) * np.ones(params.num_processes)).type(dtypelong).unsqueeze(-1).cpu()
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

    # min_corr = torch.tensor(params.min_corr)

    # alpha = params.loss_corr_coeff_update
    # loss_corr_coeff = params.loss_corr_coeff

    # # compute validation gradient for start_ind_array
    # loss, grads_L2, out1, start_ind_array = compute_td_loss(
    #     agent, params.num_mini_batch, params.mini_batch_size_val, replay_buffer, optimizer, params.gamma, params.loss_var_coeff, device, train=False,
    # )

    # out1_vec = deque(maxlen=10)
    # out1_val_vec = deque(maxlen=10)

    L2_grad_vec = deque(maxlen=10)
    L2_grad_val_vec = deque(maxlen=10)
    out1_vec_correlation = deque(maxlen=10)
    out1_val_vec_correlation = deque(maxlen=10)

    atten_grad_L2 = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    atten_grad_Loss = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    out1_vec = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    out1_val_vec = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    grad_out1_vec = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    grad_out1_val_vec = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    # atten_grad_Loss_val = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    gama = params.gamma_loss
    compute_analytic = True
    # k=0


    for ts in range(params.continue_from_epoch, params.continue_from_epoch+num_updates):
        # Update the q-network & the target network

        #### Update Theta #####
        # loss, grads_L2, out1, start_ind_array, out1_grad = compute_td_loss(
        #     agent, params.num_mini_batch, params.mini_batch_size, replay_buffer, optimizer, params.gamma, params.loss_var_coeff, k=int(k%5), device=device, train=True, compute_analytic=compute_analytic,
        # )
        # loss, grads_L2, out1, start_ind_array, out1_grad, grads, shapes  = compute_td_loss(
        #     agent, params.num_mini_batch, params.mini_batch_size, replay_buffer, optimizer, params.gamma, params.loss_var_coeff, k=0, device=device, train=True, compute_analytic=compute_analytic,
        # )

        # loss, grads_L2, out1, out2, out3, _, mean_grad_400 = compute_td_gard(agent, params.num_mini_batch, 1, replay_buffer_400, optimizer, params.gamma, device=device)

        loss, grads_L2, out1, start_ind_array, out1_grad  = compute_td_loss(
            agent, params.num_mini_batch, params.mini_batch_size, replay_buffer, optimizer, params.gamma, params.loss_var_coeff, k=params.k, min_corr=params.min_corr, device=device, train=True, compute_analytic=compute_analytic,
        )
        losses.append(loss.data)

        # k += 1

        # val_loss, val_grads_L2, val_out1, _, val_out1_grad = compute_td_loss(
        #     agent, params.num_mini_batch, params.mini_batch_size, val_replay_buffer, optimizer, params.gamma, params.loss_var_coeff, device, train=False, compute_analytic=compute_analytic,
        #     same_ind=True, start_ind_array=start_ind_array
        # )
        # val_losses.append(val_loss.data)

        # # Corr_dqn_L2_grad = 0
        # # for i in range(grads_L2.size(0)):
        # #     Corr_dqn_L2_grad += ((grads_L2[i] * val_grads_L2[i]).sum() / (grads_L2[i].norm(2) * val_grads_L2[i].norm(2)))/grads_L2.size(0)
        #
        # # compute_analytic = False
        # train_grads_L2 = grads_L2.mean(0)
        # val_grads_L2 = val_grads_L2.mean(0)
        # L2_grad_vec.append(train_grads_L2)
        # L2_grad_val_vec.append(val_grads_L2)
        # L2_grad_vec_mean =  sum(L2_grad_vec)/len(L2_grad_vec)
        # L2_grad_val_vec_mean =  sum(L2_grad_val_vec)/len(L2_grad_val_vec)
        # Corr_dqn_L2_grad = (train_grads_L2 * val_grads_L2).sum() / (train_grads_L2.norm(2) * val_grads_L2.norm(2))
        # Corr_dqn_L2_grad_mean = (L2_grad_vec_mean * L2_grad_val_vec_mean).sum() / (L2_grad_vec_mean.norm(2) * L2_grad_val_vec_mean.norm(2))
        # print("correlation L2 grad: {}".format(Corr_dqn_L2_grad))
        # print("correlation L2 grad mean: {}".format(Corr_dqn_L2_grad_mean))
        #
        # train_out1_Corr = out1.mean(0)
        # val_out1_Corr = val_out1.mean(0)
        # out1_vec_correlation.append(train_out1_Corr)
        # out1_val_vec_correlation.append(val_out1_Corr)
        # Corr_dqn_out1 = (train_out1_Corr * val_out1_Corr).sum() / (train_out1_Corr.norm(2) * val_out1_Corr.norm(2))
        # print("correlation out1: {}".format(Corr_dqn_out1))
        #

        # out1_vec_mean =  sum(out1_vec_correlation)/len(out1_vec_correlation)
        # out1_val_vec_mean =  sum(out1_val_vec_correlation)/len(out1_val_vec_correlation)
        # Corr_dqn_out1_mean = (out1_vec_mean*out1_val_vec_mean).sum() / (out1_vec_mean.norm(2) * out1_val_vec_mean.norm(2))
        # print("correlation out1 mean: {}".format(Corr_dqn_out1_mean))
        # L2_dqn_out1 = ((train_out1_Corr - val_out1_Corr)**2).sum()
        # print("L2 out1: {}".format(L2_dqn_out1))
        # L2_dqn_out1_mean = ((out1_vec_mean - out1_val_vec_mean)**2).sum()
        # print("L2 out1 mean: {}".format(L2_dqn_out1_mean))


        # atten_grads_loss = torch.autograd.grad(loss,
        #                                        agent.q_network.input_attention,
        #                                        retain_graph=True)
        # atten_grads_L2 = torch.autograd.grad(L2_dqn_out1,
        #                                     agent.q_network.input_attention,
        #                                     retain_graph=True)
        #
        # atten_grad_Loss.append(atten_grads_loss[0])
        # atten_grad_L2.append(atten_grads_L2[0])
        #
        # if (Corr_dqn_out1_mean < min_corr):
        #
        #     # compute_analytic = True
        #     # atten_grads_loss = torch.autograd.grad(loss,
        #     #                                        agent.q_network.input_attention,
        #     #                                        retain_graph=True)
        #     # # atten_grads_L2 = torch.autograd.grad(((train_out1_Corr - val_out1_Corr)**2).mean(),
        #     # #                                        agent.q_network.input_attention,
        #     # #                                        retain_graph=True)
        #     #
        #     # atten_grad_Loss.append(atten_grads_loss[0])
        #     # # atten_grad_L2.append(atten_grads_L2[0])
        #
        #     atten_grad_Loss_mean = sum(atten_grad_Loss)/len(atten_grad_Loss)
        #     atten_grad_L2_mean = sum(atten_grad_L2)/len(atten_grad_L2)
        #
        #     # out1_vec.append(out1.mean(0))
        #     # out1_val_vec.append(val_out1.mean(0))
        #     # out1_vec_mean = sum(out1_vec) / len(out1_vec)
        #     # out1_val_vec_mean = sum(out1_val_vec) / len(out1_val_vec)
        #     #
        #     # grad_out1_vec.append(out1_grad)
        #     # grad_out1_val_vec.append(val_out1_grad)
        #     # grad_out1_vec_mean = sum(grad_out1_vec) / len(grad_out1_vec)
        #     # grad_out1_val_vec_mean = sum(grad_out1_val_vec) / len(grad_out1_val_vec)
        #     #
        #     # atten_grad_L2_mean = torch.matmul((out1_vec_mean-out1_val_vec_mean).unsqueeze(0),(grad_out1_vec_mean-grad_out1_val_vec_mean)).squeeze()
        #
        #     total_grad = atten_grad_Loss_mean + gama*atten_grad_L2_mean
        #
        #     optimizer_val.zero_grad()
        #     agent.q_network.input_attention.grad = total_grad
        #     optimizer_val.step()
        #
        #
        #
        #     print("val updat gama {}".format(gama))
        #     print("target attention {}".format(torch.sigmoid(target_q_network.input_attention).data))
        #     print("attention {}".format(torch.sigmoid(q_network.input_attention).data))
        #     print("grad attention L2 {}".format(atten_grad_L2_mean))
        #     print("grad attention loss {}".format(atten_grad_Loss_mean))
        #     print("total grad attention {}".format(total_grad))

        # total_loss = loss + gama*L2_dqn_out1
        # total_loss = loss
        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()

        total_num_steps = (ts + 1) * params.num_processes * params.task_steps * params.mini_batch_size

        # if ts == 100:
        #     gama = 100
        # if ts == 200:
        #     gama = 10
        # if ts == 2000:
        #     gama = 1

        if ts % params.target_network_update_f == 0:
            hard_update(agent.q_network, agent.target_q_network)

        if not params.debug and (ts % params.save_interval == 0 or ts == params.continue_from_epoch + num_updates- 1):
            torch.save(
                {'state_dict': q_network.state_dict(),'target_state_dict': target_q_network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'step': ts, 'obs_rms': getattr(utils.get_vec_normalize(envs), 'obs_rms', None)},
                os.path.join(logdir, params.env + "-epoch-{}.pt".format(ts)))

        # if not params.debug and (ts % params.save_grad == 0):
        #     if ts==0:
        #         torch.save({'shapes': shapes}, os.path.join(logdir_grad, params.env + "-epoch-{}-shapes.pt".format(ts)))
        #     torch.save({'grad_ens': grads},os.path.join(logdir_grad, params.val_env + "-epoch-{}-optimizer_grad.pt".format(ts, params.max_grad_sum)))

        if ts % params.log_every == 0:
            out_str = "Iter {}, Timestep {}".format(ts, total_num_steps)
            if len(episode_rewards) > 1:
                # out_str += ", Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".format(len(episode_rewards), np.mean(episode_rewards),
                #         np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards))
                # summary_writer.add_scalar(f'eval/train episode reward', np.mean(episode_rewards), ts)
                # summary_writer.add_scalar(f'eval/train episode len', np.mean(episode_len), ts)

                eval_r = {}
                for eval_disp_name, eval_env_name in EVAL_ENVS.items():
                    eval_r[eval_disp_name], eval_actions, num_uniform = evaluate(agent, eval_envs_dic, eval_disp_name, eval_locations_dic,
                                                      params.num_processes,
                                                      eval_env_name[1],
                                                      steps=params.task_steps,
                                                      recurrent=params.recurrent_policy, obs_recurrent=params.obs_recurrent,
                                                      multi_task=True, free_exploration=params.free_exploration)

                    if eval_disp_name == 'train_eval' and ts % 100 == 0:
                        print(eval_actions.data)
                    if ts == (params.continue_from_epoch + num_updates - 1):
                        print(eval_disp_name + ": {}".format(eval_actions.data))
                    if not params.debug:
                        summary_writer.add_scalar(f'eval/{eval_disp_name}', np.mean(eval_r[eval_disp_name]), total_num_steps)
                        summary_writer.add_scalar(f'eval_epoch/{eval_disp_name}', np.mean(eval_r[eval_disp_name]), ts)
                        summary_writer.add_scalar(f'entropy eval/{eval_disp_name}', num_uniform/eval_env_name[1], total_num_steps)
                        wandb.log({f'eval/{eval_disp_name}': np.mean(eval_r[eval_disp_name])}, step=total_num_steps)
                        wandb.log({f'eval_epoch/{eval_disp_name}': np.mean(eval_r[eval_disp_name])}, step=ts)
                        wandb.log({f'entropy eval/{eval_disp_name}': num_uniform/max(eval_env_name[1],params.num_processes)}, step=total_num_steps)
            if not params.debug:
                if len(losses) > 0:
                    out_str += ", TD Loss: {}".format(losses[-1])
                    summary_writer.add_scalar(f'losses/TD_loss', losses[-1], total_num_steps)
                    summary_writer.add_scalar(f'losses/TD_loss_epoch', losses[-1], ts)
                    wandb.log({f'losses/TD_loss': losses[-1]}, step=total_num_steps)
                    wandb.log({f'losses/TD_loss_epoch': losses[-1]}, step=ts)
                    # summary_writer.add_scalar(f'losses/val_TD_loss', val_losses[-1], total_num_steps)
                    # wandb.log({f'losses/val_TD_loss': val_losses[-1]}, step=total_num_steps)

                print(out_str)

                # summary_writer.add_scalar(f'Correlation/Corr_dqn_L2_out1', Corr_dqn_out1, total_num_steps)
                # summary_writer.add_scalar(f'Correlation/Corr_dqn_L2_out1_mean', Corr_dqn_out1_mean, total_num_steps)
                # summary_writer.add_scalar(f'Correlation/Corr_dqn_L2_grad', Corr_dqn_L2_grad, total_num_steps)
                # summary_writer.add_scalar(f'Correlation/Corr_dqn_L2_grad_mean', Corr_dqn_L2_grad_mean, total_num_steps)
                #
                # summary_writer.add_scalar(f'L2/L2_dqn_out1', L2_dqn_out1, total_num_steps)
                # summary_writer.add_scalar(f'L2/L2_dqn_out1_mean', L2_dqn_out1_mean, total_num_steps)
                #
                # wandb.log({f'Correlation/Corr_dqn_L2_out1': Corr_dqn_out1}, step=total_num_steps)
                # wandb.log({f'Correlation/Corr_dqn_L2_out1_mean': Corr_dqn_out1_mean}, step=total_num_steps)
                # wandb.log({f'Correlation/Corr_dqn_L2_grad': Corr_dqn_L2_grad}, step=total_num_steps)
                # wandb.log({f'Correlation/Corr_dqn_L2_grad_mean': Corr_dqn_L2_grad_mean}, step=total_num_steps)
                #
                # wandb.log({f'L2/L2_dqn_out1': L2_dqn_out1}, step=total_num_steps)
                # wandb.log({f'L2/L2_dqn_out1_mean': L2_dqn_out1_mean}, step=total_num_steps)

    if not params.debug:
        wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--load_env", type=str, default=None)
    parser.add_argument('--val-env', type=str, default=None)
    parser.add_argument("--num-processes", type=int, default=25, help='how many envs to use (default: 25)')
    parser.add_argument("--num-steps", type=int, default=5, help='number of forward steps in (default: 5)')
    parser.add_argument("--seed", type=int, default=1, help='random seed (default: 1)')
    parser.add_argument("--task_steps", type=int, default=20, help='number of steps in each task')
    parser.add_argument("--free_exploration", type=int, default=0, help='number of steps in each task without reward')
    parser.add_argument("--recurrent-policy", action='store_true', default=False, help='use a recurrent policy')
    parser.add_argument("--obs_recurrent", action='store_true', default=False, help='use a recurrent policy and observations input')
    parser.add_argument("--obs_rand_loc", action='store_true', default=False, help='use a recurrent policy and observations input with random reward position')
    parser.add_argument("--no_normalize", action='store_true', default=False, help='no normalize inputs')
    parser.add_argument("--rotate", action='store_true', default=False, help='rotate observations')
    parser.add_argument("--continue_from_epoch", type=int, default=0, help='load previous training (from model save dir) and continue')
    parser.add_argument("--saved_epoch", type=int, default=0, help='load previous training (from model save dir) and continue')
    parser.add_argument("--log-dir", default='/tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument("--save-dir", default='./trained_models/', help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument("--num-mini-batch", type=int, default=32, help='number of mini-batches (default: 32)')
    parser.add_argument("--mini-batch-size", type=int, default=32, help='size of mini-batches (default: 32)')
    parser.add_argument("--mini-batch-size-val", type=int, default=32, help='size of mini-batches (default: 32)')
    # parser.add_argument("--CnnDQN", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--learning_rate_val", type=float, default=0.1)
    # parser.add_argument("--target_update_rate", type=float, default=0.1)
    # parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--save-interval",type=int,default=1000, help='save interval, one save per n updates (default: 1000)')
    parser.add_argument("--save-grad",type=int,default=1000, help='save gradient, one save per n updates (default: 1000)')
    parser.add_argument("--max-grad-sum",type=int,default=1000, help='max gradient sum, one save per n updates (default: 1000)')
    parser.add_argument("--max-val-iter",type=int,default=50, help='maximum validation updates per step (default: 50)')
    parser.add_argument("--max_ts", type=int, default=1400000)
    # parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gamma_loss", type=float, default=10)
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
    parser.add_argument("--min_corr", type=float, default=0.9995)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--analytic', type=int, default=100)
    parser.add_argument('--k', type=int, default=0)
    main_dqn(parser.parse_args())
