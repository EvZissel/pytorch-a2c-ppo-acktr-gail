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

        q_value, _, rnn_hxs = self.q_network.forward(state, hidden_state, masks)
        if random.random() > epsilon:
            return q_value.max(1)[1].unsqueeze(-1), rnn_hxs
        return torch.tensor(np.random.randint(self.env.action_space.n, size=q_value.size()[0])).type(dtypelong).unsqueeze(-1), rnn_hxs


def compute_td_loss(agent, num_mini_batch, mini_batch_size, replay_buffer, optimizer, gamma, loss_var_coeff, device = "CPU", train=True, compute_analytic=False, same_ind=False, start_ind_array=None):
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
    states_all_all = 0
    # q_values_all_all = 0
    # exp_q_values_all_all = 0
    # actions_all_all = 0
    # states_all = 0
    for i in range(num_processes):
        all_losses.append(0)
        # all_grad_W2.append(0)
        # all_grad_b2.append(0)

    recurrent_hidden = torch.zeros(1, agent.q_network.recurrent_hidden_state_size).type(dtype)
    for start_ind in start_ind_array:
        data_sampler = replay_buffer.sampler(num_processes, start_ind, num_steps_per_batch)

        losses = []
        # grad_W2 = []
        # grad_b2 = []
        grad_L2_states = 0
        out1_states = 0
        states_all = 0
        # q_values_all = 0
        # exp_q_values_all = 0
        # actions_all = 0
        # states_all_b = 0

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
                states_all = states[:-1, :].unsqueeze(1)
                # q_values_all = q_values[:-1, :].unsqueeze(1)
                # exp_q_values_all = expected_q_value.unsqueeze(1)
                # actions_all = actions
            else:
                grad_L2_states = torch.cat((grad_L2_states, grad_L2_states_b), dim=1)
                out1_states = torch.cat((out1_states, out_1), dim=1)
                states_all = torch.cat((states_all, states[:-1, :].unsqueeze(1)), dim=1)
                # q_values_all = torch.cat((q_values_all, q_values[:-1, :].unsqueeze(1)), dim=1)
                # exp_q_values_all = torch.cat((exp_q_values_all, expected_q_value.unsqueeze(1)), dim=1)
                # actions_all = torch.cat((actions_all, actions), dim=1)

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
            states_all_all = states_all
            # q_values_all_all = q_values_all
            # exp_q_values_all_all = exp_q_values_all
            # actions_all_all = actions_all
        else:
            grad_L2_states_all = torch.cat((grad_L2_states_all, grad_L2_states.mean(1)), dim=0)
            out1_states_all = torch.cat((out1_states_all, out1_states), dim=0)
            states_all_all = torch.cat((states_all_all, states_all), dim=0)
            # q_values_all_all = torch.cat((q_values_all_all, q_values_all), dim=0)
            # exp_q_values_all_all = torch.cat((exp_q_values_all_all, exp_q_values_all), dim=0)
            # actions_all_all = torch.cat((actions_all_all, actions_all), dim=0)


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


    out_1_grad = torch.zeros((out1_states_all.size()[-1],agent.q_network.input_attention.size()[0])).type(dtype)
    # q_value_grad = torch.zeros((q_values_all_all.size()[-1],agent.q_network.input_attention.size()[0])).type(dtype)
    if compute_analytic:
        out1_states_all_flat = torch.flatten(out1_states_all, start_dim=0, end_dim=1)
        out1_states = out1_states_all_flat.mean(0)
        for i in range(out1_states.size()[0]):
            loss_i = out1_states[i]
            # loss_i.backward()
            loss_i_grads = torch.autograd.grad(loss_i,
                                                agent.q_network.input_attention_sig,
                                                retain_graph=True)
            out_1_grad[i,:] = loss_i_grads[0]

        # q_values_all_flat = torch.flatten(q_values_all_all, start_dim=0, end_dim=1)
        # q_values = q_values_all_flat.mean(0)
        # for i in range(q_values.size()[0]):
        #     q_loss_i = q_values[i]
        #     q_loss_i_grads = torch.autograd.grad(q_loss_i,
        #                                         agent.q_network.input_attention_sig,
        #                                         retain_graph=True)
        #     q_value_grad[i,:] = q_loss_i_grads[0]


    optimizer.zero_grad()

    if train:
        total_loss.backward()
        # loss_attn_grad = agent.q_network.input_attention_sig.grad * torch.sigmoid(agent.q_network.input_attention) * (1 - torch.sigmoid(agent.q_network.input_attention))
        optimizer.step()

    # grads = None
    # else:
    #     grads = torch.autograd.grad(total_loss,
    #                                 updated_train_params.values(),
    #                                 create_graph=create_graph)
    out1_states_all = out1_states_all.mean(1)

    return total_loss, grad_L2_states_all, out1_states_all, start_ind_array, out_1_grad


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
    if not params.debug:
        logdir_ = 'offline_soft_OL_out1_analytic_diffB_train_analytic' +  params.env + '_' + str(params.seed) + '_num_arms_' + str(params.num_processes) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        if params.rotate:
            logdir_ = logdir_ + '_rotate'
        if params.zero_ind:
            logdir_ = logdir_ + '_Zero_ind'

        logdir = os.path.join('dqn_runs_offline', logdir_)
        logdir = os.path.join(os.path.expanduser(params.log_dir), logdir)
        utils.cleanup_log_dir(logdir)
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_hparams(vars(params), {})

        wandb.init(project="main_dqn_offline_only_analytic", entity="ev_zisselman", config=params, name=logdir_, id=logdir_)

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

    val_envs = make_vec_envs(params.val_env, params.seed, params.num_processes, eval_locations_dic['valid_eval'],
                         params.gamma, None, device, False, steps=params.task_steps,
                         free_exploration=params.free_exploration, recurrent=params.recurrent_policy,
                         obs_recurrent=params.obs_recurrent, multi_task=True, normalize=not params.no_normalize, rotate=params.rotate, obs_rand_loc=params.obs_rand_loc)

    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        eval_envs_dic[eval_disp_name] = make_vec_envs(eval_env_name[0], params.seed, params.num_processes, eval_locations_dic[eval_disp_name],
                                                      None, None, device, True, steps=params.task_steps,
                                                      recurrent=params.recurrent_policy,
                                                      obs_recurrent=params.obs_recurrent, multi_task=True,
                                                      free_exploration=params.free_exploration, normalize=not params.no_normalize, rotate=params.rotate, obs_rand_loc=params.obs_rand_loc)

    q_network = DQN_RNNLast_analytic(envs.observation_space.shape, envs.action_space.n, params.zero_ind, recurrent=True, hidden_size=params.hidden_size)
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
    optimizer_val = optim.Adam(attention_parameters, lr=params.learning_rate_val, weight_decay=params.weight_decay)
    # scheduler = MultiStepLR(optimizer_val, milestones=[30, 80], gamma=0.1)

    replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)
    # grad_replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)
    val_replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)

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

    val_obs = val_envs.reset()
    val_replay_buffer.obs[0].copy_(val_obs)
    # val_replay_buffer.to(device)

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

    # Collect validation data
    for step in range(params.num_steps):

        # val_actions = torch.tensor(np.random.randint(agent.num_actions, size=params.num_processes)).type(dtypelong).unsqueeze(-1)
        val_actions = torch.tensor(np.random.randint(agent.num_actions) * np.ones(params.num_processes)).type(dtypelong).unsqueeze(-1).cpu()

        val_next_obs, val_reward, val_done, val_infos = val_envs.step(val_actions.cpu())


        for info in val_infos:
            if 'episode' in info.keys():
                val_episode_rewards.append(info['episode']['r'])
                val_episode_len.append(info['episode']['l'])

        val_masks = torch.FloatTensor(
            [[0.0] if val_done_ else [1.0] for val_done_ in val_done])

        val_replay_buffer.insert(val_next_obs, val_actions, val_reward, val_masks)

    # val_replay_buffer.copy(replay_buffer)
    # # replay_buffer.obs[:,:,:-2] = obs[:,:-2].repeat([replay_buffer.obs.size(0),1,1])
    # val_replay_buffer.obs[:,:,:-2] = val_obs[:,:-2].repeat([replay_buffer.obs.size(0),1,1])

    # Training
    num_updates = int(
        params.max_ts  // params.num_processes // params.task_steps // params.mini_batch_size)

    min_corr = torch.tensor(params.min_corr)

    # alpha = params.loss_corr_coeff_update
    # loss_corr_coeff = params.loss_corr_coeff

    # # compute validation gradient for start_ind_array
    # loss, grads_L2, out1, start_ind_array = compute_td_loss(
    #     agent, params.num_mini_batch, params.mini_batch_size_val, replay_buffer, optimizer, params.gamma, params.loss_var_coeff, device, train=False,
    # )

    out1_vec = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    out1_val_vec = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    L2_grad_vec = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    L2_grad_val_vec = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    out1_grad_vec = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    out1_grad_val_vec = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))

    # Q_value = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    # Q_value_val = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    # Q_value_gard = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    # Q_value_gard_val = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    #
    # exp_q_values_vac = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    # exp_q_values_val_vac = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    # actions_vac = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))
    # actions_val_vac = deque(maxlen=int(params.mini_batch_size_val/params.mini_batch_size))


    # m = 0
    # v = 0
    #
    # m_test = 0
    # v_test = 0
    #
    # beta1 = 0.9
    # beta2 = 0.999
    # epsilon = 1e-8
    # t=1
    gama = 100
    lr_val = 0.1


    for ts in range(params.continue_from_epoch, params.continue_from_epoch+num_updates):
        # Update the q-network & the target network

        #### Update Theta #####
        loss, grads_L2, out1, start_ind_array, out1_grad = compute_td_loss(
            agent, params.num_mini_batch, params.mini_batch_size, replay_buffer, optimizer, params.gamma, params.loss_var_coeff, device, train=True, compute_analytic=True,
        )
        losses.append(loss.data)

        val_loss, val_grads_L2, val_out1, _, out1_grad_val = compute_td_loss(
            agent, params.num_mini_batch, params.mini_batch_size, val_replay_buffer, optimizer, params.gamma, params.loss_var_coeff, device, train=False, compute_analytic=True,
            same_ind=True, start_ind_array=start_ind_array
        )
        val_losses.append(val_loss.data)

        # Corr_dqn_L2_grad = 0
        # for i in range(grads_L2.size(0)):
        #     Corr_dqn_L2_grad += ((grads_L2[i] * val_grads_L2[i]).sum() / (grads_L2[i].norm(2) * val_grads_L2[i].norm(2)))/grads_L2.size(0)
        L2_grad_vec.append(grads_L2)
        L2_grad_val_vec.append(val_grads_L2)
        L2_grad_vec_mean =  (sum(L2_grad_vec)/len(L2_grad_vec)).mean(0)
        L2_grad_val_vec_mean =  (sum(L2_grad_val_vec)/len(L2_grad_val_vec)).mean(0)
        train_grads_L2 = grads_L2.mean(0)
        val_grads_L2 = val_grads_L2.mean(0)
        Corr_dqn_L2_grad = (train_grads_L2 * val_grads_L2).sum() / (train_grads_L2.norm(2) * val_grads_L2.norm(2))
        Corr_dqn_L2_grad_mean = (L2_grad_vec_mean * L2_grad_val_vec_mean).sum() / (L2_grad_vec_mean.norm(2) * L2_grad_val_vec_mean.norm(2))
        print("correlation L2 grad: {}".format(Corr_dqn_L2_grad))
        print("correlation L2 grad mean: {}".format(Corr_dqn_L2_grad_mean))


        out1_vec.append(out1)
        out1_val_vec.append(val_out1)
        out1_vec_mean =  (sum(out1_vec)/len(out1_vec)).mean(0)
        out1_val_vec_mean =  (sum(out1_val_vec)/len(out1_val_vec)).mean(0)
        train_out1_Corr = out1.mean(0)
        val_out1_Corr = val_out1.mean(0)
        Corr_dqn_out1 = (train_out1_Corr * val_out1_Corr).sum() / (train_out1_Corr.norm(2) * val_out1_Corr.norm(2))
        print("correlation out1: {}".format(Corr_dqn_out1))


        Corr_dqn_out1_mean = (out1_vec_mean*out1_val_vec_mean).sum() / (out1_vec_mean.norm(2) * out1_val_vec_mean.norm(2))
        print("correlation out1 mean: {}".format(Corr_dqn_out1_mean))
        L2_dqn_out1 = ((train_out1_Corr - val_out1_Corr)**2).sum()
        print("L2 out1: {}".format(L2_dqn_out1))
        L2_dqn_out1_mean = ((out1_vec_mean - out1_val_vec_mean)**2).sum()
        print("L2 out1 mean: {}".format(L2_dqn_out1_mean))

        out1_grad_vec.append(out1_grad)
        out1_grad_val_vec.append(out1_grad_val)

        # Q_value.append(q_values)
        # Q_value_val.append(q_values_val)
        # Q_value_gard.append(q_value_grad)
        # Q_value_gard_val.append(q_value_grad_val)
        #
        # exp_q_values_vac.append(exp_q_values)
        # exp_q_values_val_vac.append(exp_q_values_val)
        # actions_vac.append(actions)
        # actions_val_vac.append(actions_val)



        out1_vec_attn = deque(maxlen=10)
        out1_val_vec_attn = deque(maxlen=10)
        iter = 0
        while (Corr_dqn_out1_mean < min_corr) and (iter < params.max_val_iter):
            iter+=1
            if iter==1:
                out1_stack = torch.flatten(torch.stack(list(out1_vec)),0,1)
                out1_val_stack = torch.flatten(torch.stack(list(out1_val_vec)),0,1)
                out1_grad_stack = torch.flatten(torch.stack(list(out1_grad_vec)),0,1)
                out1_grad_val_stack = torch.flatten(torch.stack(list(out1_grad_val_vec)),0,1)

                # Q_value_stack = torch.flatten(torch.stack(list(Q_value)),0,1)
                # Q_value_val_stack = torch.flatten(torch.stack(list(Q_value_val)),0,1)
                # Q_value_gard_stack = torch.flatten(torch.stack(list(Q_value_gard)),0,1)
                # Q_value_gard_val_stack = torch.flatten(torch.stack(list(Q_value_gard_val)),0,1)

                # exp_q_values_stack = torch.flatten(torch.stack(list(exp_q_values_vac)),0,1)
                # exp_q_values_val_stack = torch.flatten(torch.stack(list(exp_q_values_val_vac)),0,1)
                # actions_stack = torch.flatten(torch.stack(list(actions_vac)),0,1)
                # actions_val_stack = torch.flatten(torch.stack(list(actions_val_vac)),0,1)

                analytic_input_attention = agent.q_network.input_attention.data

                out1_analytic = out1_stack.mean(0)
                out1_val_analytic = out1_val_stack.mean(0)

                # Q_value_analytic = Q_value_stack.mean(0)
                # Q_value_val_analytic = Q_value_val_stack.mean(0)

            else:
                out1_analytic = out1_analytic-torch.matmul(out1_grad_stack,grad_atten_analytic.unsqueeze(1)).squeeze()
                out1_val_analytic = out1_val_analytic-torch.matmul(out1_grad_val_stack,grad_atten_analytic.unsqueeze(1)).squeeze()

                # Q_value_analytic = Q_value_analytic-lr_val*torch.matmul(out1_grad_stack,analytic_input_attention.unsqueeze(1)).squeeze()
                # Q_value_val_analytic = Q_value_val_analytic-lr_val*torch.matmul(out1_grad_val_stack,analytic_input_attention.unsqueeze(1)).squeeze()

            out1_vec_attn.append(out1_analytic)
            out1_val_vec_attn.append(out1_val_analytic)
            out1_vec_mean = sum(out1_vec_attn) / len(out1_vec_attn)
            out1_val_vec_mean = sum(out1_val_vec_attn) / len(out1_val_vec_attn)
            Corr_dqn_out1_mean = (out1_vec_mean * out1_val_vec_mean).sum() / (out1_vec_mean.norm(2) * out1_val_vec_mean.norm(2))

            grad_sig_atten_analytic = torch.matmul((out1_analytic - out1_val_analytic).unsqueeze(0),(out1_grad_stack - out1_grad_val_stack)).squeeze()
            grad_atten_analytic = grad_sig_atten_analytic * torch.sigmoid(analytic_input_attention) * (1 - torch.sigmoid(analytic_input_attention))

            # grad_sig_loss_analytic = (actions_stack*Q_value_stack - exp_q_values_stack)*actions_stack*Q_value_gard_stack
            # grad_loss_analytic = grad_sig_loss_analytic * torch.sigmoid(analytic_input_attention) * (1 - torch.sigmoid(analytic_input_attention))

            # m = beta1*m + (1-beta1)*grad_atten_analytic
            # v = beta2*v + (1-beta2)*(grad_atten_analytic**2)
            #
            # m_hat = m/(1-beta1**t)
            # v_hat = v/(1-beta2**t)
            #
            # analytic_input_attention = analytic_input_attention - params.learning_rate_val*m_hat/(torch.sqrt(v_hat)+ epsilon)

            # m_test = beta1*m_test + (1-beta1)*grad_atten_pytoch
            # v_test = beta2*v_test + (1-beta2)*((grad_atten_pytoch)**2)
            # m_hat_test = m_test/(1-beta1**t)
            # v_hat_test = v_test/(1-beta2**t)
            # t+=1

            # pytorch_input_attention_test = agent.q_network.input_attention.data - params.learning_rate_val * m_hat_test / (torch.sqrt(v_hat_test) + epsilon)
            grad_loss = 0
            if (gama < 100):
                loss, grads_L2, out1, start_ind_array, out1_grad = compute_td_loss(
                    agent, params.num_mini_batch, params.mini_batch_size, replay_buffer, optimizer, params.gamma, params.loss_var_coeff, device, train=True, compute_analytic=True)

                grad_loss = torch.autograd.grad(loss,
                                                agent.q_network.input_attention_sig,
                                                retain_graph=True)
                grad_loss = grad_loss * torch.sigmoid(analytic_input_attention) * (1 - torch.sigmoid(analytic_input_attention))
            optimizer_val.zero_grad()
            agent.q_network.input_attention.grad = grad_loss + gama*grad_atten_analytic.detach()
            # grad_loss_attn_pytoch = grad_sig_loss_attn_pytorch[0]*(torch.sigmoid(agent.q_network.input_attention)*(1-torch.sigmoid(agent.q_network.input_attention)))
            optimizer_val.step()





            print("val iter {} update attention out1 correlation {}".format(iter, Corr_dqn_out1_mean))
            print("target attention {}".format(torch.sigmoid(target_q_network.input_attention).data))
            print("attention {}".format(torch.sigmoid(q_network.input_attention).data))
            # print("python attention test {}".format(torch.sigmoid(pytorch_input_attention_test).data))
            print("analytic attention {}".format(torch.sigmoid(analytic_input_attention).data))
            # print("grad sig attention analytic {}".format(grad_sig_atten_analytic))
            # print("grad attention pytorch {}".format(grad_atten_pytoch))
            # print("grad sig attention pytorch {}".format(grad_sig_atten_pytoch[0]))
            print("grad attention analytic {}".format(grad_atten_analytic))
            # print("grad sig attention analytic {}".format(grad_sig_atten_analytic))

            # print("grad loss attention pytorch {}".format(grad_loss_attn_pytorch))
            # print("grad loss attention pytorch {}".format(grad_loss_attn_pytoch))
            # print("minus_out1_Corr_dqn_L2_grad {}".format( -loss_corr_coeff*attn_out1_Corr_dqn_L2_grad[0]))




        total_num_steps = (ts + 1) * params.num_processes * params.task_steps * params.mini_batch_size
        if ts == 100:
            gama = 10
        if ts == 200:
            gama = 1

        if ts % params.target_network_update_f == 0:
            hard_update(agent.q_network, agent.target_q_network)

        if not params.debug and (ts % params.save_interval == 0 or ts == params.continue_from_epoch + num_updates- 1):
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

                    if not params.debug:
                        summary_writer.add_scalar(f'eval/{eval_disp_name}', np.mean(eval_r[eval_disp_name]), total_num_steps)
                        wandb.log({f'eval/{eval_disp_name}': np.mean(eval_r[eval_disp_name])}, step=total_num_steps)
            if not params.debug:
                if len(losses) > 0:
                    out_str += ", TD Loss: {}".format(losses[-1])
                    summary_writer.add_scalar(f'losses/TD_loss', losses[-1], total_num_steps)
                    wandb.log({f'losses/TD_loss': losses[-1]}, step=total_num_steps)
                    summary_writer.add_scalar(f'losses/val_TD_loss', val_losses[-1], total_num_steps)
                    wandb.log({f'losses/val_TD_loss': val_losses[-1]}, step=total_num_steps)

                print(out_str)

                summary_writer.add_scalar(f'Correlation/Corr_dqn_L2_out1', Corr_dqn_out1, total_num_steps)
                summary_writer.add_scalar(f'Correlation/Corr_dqn_L2_out1_mean', Corr_dqn_out1_mean, total_num_steps)
                summary_writer.add_scalar(f'Correlation/Corr_dqn_L2_grad', Corr_dqn_L2_grad, total_num_steps)
                summary_writer.add_scalar(f'Correlation/Corr_dqn_L2_grad_mean', Corr_dqn_L2_grad_mean, total_num_steps)

                summary_writer.add_scalar(f'L2/L2_dqn_out1', L2_dqn_out1, total_num_steps)
                summary_writer.add_scalar(f'L2/L2_dqn_out1_mean', L2_dqn_out1_mean, total_num_steps)

                wandb.log({f'Correlation/Corr_dqn_L2_out1': Corr_dqn_out1}, step=total_num_steps)
                wandb.log({f'Correlation/Corr_dqn_L2_out1_mean': Corr_dqn_out1_mean}, step=total_num_steps)
                wandb.log({f'Correlation/Corr_dqn_L2_grad': Corr_dqn_L2_grad}, step=total_num_steps)
                wandb.log({f'Correlation/Corr_dqn_L2_grad_mean': Corr_dqn_L2_grad_mean}, step=total_num_steps)

                wandb.log({f'L2/L2_dqn_out1': L2_dqn_out1}, step=total_num_steps)
                wandb.log({f'L2/L2_dqn_out1_mean': L2_dqn_out1_mean}, step=total_num_steps)

    if not params.debug:
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
    parser.add_argument("--obs_rand_loc", action='store_true', default=False, help='use a recurrent policy and observations input with random reward position')
    parser.add_argument("--no_normalize", action='store_true', default=False, help='no normalize inputs')
    parser.add_argument("--rotate", action='store_true', default=False, help='rotate observations')
    parser.add_argument("--continue_from_epoch", type=int, default=0, help='load previous training (from model save dir) and continue')
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
    # parser.add_argument("--save-grad",type=int,default=1000, help='save gradient, one save per n updates (default: 1000)')
    parser.add_argument("--max-grad-sum",type=int,default=1000, help='max gradient sum, one save per n updates (default: 1000)')
    parser.add_argument("--max-val-iter", type=int, default=50, help='maximum validation updates per step (default: 50)')
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
    parser.add_argument("--min_corr", type=float, default=0.9995)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--analytic', type=int, default=100)
    main_dqn(parser.parse_args())
