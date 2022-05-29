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
import math
import random
from copy import deepcopy
import os
import time
from collections import deque

import numpy as np
import torch
import torch.optim as optim
from helpers_dqn import ReplayBuffer, make_atari, make_gym_env, wrap_deepmind, wrap_pytorch, ReplayBufferBandit
from models_dqn import DQN, CnnDQN, DQN_softAttn
from a2c_ppo_acktr.envs import make_vec_envs
from torch.utils.tensorboard import SummaryWriter
from a2c_ppo_acktr import utils
import pandas as pd
import itertools
from collections import OrderedDict

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

        q_value, rnn_hxs = self.q_network.forward(state, hidden_state, masks)
        if random.random() > epsilon:
            return q_value.max(1)[1].unsqueeze(-1), rnn_hxs
        return torch.tensor(np.random.randint(self.env.action_space.n, size=q_value.size()[0])).type(dtypelong).unsqueeze(-1), rnn_hxs


def compute_td_loss(agent, num_mini_batch, mini_batch_size, replay_buffer, optimizer, gamma, loss_var_coeff, train=True, create_graph=False):
    num_processes = replay_buffer.rewards.size(1)
    num_steps = replay_buffer.rewards.size(0)
    num_steps_per_batch = int(num_steps/num_mini_batch)

    # all_losses =[]
    start_ind_array = [i for i in range(0, num_steps, num_steps_per_batch)]
    # start_ind_array = random.choices(start_ind_array, k=mini_batch_size)
    start_ind_array = np.random.choice(start_ind_array, size=mini_batch_size, replace=False)
    # start_ind = random.choices(start_ind_array, k=1)

    all_losses = []
    for i in range(num_processes):
        all_losses.append(0)

    for start_ind in start_ind_array:
        data_sampler = replay_buffer.sampler(num_processes, start_ind, num_steps_per_batch)

        losses = []
        recurrent_hidden = torch.zeros(1, agent.q_network.recurrent_hidden_state_size).type(dtypelong)
        for states, actions, rewards, done in data_sampler:

            # double q-learning
            with torch.no_grad():
                    # recurrent_hidden.detach()
                online_q_values, _ = agent.q_network(states, recurrent_hidden, done)
                _, max_indicies = torch.max(online_q_values, dim=1)
                target_q_values, _ = agent.target_q_network(states, recurrent_hidden, done)
                next_q_value = target_q_values.gather(1, max_indicies.unsqueeze(1))

                next_q_value = next_q_value * done
                expected_q_value = (rewards + gamma * next_q_value[1:, :]).squeeze(1)

            # Normal DDQN update
            q_values, _ = agent.q_network(states, recurrent_hidden, done)
            q_value = q_values[:-1, :].gather(1, actions).squeeze(1)

            losses.append((q_value - expected_q_value.data).pow(2).mean())

        # loss = torch.stack(losses)
        # loss = losses.mean(0)
        # all_losses.append(loss)
        for i in range(num_processes):
            all_losses[i] += losses[i]/mini_batch_size

    total_loss = torch.stack(all_losses)
    total_loss = total_loss.mean() + loss_var_coeff * total_loss.var()
    # optimizer.zero_grad()
    # total_loss.backward()
    # optimizer.step()

    # return total_loss, all_losses.mean(), all_losses.var()
    updated_train_params = OrderedDict()
    updated_val_params = OrderedDict()
    for name, p in agent.q_network.named_parameters():
        if 'attention' in name:
            # p.requires_grad = False
            updated_val_params[name] = p
        else:
            # p.requires_grad = True
            updated_train_params[name] = p

    optimizer.zero_grad()

    if train:
        total_loss.backward()
        optimizer.step()
        grads = None
    else:
        grads = torch.autograd.grad(total_loss,
                                    updated_train_params.values(),
                                    create_graph=create_graph)
    return total_loss, grads


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

    logdir = 'offline_soft_' +  params.env + '_' + str(params.seed) + '_num_arms_' + str(params.num_processes)+ '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    if params.rotate:
        logdir = logdir + '_rotate'
    if params.zero_ind:
        logdir = logdir + '_Zero_ind'

    logdir = os.path.join('dqn_runs_offline', logdir)
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

    q_network = DQN_softAttn(envs.observation_space.shape, envs.action_space.n, params.zero_ind, recurrent=True, hidden_size=params.hidden_size)
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

    replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)
    grad_replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)
    val_replay_buffer = ReplayBufferBandit(params.num_steps, params.num_processes, envs.observation_space.shape, envs.action_space)

    # Load previous model
    if (params.continue_from_epoch > 0) and params.save_dir != "":
        save_path = params.save_dir
        q_network_weighs = torch.load(os.path.join(save_path, params.env + "-epoch-{}.pt".format(params.continue_from_epoch)), map_location=device)
        agent.q_network.load_state_dict(q_network_weighs['state_dict'])
        hard_update(agent.q_network, agent.target_q_network)
        optimizer.load_state_dict(q_network_weighs['optimizer_state_dict'])


    obs = envs.reset()
    replay_buffer.obs[0].copy_(obs)
    replay_buffer.to(device)

    val_obs = val_envs.reset()
    val_replay_buffer.obs[0].copy_(val_obs)
    val_replay_buffer.to(device)

    episode_rewards = deque(maxlen=25)
    episode_len = deque(maxlen=25)
    val_episode_rewards = deque(maxlen=25)
    val_episode_len = deque(maxlen=25)

    losses = []
    grad_losses = []
    val_losses = []

    # Collect train data
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

    # Collect train data for gradient calculation
    grad_replay_buffer.obs[0].copy_(next_obs)
    grad_replay_buffer.to(device)

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

        grad_replay_buffer.insert(next_obs, actions, reward, masks)

    # Collect validation data
    for step in range(params.num_steps):

        val_actions = torch.tensor(np.random.randint(agent.num_actions, size=params.num_processes)).type(dtypelong).unsqueeze(-1)

        val_next_obs, val_reward, val_done, val_infos = val_envs.step(val_actions.cpu())


        for info in val_infos:
            if 'episode' in info.keys():
                val_episode_rewards.append(info['episode']['r'])
                val_episode_len.append(info['episode']['l'])

        val_masks = torch.FloatTensor(
            [[0.0] if val_done_ else [1.0] for val_done_ in val_done])

        val_replay_buffer.insert(val_next_obs, val_actions, val_reward, val_masks)

    # Training
    num_updates = int(
        params.max_ts  // params.num_processes // params.task_steps // params.mini_batch_size)


    # grad_sumQ = deque(maxlen=params.max_grad_sum)
    # grad_grad_sumQ = deque(maxlen=params.max_grad_sum)
    # grad_sumQ_val = deque(maxlen=params.max_grad_sum)
    #
    # Corr_all_grad_vec = deque(maxlen=params.max_grad_sum)
    # Corr_gru_grad_vec = deque(maxlen=params.max_grad_sum)
    # Corr_dqn_grad_vec = deque(maxlen=params.max_grad_sum)
    min_corr = 0.9
    # max_loss = 0.3

    # grad_sumQ = deque(maxlen=params.max_grad_sum)
    grad_grad_sumQ = deque(maxlen=params.max_grad_sum)
    grad_sumQ_val = deque(maxlen=params.max_grad_sum)
    dqn_grad_L2_sumQ = deque(maxlen=params.max_grad_sum)
    dqn_val_L2_sumQ = deque(maxlen=params.max_grad_sum)
    alpha = params.loss_corr_coeff_update

    Corr_all_grad_vec = deque(maxlen=params.max_grad_sum)
    Corr_dqn_L2_grad_vec = deque(maxlen=params.max_grad_sum)
    # Corr_gru_grad_vec = deque(maxlen=params.max_grad_sum)
    # Corr_dqn_grad_vec = deque(maxlen=params.max_grad_sum)

    for ts in range(params.continue_from_epoch, params.continue_from_epoch+num_updates):
        # Update the q-network & the target network

        #### Update Theta #####
        loss, _ = compute_td_loss(
            agent, params.num_mini_batch, params.mini_batch_size, replay_buffer, optimizer, params.gamma, params.loss_var_coeff, train=True,
        )
        losses.append(loss.data)

        #### Update Phi #####
        grad_loss, grad_grads = compute_td_loss(
            agent, params.num_mini_batch, params.mini_batch_size_val, grad_replay_buffer, optimizer, params.gamma, params.loss_var_coeff, train=False,
        )
        grad_losses.append(grad_loss.data)

        # compute validation gradient
        val_loss, val_grads = compute_td_loss(
            agent, params.num_mini_batch, params.mini_batch_size_val, val_replay_buffer, optimizer, params.gamma, params.loss_var_coeff, train=False,
        )
        val_losses.append(val_loss.data)

        # mean_grad = _flatten_grad(grads)
        grad_mean_grad = _flatten_grad(grad_grads)
        val_mean_grad = _flatten_grad(val_grads)

        dqn_grad_L2 = _flatten_grad(grad_grads[6:])
        dqn_val_L2 = _flatten_grad(val_grads[6:])

        # grad_sumQ.append(mean_grad)
        grad_grad_sumQ.append(grad_mean_grad)
        grad_sumQ_val.append(val_mean_grad)

        dqn_grad_L2_sumQ.append(dqn_grad_L2)
        dqn_val_L2_sumQ.append(dqn_val_L2)

        # mean_flat_grad = sum(grad_sumQ)
        mean_grad_grad = sum(grad_grad_sumQ)
        mean_grad_val = sum(grad_sumQ_val)

        mean_dqn_grad_L2 = sum(dqn_grad_L2_sumQ)
        mean_dqn_val_L2 = sum(dqn_val_L2_sumQ)

        # mean_grad = optimizer._unflatten_grad(mean_flat_grad, shapes)
        # mean_grad_grad = optimizer._unflatten_grad(mean_flat_grad_grad, grad_shapes)
        # mean_grad_val = optimizer._unflatten_grad(mean_flat_grad_val, val_shapes)

        # # mean_gru = _flatten_grad(grads[0:4])
        # mean_gru_grad = _flatten_grad(grad_grads[0:4])
        # mean_gru_val = _flatten_grad(val_grads[0:4])
        # # mean_dqn = _flatten_grad(grads[4:])
        # mean_dqn_grad = _flatten_grad(grad_grads[4:])
        # mean_dqn_val = _flatten_grad(val_grads[4:])

        # Corr_all = (mean_grad * val_mean_grad).sum() / (mean_grad.norm(2) * val_mean_grad.norm(2))
        # Corr_gru = (mean_gru * mean_gru_val).sum() / (mean_gru.norm(2) * mean_gru_val.norm(2))
        # Corr_dqn = (mean_dqn * mean_dqn_val).sum() / (mean_dqn.norm(2) * mean_dqn_val.norm(2))

        Mean_Corr_all_grad = (mean_grad_grad * mean_grad_val).sum() / (mean_grad_grad.norm(2) * mean_grad_val.norm(2))
        Mean_Corr_dqn_L2_grad = (mean_dqn_grad_L2 * mean_dqn_val_L2).sum() / (mean_dqn_grad_L2.norm(2) * mean_dqn_val_L2.norm(2))

        # Corr_all_grad = (grad_mean_grad * val_mean_grad).sum() / (grad_mean_grad.norm(2) * val_mean_grad.norm(2))
        # Corr_gru_grad = (mean_gru_grad * mean_gru_val).sum() / (mean_gru_grad.norm(2) * mean_gru_val.norm(2))
        # Corr_dqn_grad = (mean_dqn_grad * mean_dqn_val).sum() / (mean_dqn_grad.norm(2) * mean_dqn_val.norm(2))

        Corr_all_grad_vec.append(Mean_Corr_all_grad)
        Corr_dqn_L2_grad_vec.append(Mean_Corr_dqn_L2_grad)
        # Corr_gru_grad_vec.append(Corr_gru_grad)
        # Corr_dqn_grad_vec.append(Corr_dqn_grad)

        # # if Mean_Corr_all_grad < min_corr:
        if ((sum(Corr_all_grad_vec) / len(Corr_all_grad_vec)) < min_corr) and (mean_grad_grad.norm(2) > 0.0001) and (mean_grad_val.norm(2) > 0.0001):
            loss_corr_coeff = params.loss_corr_coeff


            grad_grad_sumQ_attn = deque(maxlen=params.max_grad_sum)
            grad_sumQ_val_attn = deque(maxlen=params.max_grad_sum)
            grad_grad_L2_sumQ_attn = deque(maxlen=params.max_grad_sum)
            grad_sumQ_L2_val_attn = deque(maxlen=params.max_grad_sum)

            # grad_grad_sumQ_attn.append(grad_mean_grad)
            # grad_sumQ_val_attn.append(val_mean_grad)
            # mean_grad_grad_attn = sum(grad_grad_sumQ_attn)
            # mean_grad_val_attn = sum(grad_sumQ_val_attn)
            #
            # Corr_attn_grad = (mean_grad_grad_attn * mean_grad_val_attn).sum() / (mean_grad_grad_attn.norm(2) * mean_grad_val_attn.norm(2))

            Corr_L2_mean_attn_grad = Mean_Corr_dqn_L2_grad
            # for i in range(params.num_val_updates):
            while (Corr_L2_mean_attn_grad < 0.9):

                # compute grad gradient
                grad_loss, grad_grads = compute_td_loss(
                    agent, params.num_mini_batch, params.mini_batch_size_val, grad_replay_buffer, optimizer,
                    params.gamma, params.loss_var_coeff, train=False, create_graph=True
                )
                grad_losses.append(grad_loss.data)

                # compute validation gradient
                val_loss, val_grads = compute_td_loss(
                    agent, params.num_mini_batch, params.mini_batch_size_val, val_replay_buffer, optimizer,
                    params.gamma, params.loss_var_coeff, train=False, create_graph=True
                )
                val_losses.append(val_loss.data)

                # mean_grad = _flatten_grad(grads)
                grad_mean_grad = _flatten_grad(grad_grads)
                val_mean_grad = _flatten_grad(val_grads)

                dqn_grad_L2 = _flatten_grad(grad_grads[6:])
                dqn_val_L2 = _flatten_grad(val_grads[6:])

                Corr_all_grad = (grad_mean_grad * val_mean_grad).sum() / (grad_mean_grad.norm(2) * val_mean_grad.norm(2))
                Corr_dqn_L2_grad = (dqn_grad_L2 * dqn_val_L2).sum() / (dqn_grad_L2.norm(2) * dqn_val_L2.norm(2))

                grad_grad_sumQ_attn.append(grad_mean_grad)
                grad_sumQ_val_attn.append(val_mean_grad)

                grad_grad_L2_sumQ_attn.append(dqn_grad_L2)
                grad_sumQ_L2_val_attn.append(dqn_val_L2)

                mean_grad_grad_attn = sum(grad_grad_sumQ_attn)/len(grad_grad_sumQ_attn)
                mean_grad_val_attn = sum(grad_sumQ_val_attn)/len(grad_sumQ_val_attn)

                mean_grad_grad_L2_attn = sum(grad_grad_L2_sumQ_attn)/len(grad_grad_L2_sumQ_attn)
                mean_grad_val_L2_attn = sum(grad_sumQ_L2_val_attn)/len(grad_sumQ_L2_val_attn)

                Corr_mean_attn_grad = (mean_grad_grad_attn * mean_grad_val_attn).sum() / (mean_grad_grad_attn.norm(2) * mean_grad_val_attn.norm(2))
                Corr_L2_mean_attn_grad = (mean_grad_grad_L2_attn * mean_grad_val_L2_attn).sum() / (mean_grad_grad_L2_attn.norm(2) * mean_grad_val_L2_attn.norm(2))


                # val_loss = grad_loss - loss_corr_coeff*Corr_all_grad
                val_loss = grad_loss - loss_corr_coeff*Corr_dqn_L2_grad
                loss_corr_coeff = loss_corr_coeff * alpha
                # val_loss = params.loss_corr_coeff * Corr_all_grad

                # if (first_time == 0):
                updated_train_params = OrderedDict()
                updated_val_params = OrderedDict()
                for name, p in agent.q_network.named_parameters():
                    if 'attention' in name:
                        # p.requires_grad = True
                        updated_val_params[name] = p
                    else:
                        # p.requires_grad = False
                        updated_train_params[name] = p


                optimizer_val.zero_grad()
                # val_loss.backward()
                attn_grads = torch.autograd.grad(val_loss,
                                                updated_val_params.values(),
                                                create_graph=False)
                agent.q_network.input_attention._grad = attn_grads[0]
                optimizer_val.step()

                # update target network attention
                if params.update_target:
                    agent.target_q_network.input_attention.data = agent.q_network.input_attention.data

                # if i == params.num_val_updates-1:
                print("update attention, attn corr {}, mean attn corr {}, grad loss {}, loss_corr_coeff {}, attn_grads norm {}".format(Corr_all_grad, Corr_mean_attn_grad, grad_loss, loss_corr_coeff,attn_grads[0].norm(2)))
                print("target attention {}".format(torch.sigmoid(target_q_network.input_attention).data))

                # if (len(grad_grad_sumQ_attn) == params.max_grad_sum) and (Corr_mean_attn_grad > min_corr):
                #     break



        total_num_steps = (ts + 1) * params.num_processes * params.task_steps * params.mini_batch_size

        if ts % params.target_network_update_f == 0:
            hard_update(agent.q_network, agent.target_q_network)

        if (ts % params.save_interval == 0 or ts == params.continue_from_epoch + num_updates- 1):
            torch.save(
                {'state_dict': q_network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
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
            if len(losses) > 0:
                out_str += ", TD Loss: {}".format(losses[-1])
                summary_writer.add_scalar(f'losses/TD_loss', losses[-1], total_num_steps)
                summary_writer.add_scalar(f'losses/val_TD_loss', val_losses[-1], total_num_steps)
                summary_writer.add_scalar(f'losses/grad_TD_loss', grad_losses[-1], total_num_steps)
                # summary_writer.add_scalar(f'losses/mean_loss', mean_loss, total_num_steps)
                # summary_writer.add_scalar(f'losses/var_loss', var_loss, total_num_steps)
                # summary_writer.add_scalars(f'GradNorms/F_norm', {'env {}'.format(i): F_norms_all[i] for i in range(len(F_norms_all))}, total_num_steps)
                # summary_writer.add_scalars(f'GradNorms/F_norms_gru',{'env {}'.format(i): F_norms_gru[i] for i in range(len(F_norms_gru))}, total_num_steps)
                # summary_writer.add_scalars(f'GradNorms/F_norms_dqn',{'env {}'.format(i): F_norms_dqn[i] for i in range(len(F_norms_dqn))}, total_num_steps)
            print(out_str)

            # summary_writer.add_scalar(f'Norms/mean_norm_all', mean_grad_grad.norm(2), total_num_steps)
            # summary_writer.add_scalar(f'Norms/mean_norms_gru', mean_gru.norm(2), total_num_steps)
            # summary_writer.add_scalar(f'Norms/mean_norms_dqn', mean_dqn.norm(2), total_num_steps)

            # summary_writer.add_scalar(f'Norms_val/val_norm_all', val_mean_grad.norm(2), total_num_steps)
            # summary_writer.add_scalar(f'Norms_val/val_norms_gru', mean_gru_val.norm(2), total_num_steps)
            # summary_writer.add_scalar(f'Norms_val/val_norms_dqn', mean_dqn_val.norm(2), total_num_steps)
            #
            # summary_writer.add_scalar(f'Norms_grad/grad_norm_all', grad_mean_grad.norm(2), total_num_steps)
            # summary_writer.add_scalar(f'Norms_grad/grad_norms_gru', mean_gru_grad.norm(2), total_num_steps)
            # summary_writer.add_scalar(f'Norms_grad/grad_norms_dqn', mean_dqn_grad.norm(2), total_num_steps)

            summary_writer.add_scalar(f'Norms_grad/grad_norm_all', mean_grad_grad.norm(2), total_num_steps)
            summary_writer.add_scalar(f'Norms_val/val_norm_all', mean_grad_val.norm(2), total_num_steps)

            # summary_writer.add_scalar(f'Norms_phi/Norms_phi', attn_grads[0].norm(2), total_num_steps)

            # summary_writer.add_scalar(f'Correlation/Corr_all', Corr_all, total_num_steps)
            # summary_writer.add_scalar(f'Correlation/Corr_gru', Corr_gru, total_num_steps)
            # summary_writer.add_scalar(f'Correlation/Corr_dqn', Corr_dqn, total_num_steps)

            # summary_writer.add_scalar(f'Correlation_grad/Corr_all_grad', Corr_all_grad, total_num_steps)
            # summary_writer.add_scalar(f'Correlation_grad/Corr_gru_grad', Corr_gru_grad, total_num_steps)
            # summary_writer.add_scalar(f'Correlation_grad/Corr_dqn_grad', Corr_dqn_grad, total_num_steps)

            summary_writer.add_scalar(f'Correlation_grad_vec/Corr_all_grad', Mean_Corr_all_grad, total_num_steps)
            summary_writer.add_scalar(f'Correlation_grad_vec/Corr_dqn_L2_grad', Mean_Corr_dqn_L2_grad, total_num_steps)
            summary_writer.add_scalar(f'Correlation_grad_vec/Mean_Corr_all_grad', (sum(Corr_all_grad_vec) / len(Corr_all_grad_vec)), total_num_steps)
            summary_writer.add_scalar(f'Correlation_grad_vec/Mean_Corr_dqn_L2_grad_vec', (sum(Corr_dqn_L2_grad_vec) / len(Corr_dqn_L2_grad_vec)), total_num_steps)
            # summary_writer.add_scalar(f'Correlation_grad_vec/Corr_gru_grad_vec', sum(Corr_gru_grad_vec) / len(Corr_gru_grad_vec), total_num_steps)
            # summary_writer.add_scalar(f'Correlation_grad_vec/Corr_dqn_grad_vec', sum(Corr_dqn_grad_vec) / len(Corr_dqn_grad_vec), total_num_steps)



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
    main_dqn(parser.parse_args())
