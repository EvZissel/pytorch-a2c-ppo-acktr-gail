import math
import random
from copy import deepcopy
import os
import glob

import numpy as np
import torch
import wandb
from a2c_ppo_acktr.envs import make_vec_envs
from models_dqn import DQN, CnnDQN, DQN_softAttn, DQN_softAttn_L2grad, DQN_RNNLast, DQN_RNNLast_analytic
from helpers_dqn import ReplayBuffer, make_atari, make_gym_env, wrap_deepmind, wrap_pytorch, ReplayBufferBandit
import torch.optim as optim
from grad_tools.grad_plot import GradPlotDqn


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

    def act(self, state, hidden_state, epsilon, masks):
        """DQN action - max q-value w/ epsilon greedy exploration."""
        # state = torch.tensor(np.float32(state)).type(dtype).unsqueeze(0)
        q_value, out_3, out_2, out_1, rnn_hxs = self.q_network.forward(state, hidden_state, masks)
        if random.random() > epsilon:
            return q_value.max(1)[1].unsqueeze(-1), out_3, out_2, out_1, rnn_hxs
        return torch.tensor(np.random.randint(self.env.action_space.n, size=q_value.size()[0])).type(dtypelong).unsqueeze(-1), out_3, out_2, out_1, rnn_hxs


def _flatten_grad(grads):
    flatten_grad = torch.cat([g.flatten() for g in grads])
    return flatten_grad

def _unflatten_grad(grads, shapes):
    unflatten_grad, idx = [], 0
    for shape in shapes:
        length = np.prod(shape)
        unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
        idx += length
    return unflatten_grad


def compute_td_loss(agent, num_mini_batch, mini_batch_size, replay_buffer, optimizer, gamma, loss_var_coeff, k=0, device = "CPU", train=True, compute_analytic=False, same_ind=False, start_ind_array=None):
    num_processes = replay_buffer.rewards.size(1)
    num_steps = replay_buffer.rewards.size(0)
    num_steps_per_batch = int(num_steps/num_mini_batch)


    if not same_ind:
        start_ind_array = [i for i in range(0, num_steps, num_steps_per_batch)]
        start_ind_array = np.random.choice(start_ind_array, size=mini_batch_size, replace=False)

    all_losses = []
    # all_grad_W2 = []
    # all_grad_b2 = []
    grads = []
    shapes = []
    grad_L2_states_all = 0
    out1_states_all = 0
    out2_states_all = 0
    out3_states_all = 0
    states_all_all = 0
    # states_all = 0
    for i in range(num_processes):
        all_losses.append(0)
        # all_grad_W2.append(0)
        # all_grad_b2.append(0)

    recurrent_hidden = torch.zeros(1, agent.q_network.recurrent_hidden_state_size).type(dtype)
    for start_ind in start_ind_array:
        data_sampler =  replay_buffer.sampler(num_processes, start_ind, num_steps_per_batch)

        losses = []
        # grad_W2 = []
        # grad_b2 = []
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
                online_q_values, _, _, _, _ = agent.q_network(states.to(device), recurrent_hidden, done.to(device))
                _, max_indicies = torch.max(online_q_values, dim=1)
                target_q_values, _, _, _, _  = agent.target_q_network(states.to(device), recurrent_hidden, done.to(device))
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
                out2_states = torch.cat((out2_states, out_1), dim=1)
                out3_states = torch.cat((out3_states, out_1), dim=1)
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
    mean_grad = 0
    optimizer.zero_grad()
    if num_processes == 416:
        # total_loss = total_loss.mean() + loss_var_coeff * total_loss.var()
        # grads, shapes = optimizer.plot_backward(all_losses)
        # total_loss.backward()
        grads, shapes, has_grads = optimizer._pack_grad([total_loss])
        mean_flat_grad = optimizer._merge_grad(grads, has_grads)
        mean_grad = optimizer._unflatten_grad(mean_flat_grad, shapes[0])
        # optimizer._set_grad(mean_grad)
        # for i in
        #     grads
        # optimizer.step()
    else:
        grads, shapes = optimizer.plot_backward(all_losses)



    # out_1_grad =0

    # if train:
    #     out1_states_all = out1_states_all[:,0,:]
    # else:
    #     out1_states_all = out1_states_all[:,2,:]

    # return total_loss, grad_L2_states_all, out1_states_all, start_ind_array, out_1_grad, grads, shapes[0]
    return total_loss, grad_L2_states_all, out1_states_all, out2_states_all, out3_states_all, start_ind_array, mean_grad, grads


def compute_activation_only(agent, num_mini_batch, mini_batch_size, replay_buffer, optimizer, gamma,  device = "CPU", start_ind_array=None):
    num_processes = replay_buffer.rewards.size(1)
    num_steps = replay_buffer.rewards.size(0)
    num_steps_per_batch = int(num_steps/num_mini_batch)

    all_losses = []
    out1_states_all = 0
    out2_states_all = 0
    out3_states_all = 0
    states_all_all = 0
    # states_all = 0
    for i in range(num_processes):
        all_losses.append(0)
        # all_grad_W2.append(0)
        # all_grad_b2.append(0)

    recurrent_hidden = torch.zeros(1, agent.q_network.recurrent_hidden_state_size).type(dtype)
    for start_ind in start_ind_array:
        data_sampler =  replay_buffer.sampler(num_processes, start_ind, num_steps_per_batch)

        out1_states = torch.tensor(0, device= device)
        out2_states = torch.tensor(0, device= device)
        out3_states = torch.tensor(0, device= device)
        states_all  = torch.tensor(0, device= device)

        for states, actions, rewards, done in data_sampler:


            # Normal DDQN update
            # with torch.backends.cudnn.flags(enabled=False):
            with torch.no_grad():
                q_values, out_3, out_2, out_1, _ = agent.q_network(states.to(device), recurrent_hidden, done.to(device))

            q_value = q_values[:-1, :].gather(1, actions.to(device)).squeeze(1)
            out_1 = out_1[:-1, :].unsqueeze(1)
            out_2 = out_2[:-1, :].unsqueeze(1)
            out_3 = out_3[:-1, :].unsqueeze(1)

            if (out1_states.sum() == 0):
                out1_states = out_1
                out2_states = out_2
                out3_states = out_3
                states_all = states[:-1, :].unsqueeze(1).to(device)
            else:

                out1_states = torch.cat((out1_states, out_1), dim=1)
                out2_states = torch.cat((out2_states, out_1), dim=1)
                out3_states = torch.cat((out3_states, out_1), dim=1)
                states_all = torch.cat((states_all, states[:-1, :].unsqueeze(1).to(device)), dim=1)

        if start_ind == start_ind_array[0] :
            out1_states_all = out1_states
            out2_states_all = out2_states
            out3_states_all = out3_states
            states_all_all = states_all
        else:

            out1_states_all = torch.cat((out1_states_all, out1_states), dim=0)
            out2_states_all = torch.cat((out2_states_all, out1_states), dim=0)
            out3_states_all = torch.cat((out3_states_all, out1_states), dim=0)
            states_all_all = torch.cat((states_all_all, states_all), dim=0)


    return states_all_all, out1_states_all, out2_states_all, out3_states_all, start_ind_array


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
                actions, _, _, _, recurrent_hidden = agent.act(obs, recurrent_hidden, epsilon=-1, masks=masks)

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


device = "cuda:0"
num_processes = 416
num_steps = 108
task_steps = 6
shapes = torch.load('/home/ev/Desktop/reinforce_atten_bandits/dqn_logs/dqn_runs_offline/offline_train_winsorizedh_bandit-obs-randchoose-v8_0_num_arms_25_16-09-2022_12-33-35/grads/h_bandit-obs-randchoose-v8-epoch-0-shapes.pt', map_location=device)
shapes = shapes['shapes']
seed = 0

num_mini_batch = 18
mini_batch_size = 1
gamma = 0.99
learning_rate = 0.001

EVAL_ENVS = {'train_eval': ['h_bandit-obs-randchoose-v14', num_processes],
             'valid_eval': ['h_bandit-obs-randchoose-v9', num_processes],
             'test_eval' : ['h_bandit-obs-randchoose-v1', 100]}

print('making envs...')
eval_envs_dic = {}
eval_locations_dic = {}
for eval_disp_name, eval_env_name in EVAL_ENVS.items():
    eval_locations_dic[eval_disp_name] = np.random.randint(0, 6, size=num_processes)

for eval_disp_name, eval_env_name in EVAL_ENVS.items():
    eval_envs_dic[eval_disp_name] = make_vec_envs(eval_env_name[0], seed, num_processes,
                                                  eval_locations_dic[eval_disp_name],
                                                  None, None, device, True, steps=task_steps,
                                                  recurrent=False,
                                                  obs_recurrent=True, multi_task=True,
                                                  free_exploration=6,
                                                  normalize=not True, rotate=False,
                                                  obs_rand_loc=False)

q_network = DQN_RNNLast(eval_envs_dic['train_eval'].observation_space.shape, eval_envs_dic['train_eval'].action_space.n, False, recurrent=True, hidden_size=64)
q_network = q_network.to(device)
target_q_network = deepcopy(q_network)

agent = Agent(eval_envs_dic['train_eval'], q_network, target_q_network)

q_network = q_network.to(device)
target_q_network = target_q_network.to(device)

replay_buffer_400 = ReplayBufferBandit(num_steps, num_processes, eval_envs_dic['train_eval'].observation_space.shape, eval_envs_dic['train_eval'].action_space)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
optimizer = GradPlotDqn(optimizer)

obs = eval_envs_dic['train_eval'].reset()
replay_buffer_400.obs[0].copy_(obs)

for step in range(num_steps):

    actions = torch.tensor(np.random.randint(agent.num_actions, size=num_processes)).type(dtypelong).unsqueeze(-1)

    next_obs, reward, done, infos = eval_envs_dic['train_eval'].step(actions.cpu())

    # for info in infos:
    #     if 'episode' in info.keys():
    #         episode_rewards.append(info['episode']['r'])
    #         episode_len.append(info['episode']['l'])

    masks = torch.FloatTensor(
        [[0.0] if done_ else [1.0] for done_ in done])

    replay_buffer_400.insert(next_obs, actions, reward, masks)


# obs = eval_envs_dic['train_eval'].reset()
# recurrent_hidden = torch.zeros(num_processes, agent.q_network.recurrent_hidden_state_size).type(dtype)
# masks = torch.ones(num_processes, 1).type(dtype)
# out1_mean = 0
# out2_mean = 0
# out3_mean = 0
# obs_mean = obs/task_steps
#
# for t in range(task_steps):
#     with torch.no_grad():
#         actions,  out_3, out_2, out_1, recurrent_hidden = agent.act(obs, recurrent_hidden, epsilon=-1, masks=masks)
#
#     # Observe reward and next obs
#     obs, _, done, infos = eval_envs_dic['train_eval'].step(actions.cpu())
#     out1_mean += out_1/task_steps
#     out2_mean += out_2/task_steps
#     out3_mean += out_3/task_steps
#     obs_mean  += obs/task_steps

num_processes = 25
num_steps = 108000
num_mini_batch = 18000
mini_batch_size = 10


EVAL_ENVS = {'train_eval': ['h_bandit-obs-randchoose-v8', num_processes],
             'valid_eval': ['h_bandit-obs-randchoose-v9', num_processes],
             'test_eval' : ['h_bandit-obs-randchoose-v1', 100]}

print('making envs...')
eval_envs_dic = {}
eval_locations_dic = {}
for eval_disp_name, eval_env_name in EVAL_ENVS.items():
    eval_locations_dic[eval_disp_name] = np.random.randint(0, 6, size=num_processes)

for eval_disp_name, eval_env_name in EVAL_ENVS.items():
    eval_envs_dic[eval_disp_name] = make_vec_envs(eval_env_name[0], seed, num_processes,
                                                  eval_locations_dic[eval_disp_name],
                                                  None, None, device, True, steps=task_steps,
                                                  recurrent=False,
                                                  obs_recurrent=True, multi_task=True,
                                                  free_exploration=6,
                                                  normalize=not True, rotate=False,
                                                  obs_rand_loc=False)


replay_buffer = ReplayBufferBandit(num_steps, num_processes, eval_envs_dic['train_eval'].observation_space.shape, eval_envs_dic['train_eval'].action_space)
obs = eval_envs_dic['train_eval'].reset()
replay_buffer.obs[0].copy_(obs)

for step in range(num_steps):

    actions = torch.tensor(np.random.randint(agent.num_actions, size=num_processes)).type(dtypelong).unsqueeze(-1)

    next_obs, reward, done, infos = eval_envs_dic['train_eval'].step(actions.cpu())

    # for info in infos:
    #     if 'episode' in info.keys():
    #         episode_rewards.append(info['episode']['r'])
    #         episode_len.append(info['episode']['l'])

    masks = torch.FloatTensor(
        [[0.0] if done_ else [1.0] for done_ in done])

    replay_buffer.insert(next_obs, actions, reward, masks)

num_processes = 416
num_steps = 108
num_mini_batch = 18
mini_batch_size = 1

cosine_Wh1_i = []
cosine_Wh2_i = []
cosine_bh1_i = []
cosine_bh2_i = []
cosine_L1w_i = []
cosine_L1b_i = []
cosine_L2w_i = []
cosine_L2b_i = []
cosine_LLw_i = []
cosine_LLb_i = []

cosine_Wh1_25 = []
cosine_Wh2_25 = []
cosine_bh1_25 = []
cosine_bh2_25 = []
cosine_L1w_25 = []
cosine_L1b_25 = []
cosine_L2w_25 = []
cosine_L2b_25 = []
cosine_LLw_25 = []
cosine_LLb_25 = []

cosine_Wh1_WOi = []
cosine_Wh2_WOi = []
cosine_bh1_WOi = []
cosine_bh2_WOi = []
cosine_L1w_WOi = []
cosine_L1b_WOi = []
cosine_L2w_WOi = []
cosine_L2b_WOi = []
cosine_LLw_WOi = []
cosine_LLb_WOi = []


cosine_Wh1 = []
cosine_Wh2 = []
cosine_bh1 = []
cosine_bh2 = []
cosine_L1w = []
cosine_L1b = []
cosine_L2w = []
cosine_L2b = []
cosine_LLw = []
cosine_LLb = []



# cosine_h1_i_act = []
# cosine_h2_i_act = []
# cosine_L1_i_act = []
# cosine_L2_i_act = []
# cosine_LL_i_act = []
#
#
# cosine_h1_i_act_25 = []
# cosine_h2_i_act_25 = []
# cosine_L1_i_act_25 = []
# cosine_L2_i_act_25 = []
# cosine_LL_i_act_25 = []
#
# cosine_h1_i_act_WOi = []
# cosine_h2_i_act_WOi = []
# cosine_L1_i_act_WOi = []
# cosine_L2_i_act_WOi = []
# cosine_LL_i_act_WOi = []
#
# cosine_h1_i_act_0 = []
# cosine_h2_i_act_0 = []
# cosine_L1_i_act_0 = []
# cosine_L2_i_act_0 = []
# cosine_LL_i_act_0 = []

Wh1_i_vec = []
Wh2_i_vec = []
bh1_i_vec = []
bh2_i_vec = []
L1w_i_vec = []
L1b_i_vec = []
L2w_i_vec = []
L2b_i_vec = []
LLw_i_vec = []
LLb_i_vec = []

for i in range(25):

    cosine_Wh1_i.append([])
    cosine_Wh2_i.append([])
    cosine_bh1_i.append([])
    cosine_bh2_i.append([])
    cosine_L1w_i.append([])
    cosine_L1b_i.append([])
    cosine_L2w_i.append([])
    cosine_L2b_i.append([])
    cosine_LLw_i.append([])
    cosine_LLb_i.append([])

    cosine_Wh1_WOi.append([])
    cosine_Wh2_WOi.append([])
    cosine_bh1_WOi.append([])
    cosine_bh2_WOi.append([])
    cosine_L1w_WOi.append([])
    cosine_L1b_WOi.append([])
    cosine_L2w_WOi.append([])
    cosine_L2b_WOi.append([])
    cosine_LLw_WOi.append([])
    cosine_LLb_WOi.append([])

    # cosine_h1_i_act.append([])
    # cosine_h2_i_act.append([])
    # cosine_L1_i_act.append([])
    # cosine_L2_i_act.append([])
    # cosine_LL_i_act.append([])
    #
    # cosine_h1_i_act_25.append([])
    # cosine_h2_i_act_25.append([])
    # cosine_L1_i_act_25.append([])
    # cosine_L2_i_act_25.append([])
    # cosine_LL_i_act_25.append([])
    #
    # cosine_h1_i_act_WOi.append([])
    # cosine_h2_i_act_WOi.append([])
    # cosine_L1_i_act_WOi.append([])
    # cosine_L2_i_act_WOi.append([])
    # cosine_LL_i_act_WOi.append([])
    #
    # cosine_h1_i_act_0.append([])
    # cosine_h2_i_act_0.append([])
    # cosine_L1_i_act_0.append([])
    # cosine_L2_i_act_0.append([])
    # cosine_LL_i_act_0.append([])

    Wh1_i_vec.append([])
    Wh2_i_vec.append([])
    bh1_i_vec.append([])
    bh2_i_vec.append([])
    L1w_i_vec.append([])
    L1b_i_vec.append([])
    L2w_i_vec.append([])
    L2b_i_vec.append([])
    LLw_i_vec.append([])
    LLb_i_vec.append([])


for epoch in range(0, 9999, 43):
    q_network_weighs = torch.load('/home/ev/Desktop/reinforce_atten_bandits/dqn_logs/dqn_runs_offline/offline_train_winsorizedh_bandit-obs-randchoose-v8_0_num_arms_25_16-09-2022_12-33-35/h_bandit-obs-randchoose-v8-epoch-{}.pt'.format(epoch),map_location=device)
    agent.q_network.load_state_dict(q_network_weighs['state_dict'])
    agent.target_q_network.load_state_dict(q_network_weighs['target_state_dict'])

    loss, grads_L2, out1, out2, out3, _, mean_grad_400, grads_400 = compute_td_loss(agent, num_mini_batch, mini_batch_size, replay_buffer_400, optimizer, gamma, 0, k=0, device=device, train=True, compute_analytic=False)

    # grads_ind = torch.load('/home/ev/Desktop/reinforce_atten_bandits/dqn_logs/dqn_runs_offline/offline_train_winsorizedh_bandit-obs-randchoose-v8_0_num_arms_25_16-09-2022_12-33-35/grads/h_bandit-obs-randchoose-v9-epoch-{}-optimizer_grad.pt'.format(
    #             epoch), map_location=device)
    # grads = grads_ind['grad_ens']
    # ind_array = grads_ind['ind_array']

    _, _, _, _, _, _, mean_grad, grads = compute_td_loss(agent, 18000, 250, replay_buffer, optimizer, gamma, 0, k=0, device=device, train=True, compute_analytic=False)

    # states_all_all, out1_states_all, out2_states_all, out3_states_all, _ = compute_activation_only(agent, 18000, 10, replay_buffer, optimizer, gamma, device=device, start_ind_array=ind_array)
    # states_all_all_mean  = states_all_all.mean(0)
    # out1_states_all_mean = out1_states_all.mean(0)
    # out2_states_all_mean = out2_states_all.mean(0)
    # out3_states_all_mean = out3_states_all.mean(0)

    mean_grad_25 = sum(grads)/len(grads)
    mean_grad_25 = _unflatten_grad(mean_grad_25, shapes)

    Wh1_25 = mean_grad_25[0]
    Wh2_25 = mean_grad_25[1]

    bh1_25 = mean_grad_25[2]
    bh2_25 = mean_grad_25[3]

    L1w_25 = mean_grad_25[4]
    L1b_25 = mean_grad_25[5]

    L2w_25 = mean_grad_25[6]
    L2b_25 = mean_grad_25[7]

    LLw_25 = mean_grad_25[8]
    LLb_25 = mean_grad_25[9]


    Wh1_i_400 = mean_grad_400[0]
    Wh2_i_400 = mean_grad_400[1]

    bh1_i_400 = mean_grad_400[2]
    bh2_i_400 = mean_grad_400[3]

    L1w_i_400 = mean_grad_400[4]
    L1b_i_400 = mean_grad_400[5]

    L2w_i_400 = mean_grad_400[6]
    L2b_i_400 = mean_grad_400[7]

    LLw_i_400 = mean_grad_400[8]
    LLb_i_400 = mean_grad_400[9]


    for i in range(25):

        grads_i = _unflatten_grad(grads[i], shapes)
        # states_all_all_mean_WOi  = torch.cat((states_all_all_mean[:i],   states_all_all_mean[i + 1:]), dim=0).mean(0)
        # out1_states_all_mean_WOi = torch.cat((out1_states_all_mean[:i], out1_states_all_mean[i + 1:]), dim=0).mean(0)
        # out2_states_all_mean_WOi = torch.cat((out2_states_all_mean[:i], out2_states_all_mean[i + 1:]), dim=0).mean(0)
        # out3_states_all_mean_WOi = torch.cat((out3_states_all_mean[:i], out3_states_all_mean[i + 1:]), dim=0).mean(0)

        Wh1_i = grads_i[0]
        Wh2_i = grads_i[1]

        bh1_i = grads_i[2]
        bh2_i = grads_i[3]

        L1w_i = grads_i[4]
        L1b_i = grads_i[5]

        L2w_i = grads_i[6]
        L2b_i = grads_i[7]

        LLw_i = grads_i[8]
        LLb_i = grads_i[9]

        grads_WOi = grads[:i] + grads[i+1:]
        mean_grad_WOi = sum(grads_WOi) / len(grads_WOi)
        mean_grad_WOi = _unflatten_grad(mean_grad_WOi, shapes)

        Wh1_WOi = mean_grad_WOi[0]
        Wh2_WOi = mean_grad_WOi[1]

        bh1_WOi = mean_grad_WOi[2]
        bh2_WOi = mean_grad_WOi[3]

        L1w_WOi = mean_grad_WOi[4]
        L1b_WOi = mean_grad_WOi[5]

        L2w_WOi = mean_grad_WOi[6]
        L2b_WOi = mean_grad_WOi[7]

        LLw_WOi = mean_grad_WOi[8]
        LLb_WOi = mean_grad_WOi[9]

        # L1_activation = torch.matmul(L1w_i, states_all_all_mean[i])  + L1b_i
        # L2_activation = torch.matmul(L2w_i, out1_states_all_mean[i]) + L2b_i
        #
        # h1_activation = torch.matmul(Wh1_i, out2_states_all_mean[i]) + bh1_i
        # h2_activation = torch.matmul(Wh2_i, out2_states_all_mean[i]) + bh2_i
        #
        # LL_activation = torch.matmul(LLw_i, out3_states_all_mean[i]) + LLb_i



        # L1_activation_WOi = torch.matmul(L1w_i, states_all_all_mean_WOi)  + L1b_i
        # L2_activation_WOi = torch.matmul(L2w_i, out1_states_all_mean_WOi) + L2b_i
        #
        # h1_activation_WOi = torch.matmul(Wh1_i, out2_states_all_mean_WOi) + bh1_i
        # h2_activation_WOi = torch.matmul(Wh2_i, out2_states_all_mean_WOi) + bh2_i
        #
        # LL_activation_WOi = torch.matmul(LLw_i, out3_states_all_mean_WOi) + LLb_i


        # L1_activation_0 = torch.matmul(L1w_i, states_all_all_mean[0])  + L1b_i
        # L2_activation_0 = torch.matmul(L2w_i, out1_states_all_mean[0]) + L2b_i
        #
        # h1_activation_0 = torch.matmul(Wh1_i, out2_states_all_mean[0]) + bh1_i
        # h2_activation_0 = torch.matmul(Wh2_i, out2_states_all_mean[0]) + bh2_i
        #
        # LL_activation_0 = torch.matmul(LLw_i, out3_states_all_mean[0]) + LLb_i
        #
        #
        # L1_activation_25 = torch.matmul(L1w_25, states_all_all_mean[i])  + L1b_25
        # L2_activation_25 = torch.matmul(L2w_25, out1_states_all_mean[i]) + L2b_25
        #
        # h1_activation_25 = torch.matmul(Wh1_25, out2_states_all_mean[i]) + bh1_25
        # h2_activation_25 = torch.matmul(Wh2_25, out2_states_all_mean[i]) + bh2_25
        #
        # LL_activation_25 = torch.matmul(LLw_25, out3_states_all_mean[i]) + LLb_25
        #
        #
        # L1_activation_400 = torch.matmul(L1w_i_400,  states_all_all_mean[i]) + L1b_i_400
        # L2_activation_400 = torch.matmul(L2w_i_400, out1_states_all_mean[i]) + L2b_i_400
        #
        # h1_activation_400 = torch.matmul(Wh1_i_400, out2_states_all_mean[i]) + bh1_i_400
        # h2_activation_400 = torch.matmul(Wh2_i_400, out2_states_all_mean[i]) + bh2_i_400
        #
        # LL_activation_400 = torch.matmul(LLw_i_400, out3_states_all_mean[i]) + LLb_i_400



        cosine_Wh1_i[i].append((Wh1_i * Wh1_i_400).sum() / (Wh1_i.norm(2) * Wh1_i_400.norm(2)))

        cosine_Wh2_i[i].append((Wh2_i * Wh2_i_400).sum() / (Wh2_i.norm(2) * Wh2_i_400.norm(2)))

        cosine_bh1_i[i].append((bh1_i * bh1_i_400).sum() / (bh1_i.norm(2) * bh1_i_400.norm(2)))

        cosine_bh2_i[i].append((bh2_i * bh2_i_400).sum() / (bh2_i.norm(2) * bh2_i_400.norm(2)))

        cosine_L1w_i[i].append((L1w_i * L1w_i_400).sum() / (L1w_i.norm(2) * L1w_i_400.norm(2)))

        cosine_L1b_i[i].append((L1b_i * L1b_i_400).sum() / (L1b_i.norm(2) * L1b_i_400.norm(2)))

        cosine_L2w_i[i].append((L2w_i * L2w_i_400).sum() / (L2w_i.norm(2) * L2w_i_400.norm(2)))

        cosine_L2b_i[i].append((L2b_i * L2b_i_400).sum() / (L2b_i.norm(2) * L2b_i_400.norm(2)))

        cosine_LLw_i[i].append((LLw_i * LLw_i_400).sum() / (LLw_i.norm(2) * LLw_i_400.norm(2)))

        cosine_LLb_i[i].append((LLb_i * LLb_i_400).sum() / (LLb_i.norm(2) * LLb_i_400.norm(2)))



        cosine_Wh1_WOi[i].append((Wh1_WOi * Wh1_25).sum() / (Wh1_WOi.norm(2) * Wh1_25.norm(2)))

        cosine_Wh2_WOi[i].append((Wh2_WOi * Wh2_25).sum() / (Wh2_WOi.norm(2) * Wh2_25.norm(2)))

        cosine_bh1_WOi[i].append((bh1_WOi * bh1_25).sum() / (bh1_WOi.norm(2) * bh1_25.norm(2)))

        cosine_bh2_WOi[i].append((bh2_WOi * bh2_25).sum() / (bh2_WOi.norm(2) * bh2_25.norm(2)))

        cosine_L1w_WOi[i].append((L1w_WOi * L1w_25).sum() / (L1w_WOi.norm(2) * L1w_25.norm(2)))

        cosine_L1b_WOi[i].append((L1b_WOi * L1b_25).sum() / (L1b_WOi.norm(2) * L1b_25.norm(2)))

        cosine_L2w_WOi[i].append((L2w_WOi * L2w_25).sum() / (L2w_WOi.norm(2) * L2w_25.norm(2)))

        cosine_L2b_WOi[i].append((L2b_WOi * L2b_25).sum() / (L2b_WOi.norm(2) * L2b_25.norm(2)))

        cosine_LLw_WOi[i].append((LLw_WOi * LLw_25).sum() / (LLw_WOi.norm(2) * LLw_25.norm(2)))

        cosine_LLb_WOi[i].append((LLb_WOi * LLb_25).sum() / (LLb_WOi.norm(2) * LLb_25.norm(2)))


        # cosine_h1_i_act[i].append((h1_activation * h1_activation_400).sum() / (h1_activation.norm(2) * h1_activation_400.norm(2)))
        # cosine_h2_i_act[i].append((h2_activation * h2_activation_400).sum() / (h2_activation.norm(2) * h2_activation_400.norm(2)))
        # cosine_L1_i_act[i].append((L1_activation * L1_activation_400).sum() / (L1_activation.norm(2) * L1_activation_400.norm(2)))
        # cosine_L2_i_act[i].append((L2_activation * L2_activation_400).sum() / (L2_activation.norm(2) * L2_activation_400.norm(2)))
        # cosine_LL_i_act[i].append((LL_activation * LL_activation_400).sum() / (LL_activation.norm(2) * LL_activation_400.norm(2)))
        #
        # cosine_h1_i_act_25[i].append((h1_activation_25 * h1_activation_400).sum() / (h1_activation_25.norm(2) * h1_activation_400.norm(2)))
        # cosine_h2_i_act_25[i].append((h2_activation_25 * h2_activation_400).sum() / (h2_activation_25.norm(2) * h2_activation_400.norm(2)))
        # cosine_L1_i_act_25[i].append((L1_activation_25 * L1_activation_400).sum() / (L1_activation_25.norm(2) * L1_activation_400.norm(2)))
        # cosine_L2_i_act_25[i].append((L2_activation_25 * L2_activation_400).sum() / (L2_activation_25.norm(2) * L2_activation_400.norm(2)))
        # cosine_LL_i_act_25[i].append((LL_activation_25 * LL_activation_400).sum() / (LL_activation_25.norm(2) * LL_activation_400.norm(2)))
        #
        # cosine_h1_i_act_WOi[i].append((h1_activation * h1_activation_WOi).sum() / (h1_activation.norm(2) * h1_activation_WOi.norm(2)))
        # cosine_h2_i_act_WOi[i].append((h2_activation * h2_activation_WOi).sum() / (h2_activation.norm(2) * h2_activation_WOi.norm(2)))
        # cosine_L1_i_act_WOi[i].append((L1_activation * L1_activation_WOi).sum() / (L1_activation.norm(2) * L1_activation_WOi.norm(2)))
        # cosine_L2_i_act_WOi[i].append((L2_activation * L2_activation_WOi).sum() / (L2_activation.norm(2) * L2_activation_WOi.norm(2)))
        # cosine_LL_i_act_WOi[i].append((LL_activation * LL_activation_WOi).sum() / (LL_activation.norm(2) * LL_activation_WOi.norm(2)))
        #
        # cosine_h1_i_act_0[i].append((h1_activation * h1_activation_0).sum() / (h1_activation.norm(2) * h1_activation_0.norm(2)))
        # cosine_h2_i_act_0[i].append((h2_activation * h2_activation_0).sum() / (h2_activation.norm(2) * h2_activation_0.norm(2)))
        # cosine_L1_i_act_0[i].append((L1_activation * L1_activation_0).sum() / (L1_activation.norm(2) * L1_activation_0.norm(2)))
        # cosine_L2_i_act_0[i].append((L2_activation * L2_activation_0).sum() / (L2_activation.norm(2) * L2_activation_0.norm(2)))
        # cosine_LL_i_act_0[i].append((LL_activation * LL_activation_0).sum() / (LL_activation.norm(2) * LL_activation_0.norm(2)))


        Wh1_i_vec[i].append(Wh1_i.norm(2))

        Wh2_i_vec[i].append(Wh2_i.norm(2))

        bh1_i_vec[i].append(bh1_i.norm(2))

        bh2_i_vec[i].append(bh2_i.norm(2))

        L1w_i_vec[i].append(L1w_i.norm(2))

        L1b_i_vec[i].append(L1b_i.norm(2))

        L2w_i_vec[i].append(L2w_i.norm(2))

        L2b_i_vec[i].append(L2b_i.norm(2))

        LLw_i_vec[i].append(LLw_i.norm(2))

        LLb_i_vec[i].append(LLb_i.norm(2))

    # print("debug")
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

    for i in range (25):
        grads_i = _unflatten_grad(grads[i], shapes)
        Wh1.append(grads_i[0])
        Wh2.append(grads_i[1])

        bh1.append(grads_i[2])
        bh2.append(grads_i[3])

        L1w.append(grads_i[4])
        L1b.append(grads_i[5])

        L2w.append(grads_i[6])
        L2b.append(grads_i[7])

        LLw.append(grads_i[8])
        LLb.append(grads_i[9])

    min_corr = 0.99
    for i in range(5):
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



        # # ind = torch.nonzero(torch.tensor(cosine_Wh1_WOi).squeeze() > 0.5).squeeze()
        # smallest_val = torch.sort(torch.tensor(cosine_Wh1_WOi)[:,len(cosine_LLw_WOi[0])-1])[0][0]
        # smallest_ind = torch.sort(torch.tensor(cosine_Wh1_WOi)[:,len(cosine_LLw_WOi[0])-1])[1][0]
        # if smallest_val < 0.5:
        #     grads =  grads[:smallest_ind] + grads[smallest_ind+1:]

    cosine_Wh1_25.append((Wh1_25 * Wh1_i_400).sum() / (Wh1_25.norm(2) * Wh1_i_400.norm(2)))

    cosine_Wh2_25.append((Wh2_25 * Wh2_i_400).sum() / (Wh2_25.norm(2) * Wh2_i_400.norm(2)))

    cosine_bh1_25.append((bh1_25 * bh1_i_400).sum() / (bh1_25.norm(2) * bh1_i_400.norm(2)))

    cosine_bh2_25.append((bh2_25 * bh2_i_400).sum() / (bh2_25.norm(2) * bh2_i_400.norm(2)))

    cosine_L1w_25.append((L1w_25 * L1w_i_400).sum() / (L1w_25.norm(2) * L1w_i_400.norm(2)))

    cosine_L1b_25.append((L1b_25 * L1b_i_400).sum() / (L1b_25.norm(2) * L1b_i_400.norm(2)))

    cosine_L2w_25.append((L2w_25 * L2w_i_400).sum() / (L2w_25.norm(2) * L2w_i_400.norm(2)))

    cosine_L2b_25.append((L2b_25 * L2b_i_400).sum() / (L2b_25.norm(2) * L2b_i_400.norm(2)))

    cosine_LLw_25.append((LLw_25 * LLw_i_400).sum() / (LLw_25.norm(2) * LLw_i_400.norm(2)))

    cosine_LLb_25.append((LLb_25 * LLb_i_400).sum() / (LLb_25.norm(2) * LLb_i_400.norm(2)))

    mean_Wh1 = sum(Wh1) / len(Wh1)
    print("iter {} Wh1 len {}".format(epoch,len(Wh1)))
    cosine_Wh1.append((mean_Wh1 * Wh1_i_400).sum() / (mean_Wh1.norm(2) * Wh1_i_400.norm(2)))

    mean_Wh2 = sum(Wh2) / len(Wh2)
    print("iter {} Wh2 len {}".format(epoch, len(Wh2)))
    cosine_Wh2.append((mean_Wh2 * Wh2_i_400).sum() / (mean_Wh2.norm(2) * Wh2_i_400.norm(2)))

    mean_bh1 = sum(bh1) / len(bh1)
    print("iter {} bh1 len {}".format(epoch, len(bh1)))
    cosine_bh1.append((mean_bh1 * bh1_i_400).sum() / (mean_bh1.norm(2) * bh1_i_400.norm(2)))

    mean_bh2 = sum(bh2) / len(bh2)
    print("iter {} bh2 len {}".format(epoch, len(bh2)))
    cosine_bh2.append((mean_bh2 * bh2_i_400).sum() / (mean_bh2.norm(2) * bh2_i_400.norm(2)))

    mean_L1w = sum(L1w) / len(L1w)
    print("iter {} L1w len {}".format(epoch, len(L1w)))
    cosine_L1w.append((mean_L1w * L1w_i_400).sum() / (mean_L1w.norm(2) * L1w_i_400.norm(2)))

    mean_L1b = sum(L1b) / len(L1b)
    print("iter {} L1b len {}".format(epoch, len(L1b)))
    cosine_L1b.append((mean_L1b * L1b_i_400).sum() / (mean_L1b.norm(2) * L1b_i_400.norm(2)))

    mean_L2w = sum(L2w) / len(L2w)
    print("iter {} L2w len {}".format(epoch, len(L2w)))
    cosine_L2w.append((mean_L2w * L2w_i_400).sum() / (mean_L2w.norm(2) * L2w_i_400.norm(2)))

    mean_L2b = sum(L2b) / len(L2b)
    print("iter {} L2b len {}".format(epoch, len(L2b)))
    cosine_L2b.append((mean_L2b * L2b_i_400).sum() / (mean_L2b.norm(2) * L2b_i_400.norm(2)))

    mean_LLw = sum(LLw) / len(LLw)
    print("iter {} LLw len {}".format(epoch, len(LLw)))
    cosine_LLw.append((mean_LLw * LLw_i_400).sum() / (mean_LLw.norm(2) * LLw_i_400.norm(2)))

    mean_LLb = sum(LLb) / len(LLb)
    print("iter {} LLb len {}".format(epoch, len(LLb)))
    cosine_LLb.append((mean_LLb * LLb_i_400).sum() / (mean_LLb.norm(2) * LLb_i_400.norm(2)))



for i in range(25):

    wandb.init(project="main_dqn_offline_maximum_entropy_GradPlot_lager_batch", entity="ev_zisselman", name=f'25_400_envs_{i}')
    j = 0
    for epoch in range(0, 9999, 43):

        wandb.log({f'eval/eval_Wh1': cosine_Wh1_i[i][j]}, step=epoch)

        wandb.log({f'eval/eval_Wh2': cosine_Wh2_i[i][j]}, step=epoch)

        wandb.log({f'eval/eval_bh1': cosine_bh1_i[i][j]}, step=epoch)

        wandb.log({f'eval/eval_bh2': cosine_bh2_i[i][j]}, step=epoch)

        wandb.log({f'eval/eval_L1w': cosine_L1w_i[i][j]}, step=epoch)

        wandb.log({f'eval/eval_L1b': cosine_L1b_i[i][j]}, step=epoch)

        wandb.log({f'eval/eval_L2w': cosine_L2w_i[i][j]}, step=epoch)

        wandb.log({f'eval/eval_L2b': cosine_L2b_i[i][j]}, step=epoch)

        wandb.log({f'eval/eval_LLw': cosine_LLw_i[i][j]}, step=epoch)

        wandb.log({f'eval/eval_LLb': cosine_LLb_i[i][j]}, step=epoch)



        wandb.log({f'eval_25/eval_Wh1_25': cosine_Wh1_25[j]}, step=epoch)

        wandb.log({f'eval_25/eval_Wh2_25': cosine_Wh2_25[j]}, step=epoch)

        wandb.log({f'eval_25/eval_bh1_25': cosine_bh1_25[j]}, step=epoch)

        wandb.log({f'eval_25/eval_bh2_25': cosine_bh2_25[j]}, step=epoch)

        wandb.log({f'eval_25/eval_L1w_25': cosine_L1w_25[j]}, step=epoch)

        wandb.log({f'eval_25/eval_L1b_25': cosine_L1b_25[j]}, step=epoch)

        wandb.log({f'eval_25/eval_L2w_25': cosine_L2w_25[j]}, step=epoch)

        wandb.log({f'eval_25/eval_L2b_25': cosine_L2b_25[j]}, step=epoch)

        wandb.log({f'eval_25/eval_LLw_25': cosine_LLw_25[j]}, step=epoch)

        wandb.log({f'eval_25/eval_LLb_25': cosine_LLb_25[j]}, step=epoch)



        wandb.log({f'eval_WOi/eval_Wh1_WOi': cosine_Wh1_WOi[i][j]}, step=epoch)

        wandb.log({f'eval_WOi/eval_Wh2_WOi': cosine_Wh2_WOi[i][j]}, step=epoch)

        wandb.log({f'eval_WOi/eval_bh1_WOi': cosine_bh1_WOi[i][j]}, step=epoch)

        wandb.log({f'eval_WOi/eval_bh2_WOi': cosine_bh2_WOi[i][j]}, step=epoch)

        wandb.log({f'eval_WOi/eval_L1w_WOi': cosine_L1w_WOi[i][j]}, step=epoch)

        wandb.log({f'eval_WOi/eval_L1b_WOi': cosine_L1b_WOi[i][j]}, step=epoch)

        wandb.log({f'eval_WOi/eval_L2w_WOi': cosine_L2w_WOi[i][j]}, step=epoch)

        wandb.log({f'eval_WOi/eval_L2b_WOi': cosine_L2b_WOi[i][j]}, step=epoch)

        wandb.log({f'eval_WOi/eval_LLw_WOi': cosine_LLw_WOi[i][j]}, step=epoch)

        wandb.log({f'eval_WOi/eval_LLb_WOi': cosine_LLb_WOi[i][j]}, step=epoch)




        wandb.log({f'eval_WOi_mean/eval_Wh1_WOi_mean': cosine_Wh1[j]}, step=epoch)

        wandb.log({f'eval_WOi_mean/eval_Wh2_WOi_mean': cosine_Wh2[j]}, step=epoch)

        wandb.log({f'eval_WOi_mean/eval_bh1_WOi_mean': cosine_bh1[j]}, step=epoch)

        wandb.log({f'eval_WOi_mean/eval_bh2_WOi_mean': cosine_bh2[j]}, step=epoch)

        wandb.log({f'eval_WOi_mean/eval_L1w_WOi_mean': cosine_L1w[j]}, step=epoch)

        wandb.log({f'eval_WOi_mean/eval_L1b_WOi_mean': cosine_L1b[j]}, step=epoch)

        wandb.log({f'eval_WOi_mean/eval_L2w_WOi_mean': cosine_L2w[j]}, step=epoch)

        wandb.log({f'eval_WOi_mean/eval_L2b_WOi_mean': cosine_L2b[j]}, step=epoch)

        wandb.log({f'eval_WOi_mean/eval_LLw_WOi_mean': cosine_LLw[j]}, step=epoch)

        wandb.log({f'eval_WOi_mean/eval_LLb_WOi_mean': cosine_LLb[j]}, step=epoch)



        # wandb.log({f'activation/eval_h1': cosine_h1_i_act[i][j]}, step=epoch)
        #
        # wandb.log({f'activation/eval_h2': cosine_h2_i_act[i][j]}, step=epoch)
        #
        # wandb.log({f'activation/eval_L1': cosine_L1_i_act[i][j]}, step=epoch)
        #
        # wandb.log({f'activation/eval_L2': cosine_L2_i_act[i][j]}, step=epoch)
        #
        # wandb.log({f'activation/eval_LL': cosine_LL_i_act[i][j]}, step=epoch)
        #
        #
        #
        # wandb.log({f'activation_25/eval_h1': cosine_h1_i_act_25[i][j]}, step=epoch)
        #
        # wandb.log({f'activation_25/eval_h2': cosine_h2_i_act_25[i][j]}, step=epoch)
        #
        # wandb.log({f'activation_25/eval_L1': cosine_L1_i_act_25[i][j]}, step=epoch)
        #
        # wandb.log({f'activation_25/eval_L2': cosine_L2_i_act_25[i][j]}, step=epoch)
        #
        # wandb.log({f'activation_25/eval_LL': cosine_LL_i_act_25[i][j]}, step=epoch)
        #
        #
        #
        # wandb.log({f'activation_WOi/eval_h1': cosine_h1_i_act_WOi[i][j]}, step=epoch)
        #
        # wandb.log({f'activation_WOi/eval_h2': cosine_h2_i_act_WOi[i][j]}, step=epoch)
        #
        # wandb.log({f'activation_WOi/eval_L1': cosine_L1_i_act_WOi[i][j]}, step=epoch)
        #
        # wandb.log({f'activation_WOi/eval_L2': cosine_L2_i_act_WOi[i][j]}, step=epoch)
        #
        # wandb.log({f'activation_WOi/eval_LL': cosine_LL_i_act_WOi[i][j]}, step=epoch)
        #
        #
        # wandb.log({f'activation_0/eval_h1': cosine_h1_i_act_0[i][j]}, step=epoch)
        #
        # wandb.log({f'activation_0/eval_h2': cosine_h2_i_act_0[i][j]}, step=epoch)
        #
        # wandb.log({f'activation_0/eval_L1': cosine_L1_i_act_0[i][j]}, step=epoch)
        #
        # wandb.log({f'activation_0/eval_L2': cosine_L2_i_act_0[i][j]}, step=epoch)
        #
        # wandb.log({f'activation_0/eval_LL': cosine_LL_i_act_0[i][j]}, step=epoch)


        wandb.log({f'L2/eval_Wh1_L2': Wh1_i_vec[i][j]}, step=epoch)

        wandb.log({f'L2/eval_Wh2_L2': Wh2_i_vec[i][j]}, step=epoch)

        wandb.log({f'L2/eval_bh1_L2': bh1_i_vec[i][j]}, step=epoch)

        wandb.log({f'L2/eval_bh2_L2': bh2_i_vec[i][j]}, step=epoch)

        wandb.log({f'L2/eval_L1w_L2': L1w_i_vec[i][j]}, step=epoch)

        wandb.log({f'L2/eval_L1b_L2': L1b_i_vec[i][j]}, step=epoch)

        wandb.log({f'L2/eval_L2w_L2': L2w_i_vec[i][j]}, step=epoch)

        wandb.log({f'L2/eval_L2b_L2': L2b_i_vec[i][j]}, step=epoch)

        wandb.log({f'L2/eval_LLw_L2': LLw_i_vec[i][j]}, step=epoch)

        wandb.log({f'L2/eval_LLb_L2': LLb_i_vec[i][j]}, step=epoch)

        j += 1

    # obs_mean_0_out  = torch.matmul(L1w_i, obs_mean[0].unsqueeze(1))  + L1b_i
    # obs_mean_6_out  = torch.matmul(L1w_i, obs_mean[6].unsqueeze(1))  + L1b_i
    # obs_mean_12_out = torch.matmul(L1w_i, obs_mean[12].unsqueeze(1)) + L1b_i
    # obs_mean_18_out = torch.matmul(L1w_i, obs_mean[18].unsqueeze(1)) + L1b_i
    # obs_mean_24_out = torch.matmul(L1w_i, obs_mean[24].unsqueeze(1)) + L1b_i
    #
    # obs_mean_0_out_val  = torch.matmul(L1w_i, obs_mean_val[0].unsqueeze(1))  + L1b_i
    # obs_mean_6_out_val  = torch.matmul(L1w_i, obs_mean_val[6].unsqueeze(1))  + L1b_i
    # obs_mean_12_out_val = torch.matmul(L1w_i, obs_mean_val[12].unsqueeze(1)) + L1b_i
    # obs_mean_18_out_val = torch.matmul(L1w_i, obs_mean_val[18].unsqueeze(1)) + L1b_i
    # obs_mean_24_out_val = torch.matmul(L1w_i, obs_mean_val[24].unsqueeze(1)) + L1b_i
    #
    # L2_GradActivateion_layer1 = ((obs_mean_6_out-obs_mean_0_out).norm(2) + (obs_mean_12_out-obs_mean_0_out).norm(2) +(obs_mean_18_out-obs_mean_0_out).norm(2) + (obs_mean_24_out-obs_mean_0_out).norm(2))/4
    # L2_GradActivateion_layer1_val = ((obs_mean_6_out_val-obs_mean_0_out).norm(2) + (obs_mean_12_out_val-obs_mean_0_out).norm(2) +(obs_mean_18_out_val-obs_mean_0_out).norm(2) + (obs_mean_24_out_val-obs_mean_0_out).norm(2))/4
    # wandb.log({f'L2_GradActivateion/Layer1': L2_GradActivateion_layer1}, step=epoch)
    # wandb.log({f'L2_GradActivateion/Layer1_val': L2_GradActivateion_layer1_val}, step=epoch)


    # L2w_i = grads_0[6]
    # L2b_i = grads_0[7]
    # out1_mean_0_out  = torch.matmul(L2w_i, out1_mean[0].unsqueeze(1))  + L2b_i
    # out1_mean_6_out  = torch.matmul(L2w_i, out1_mean[6].unsqueeze(1))  + L2b_i
    # out1_mean_12_out = torch.matmul(L2w_i, out1_mean[12].unsqueeze(1)) + L2b_i
    # out1_mean_18_out = torch.matmul(L2w_i, out1_mean[18].unsqueeze(1)) + L2b_i
    # out1_mean_24_out = torch.matmul(L2w_i, out1_mean[24].unsqueeze(1)) + L2b_i
    #
    # out1_mean_0_out_val  = torch.matmul(L2w_i, out1_mean_val[0].unsqueeze(1))  + L2b_i
    # out1_mean_6_out_val  = torch.matmul(L2w_i, out1_mean_val[6].unsqueeze(1))  + L2b_i
    # out1_mean_12_out_val = torch.matmul(L2w_i, out1_mean_val[12].unsqueeze(1)) + L2b_i
    # out1_mean_18_out_val = torch.matmul(L2w_i, out1_mean_val[18].unsqueeze(1)) + L2b_i
    # out1_mean_24_out_val = torch.matmul(L2w_i, out1_mean_val[24].unsqueeze(1)) + L2b_i
    #
    # L2_GradActivateion_layer2     = ((out1_mean_6_out-out1_mean_0_out).norm(2) + (out1_mean_12_out-out1_mean_0_out).norm(2) +(out1_mean_18_out-out1_mean_0_out).norm(2) + (out1_mean_24_out-out1_mean_0_out).norm(2))/4
    # L2_GradActivateion_layer2_val = ((out1_mean_6_out_val-out1_mean_0_out).norm(2) + (out1_mean_12_out_val-out1_mean_0_out).norm(2) +(out1_mean_18_out_val-out1_mean_0_out).norm(2) + (out1_mean_24_out_val-out1_mean_0_out).norm(2))/4
    # wandb.log({f'L2_GradActivateion/Layer2': L2_GradActivateion_layer2}, step=epoch)
    # wandb.log({f'L2_GradActivateion/Layer2_val': L2_GradActivateion_layer2_val}, step=epoch)


    # LLw_i = grads_0[8]
    # LLb_i = grads_0[9]
    # out3_mean_0_out  = torch.matmul(LLw_i, out3_mean[0].unsqueeze(1))  + LLb_i
    # out3_mean_6_out  = torch.matmul(LLw_i, out3_mean[6].unsqueeze(1))  + LLb_i
    # out3_mean_12_out = torch.matmul(LLw_i, out3_mean[12].unsqueeze(1)) + LLb_i
    # out3_mean_18_out = torch.matmul(LLw_i, out3_mean[18].unsqueeze(1)) + LLb_i
    # out3_mean_24_out = torch.matmul(LLw_i, out3_mean[24].unsqueeze(1)) + LLb_i
    #
    # out3_mean_0_out_val  = torch.matmul(LLw_i, out3_mean_val[0].unsqueeze(1))  + LLb_i
    # out3_mean_6_out_val  = torch.matmul(LLw_i, out3_mean_val[6].unsqueeze(1))  + LLb_i
    # out3_mean_12_out_val = torch.matmul(LLw_i, out3_mean_val[12].unsqueeze(1)) + LLb_i
    # out3_mean_18_out_val = torch.matmul(LLw_i, out3_mean_val[18].unsqueeze(1)) + LLb_i
    # out3_mean_24_out_val = torch.matmul(LLw_i, out3_mean_val[24].unsqueeze(1)) + LLb_i
    # L2_GradActivateion_layerL = ((out3_mean_6_out-out3_mean_0_out).norm(2) + (out3_mean_12_out-out3_mean_0_out).norm(2) +(out3_mean_18_out-out3_mean_0_out).norm(2) + (out3_mean_24_out-out3_mean_0_out).norm(2))/4
    # L2_GradActivateion_layerL_val = ((out3_mean_6_out_val-out3_mean_0_out).norm(2) + (out3_mean_12_out_val-out3_mean_0_out).norm(2) +(out3_mean_18_out_val-out3_mean_0_out).norm(2) + (out3_mean_24_out_val-out3_mean_0_out).norm(2))/4
    # wandb.log({f'L2_GradActivateion/LayerL': L2_GradActivateion_layerL}, step=epoch)
    # wandb.log({f'L2_GradActivateion/LayerL_val': L2_GradActivateion_layerL_val}, step=epoch)

    wandb.finish()

# num_processes = 416
# EVAL_ENVS = {'train_eval': ['h_bandit-obs-randchoose-v14', 416],
#              'valid_eval': ['h_bandit-obs-randchoose-v9', 25],
#              'test_eval' : ['h_bandit-obs-randchoose-v1', 100]}
#
# print('making envs...')
# eval_envs_dic = {}
# eval_locations_dic = {}
# for eval_disp_name, eval_env_name in EVAL_ENVS.items():
#     eval_locations_dic[eval_disp_name] = np.random.randint(0, 6, size=num_processes)
#
# for eval_disp_name, eval_env_name in EVAL_ENVS.items():
#     eval_envs_dic[eval_disp_name] = make_vec_envs(eval_env_name[0], seed, num_processes,
#                                                   eval_locations_dic[eval_disp_name],
#                                                   None, None, device, True, steps=task_steps,
#                                                   recurrent=False,
#                                                   obs_recurrent=True, multi_task=True,
#                                                   free_exploration=6,
#                                                   normalize=not True, rotate=False,
#                                                   obs_rand_loc=False)
#
# wandb.init(project="main_dqn_offline_maximum_entropy_GradPlot", entity="ev_zisselman", name=f'416_envs')
# for epoch in range(0, 5978, 43):
#     q_network_weighs = torch.load('/home/ev/Desktop/reinforce_atten_bandits/dqn_logs/dqn_runs_offline/offline_train_winsorized_400h_bandit-obs-randchoose-v14_0_num_arms_416_03-09-2022_15-58-15/h_bandit-obs-randchoose-v14-epoch-{}.pt'.format(epoch),map_location=device)
#     agent.q_network.load_state_dict(q_network_weighs['state_dict'])
#
#     eval_r = {}
#     for eval_disp_name, eval_env_name in EVAL_ENVS.items():
#         eval_r[eval_disp_name], eval_actions, num_uniform = evaluate(agent, eval_envs_dic, eval_disp_name,
#                                                                      eval_locations_dic,
#                                                                      num_processes,
#                                                                      eval_env_name[1],
#                                                                      steps=task_steps,
#                                                                      recurrent=False,
#                                                                      obs_recurrent=True,
#                                                                      multi_task=True,
#                                                                      free_exploration=6)
#         wandb.log({f'eval_epoch/{eval_disp_name}': np.mean(eval_r[eval_disp_name])}, step=epoch)
#         wandb.log({f'entropy eval/{eval_disp_name}': num_uniform / max(eval_env_name[1], num_processes)}, step=epoch)
#
# # wandb.finish()
#
#
# # for epoch in range(0, 2667, 43):
#     grads = torch.load('/home/ev/Desktop/reinforce_atten_bandits/dqn_logs/dqn_runs_offline/offline_train_winsorized_h_bandit-obs-randchoose-v8_0_num_arms_25_03-09-2022_15-58-09/grads/h_bandit-obs-randchoose-v9-epoch-{}-optimizer_grad.pt'.format(epoch), map_location=device)
#     mean_grads = sum(grads['grad_ens'])/len(grads['grad_ens'])
#
#
#     grads400 = torch.load('/home/ev/Desktop/reinforce_atten_bandits/dqn_logs/dqn_runs_offline/offline_train_winsorized_400h_bandit-obs-randchoose-v14_0_num_arms_416_03-09-2022_15-58-15/grads/h_bandit-obs-randchoose-v9-epoch-{}-optimizer_grad.pt'.format(epoch), map_location=device)
#     grads400 = _unflatten_grad(grads400['grad_ens'][0], shapes)
#
#     Wh1_i = mean_grads[0]
#     cosine_Wh1_i = (Wh1_i * grads400[0]).sum() / (Wh1_i.norm(2) * grads400[0].norm(2))
#     wandb.log({f'eval_Wh1': cosine_Wh1_i}, step=epoch)
#
#     Wh2_i = mean_grads[1]
#     cosine_Wh2_i = (Wh2_i * grads400[1]).sum() / (Wh2_i.norm(2) * grads400[1].norm(2))
#     wandb.log({f'eval_Wh2': cosine_Wh2_i}, step=epoch)
#
#     bh1_i = mean_grads[2]
#     cosine_bh1_i = (bh1_i * grads400[2]).sum() / (bh1_i.norm(2) * grads400[2].norm(2))
#     wandb.log({f'eval_bh1': cosine_bh1_i}, step=epoch)
#
#     bh2_i = mean_grads[3]
#     cosine_bh2_i = (bh2_i * grads400[3]).sum() / (bh2_i.norm(2) * grads400[3].norm(2))
#     wandb.log({f'eval_bh2': cosine_bh2_i}, step=epoch)
#
#     L1w_i = mean_grads[4]
#     cosine_L1w_i = (L1w_i * grads400[4]).sum() / (L1w_i.norm(2) * grads400[4].norm(2))
#     wandb.log({f'eval_L1w': cosine_L1w_i}, step=epoch)
#
#     L1b_i = mean_grads[5]
#     cosine_L1b_i = (L1b_i * grads400[5]).sum() / (L1b_i.norm(2) * grads400[5].norm(2))
#     wandb.log({f'eval_L1b': cosine_L1b_i}, step=epoch)
#
#     L2w_i = mean_grads[6]
#     cosine_L2w_i = (L2w_i * grads400[6]).sum() / (L2w_i.norm(2) * grads400[6].norm(2))
#     wandb.log({f'eval_L2w': cosine_L2w_i}, step=epoch)
#
#     L2b_i = mean_grads[7]
#     cosine_L2b_i = (L2b_i * grads400[7]).sum() / (L2b_i.norm(2) * grads400[7].norm(2))
#     wandb.log({f'eval_L2b': cosine_L2b_i}, step=epoch)
#
#     LLw_i = mean_grads[8]
#     cosine_LLw_i = (LLw_i * grads400[8]).sum() / (LLw_i.norm(2) * grads400[8].norm(2))
#     wandb.log({f'eval_LLw': cosine_LLw_i}, step=epoch)
#
#     LLb_i = mean_grads[9]
#     cosine_LLb_i = (LLb_i * grads400[9]).sum() / (LLb_i.norm(2) * grads400[9].norm(2))
#     wandb.log({f'eval_LLb': cosine_LLb_i}, step=epoch)
#
# wandb.finish()

# for i in range(25):
#
#     wandb.init(project="main_dqn_offline_maximum_entropy_GradPlot", entity="ev_zisselman", name=f'rotated_25_{i}')
#
#     for epoch in range(0,2667,86):
#         grads = torch.load('/home/ev/Desktop/reinforce_atten_bandits/dqn_logs/dqn_runs_offline/offline_train_winsorized_testh_bandit-obs-randchoose-v8_0_num_arms_25_31-08-2022_13-13-15_rotate/grads/h_bandit-obs-randchoose-v9-epoch-{}-optimizer_grad.pt'.format(epoch), map_location=device)
#         grads = grads['grad_ens']
#
#         grads400 = torch.load('/home/ev/Desktop/reinforce_atten_bandits/dqn_logs/dqn_runs_offline/offline_train_winsorized_testh_bandit-obs-randchoose-v14_0_num_arms_416_31-08-2022_15-58-29_rotate/grads/h_bandit-obs-randchoose-v9-epoch-{}-optimizer_grad.pt'.format(epoch), map_location=device)
#         grads400 = _unflatten_grad(grads400['grad_ens'][0], shapes)
#
#
#
#         grads_i = _unflatten_grad(grads[i],shapes)
#
#         Wh1_i = grads_i[0]
#         cosine_Wh1_i = (Wh1_i * grads400[0]).sum() / (Wh1_i.norm(2) * grads400[0].norm(2))
#         wandb.log({f'eval_Wh1': cosine_Wh1_i}, step=epoch)
#
#         Wh2_i = grads_i[1]
#         cosine_Wh2_i = (Wh2_i * grads400[1]).sum() / (Wh2_i.norm(2) * grads400[1].norm(2))
#         wandb.log({f'eval_Wh2': cosine_Wh2_i}, step=epoch)
#
#         bh1_i = grads_i[2]
#         cosine_bh1_i = (bh1_i * grads400[2]).sum() / (bh1_i.norm(2) * grads400[2].norm(2))
#         wandb.log({f'eval_bh1': cosine_bh1_i}, step=epoch)
#
#         bh2_i = grads_i[3]
#         cosine_bh2_i = (bh2_i * grads400[3]).sum() / (bh2_i.norm(2) * grads400[3].norm(2))
#         wandb.log({f'eval_bh2': cosine_bh2_i}, step=epoch)
#
#         L1w_i = grads_i[4]
#         cosine_L1w_i = (L1w_i * grads400[4]).sum() / (L1w_i.norm(2) * grads400[4].norm(2))
#         wandb.log({f'eval_L1w': cosine_L1w_i}, step=epoch)
#
#         L1b_i = grads_i[5]
#         cosine_L1b_i = (L1b_i * grads400[5]).sum() / (L1b_i.norm(2) * grads400[5].norm(2))
#         wandb.log({f'eval_L1b': cosine_L1b_i}, step=epoch)
#
#         L2w_i = grads_i[6]
#         cosine_L2w_i = (L2w_i * grads400[6]).sum() / (L2w_i.norm(2) * grads400[6].norm(2))
#         wandb.log({f'eval_L2w': cosine_L2w_i}, step=epoch)
#
#         L2b_i = grads_i[7]
#         cosine_L2b_i = (L2b_i * grads400[7]).sum() / (L2b_i.norm(2) * grads400[7].norm(2))
#         wandb.log({f'eval_L2b': cosine_L2b_i}, step=epoch)
#
#         LLw_i = grads_i[8]
#         cosine_LLw_i = (LLw_i * grads400[8]).sum() / (LLw_i.norm(2) * grads400[8].norm(2))
#         wandb.log({f'eval_LLw': cosine_LLw_i}, step=epoch)
#
#         LLb_i = grads_i[9]
#         cosine_LLb_i = (LLb_i * grads400[9]).sum() / (LLb_i.norm(2) * grads400[9].norm(2))
#         wandb.log({f'eval_LLb': cosine_LLb_i}, step=epoch)
#
#     wandb.finish()
