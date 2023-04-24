
from a2c_ppo_acktr.procgen_wrappers import *
# from a2c_ppo_acktr.envs import make_vec_envs, make_ProcgenEnvs
import matplotlib.pyplot as plt
from procgen import ProcgenEnv, ProcgenGym3Env
# from a2c_ppo_acktr.envs import make_ProcgenEnvs
from a2c_ppo_acktr.const_env import ProcgenConatEnvs
import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
from a2c_ppo_acktr.model import Policy, MLPAttnBase, MLPHardAttnBase, MLPHardAttnReinforceBase, ImpalaModel
from a2c_ppo_acktr import algo, utils
from evaluation import evaluate_procgen

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, FixedCategorical

EVAL_ENVS = ['train_eval','test_eval']

# num_processes = 600
env_name = "dodgeball"
start_level = 2
num_level = 1
# num_level_test = 128
distribution_mode = "easy"
seed = 0
normalize_rew = False
no_normalize = False
n_steps = 1

device = torch.device("cuda:{}".format(0))

# test_env = ProcgenGym3Env(num=1,
#                           env_name=env_name,
#                           start_level=start_level,
#                           num_levels=num_level,
#                           distribution_mode=distribution_mode,
#                           render_mode="rgb_array",
#                           use_generated_assets=False,
#                           center_agent=False,
#                           use_backgrounds=True,
#                           restrict_themes=False,
#                           use_monochrome_assets=False)

test_env2 = ProcgenGym3Env(num=1,
                          env_name=env_name,
                          start_level=start_level,
                          num_levels=num_level,
                          distribution_mode=distribution_mode,
                          render_mode="rgb_array",
                          use_generated_assets=False,
                          center_agent=False,
                          use_backgrounds=False,
                          restrict_themes=False,
                          use_monochrome_assets=False)


rew, obs, first = test_env2.observe()
obs = obs['rgb'].transpose(0, 3, 1, 2)
obs = torch.tensor(obs).float()
obs0 = obs.clone()
obs_sum = obs.clone()
num_steps = 1
average_obs = (obs_sum/num_steps).int()
diff_obs = average_obs-obs0


myobj = plt.imshow(obs[0].transpose(0, 2).transpose(0, 1).int())
plt.show()


# myobj = plt.imshow(diff_obs[0].transpose(0, 2).transpose(0, 1))
# plt.show()
# # myobj = plt.imshow(test_env.get_info()[0]['rgb'])
# # plt.show()

action = np.array([9])
test_env2.act(action)

rew, obs, first = test_env2.observe()
obs = obs['rgb'].transpose(0, 3, 1, 2)
obs = torch.tensor(obs).float()
obs_sum += obs
num_steps += 1
average_obs = (obs_sum/num_steps)
diff_obs = torch.tensor((average_obs - obs0))

myobj = plt.imshow(obs[0].transpose(0, 2).transpose(0, 1).int())
plt.show()
# # myobj = plt.imshow(test_env.get_info()[0]['rgb'])
# # plt.show()
# myobj = plt.imshow(diff_obs[0].transpose(0, 2).transpose(0, 1))
# plt.show()

action = np.array([9])
test_env2.act(action)

rew, obs, first = test_env2.observe()
obs = obs['rgb'].transpose(0, 3, 1, 2)
obs = torch.tensor(obs).float()
obs_sum += obs
num_steps += 1
average_obs = (obs_sum/num_steps)
diff_obs = torch.tensor((average_obs - obs0))


myobj = plt.imshow(obs[0].transpose(0, 2).transpose(0, 1).int())
plt.show()

# myobj = plt.imshow(diff_obs[0].transpose(0, 2).transpose(0, 1))
# plt.show()

action = np.array([9])
test_env2.act(action)

rew, obs, first = test_env2.observe()
obs = obs['rgb'].transpose(0, 3, 1, 2)
obs = torch.tensor(obs)
obs_sum += obs
num_steps += 1
average_obs = (obs_sum/num_steps).int()
diff_obs = torch.tensor((average_obs - obs0))


myobj = plt.imshow(obs[0].transpose(0, 2).transpose(0, 1).int())
plt.show()

# myobj = plt.imshow(diff_obs[0].transpose(0, 2).transpose(0, 1))
# plt.show()

action = np.array([9])
test_env2.act(action)

rew, obs, first = test_env2.observe()
obs = obs['rgb'].transpose(0, 3, 1, 2)
obs = torch.tensor(obs)
obs_sum += obs
num_steps += 1
average_obs = (obs_sum/num_steps).int()
diff_obs = torch.tensor((average_obs - obs0))


myobj = plt.imshow(obs[0].transpose(0, 2).transpose(0, 1).int())
plt.show()

# myobj = plt.imshow(diff_obs[0].transpose(0, 2).transpose(0, 1).int())
# plt.show()

print('stop')




























eval_envs_dic = {}
  # eval_envs_dic['train_eval'] = ProcgenConatEnvs(env_name=env_name,
#                                                num_envs=num_level,
#                                                start_level=start_level,
#                                                distribution_mode=distribution_mode,
#                                                use_generated_assets=True,
#                                                use_backgrounds=False,
#                                                restrict_themes=True,
#                                                use_monochrome_assets=True,
#                                                normalize_rew=normalize_rew,
#                                                num_stack=1,
#                                                seed=0,
#                                                device=device,
#                                                mask_size=0,
#                                                mask_all=False)

eval_envs_dic['train_eval'] = ProcgenEnv(num_envs=1,
                                         env_name=env_name,
                                         start_level=start_level,
                                         num_levels=1,
                                         distribution_mode=distribution_mode,
                                         use_generated_assets=False,
                                         use_backgrounds=True,
                                         restrict_themes=False,
                                         use_monochrome_assets=False,
                                         rand_seed=seed)

obs = eval_envs_dic['train_eval'].reset()
myobj = plt.imshow(obs['rgb'][0])
plt.show()

action = np.array([5])
next_obs, reward, done, infos = eval_envs_dic['train_eval'].step(action)
plt.imshow(next_obs['rgb'][0])
plt.show()

action = np.array([5])
next_obs, reward, done, infos = eval_envs_dic['train_eval'].step(action)
plt.imshow(next_obs['rgb'][0])
plt.show()

action = np.array([5])
next_obs, reward, done, infos = eval_envs_dic['train_eval'].step(action)
plt.imshow(next_obs['rgb'][0])
plt.show()

action = np.array([5])
next_obs, reward, done, infos = eval_envs_dic['train_eval'].step(action)
plt.imshow(next_obs['rgb'][0])
plt.show()

action = np.array([5])
next_obs, reward, done, infos = eval_envs_dic['train_eval'].step(action)
plt.imshow(next_obs['rgb'][0])
plt.show()

action = np.array([5])
next_obs, reward, done, infos = eval_envs_dic['train_eval'].step(action)
plt.imshow(next_obs['rgb'][0])
plt.show()

action = np.array([5])
next_obs, reward, done, infos = eval_envs_dic['train_eval'].step(action)
plt.imshow(next_obs['rgb'][0])
plt.show()

action = np.array([5])
next_obs, reward, done, infos = eval_envs_dic['train_eval'].step(action)
plt.imshow(next_obs['rgb'][0])
plt.show()

action = np.array([5])
next_obs, reward, done, infos = eval_envs_dic['train_eval'].step(action)
plt.imshow(next_obs['rgb'][0])
plt.show()

print("something")
# eval_envs_dic['train_eval'] = []
# for i in range(num_level):
#     eval_envs_dic['train_eval'].append(make_ProcgenEnvs(num_envs=1,
#                                                         env_name=env_name,
#                                                         start_level=start_level+i,
#                                                         num_levels=1,
#                                                         distribution_mode=distribution_mode,
#                                                         use_generated_assets=True,
#                                                         use_backgrounds=False,
#                                                         restrict_themes=True,
#                                                         use_monochrome_assets=True,
#                                                         rand_seed=0,
#                                                         mask_size=0,
#                                                         normalize_rew=False,
#                                                         mask_all=False,
#                                                         device=device))

# test_start_level = 10000
# eval_envs_dic['test_eval']  = ProcgenConatEnvs(env_name=env_name,
#                                                num_envs=num_level_test,
#                                                start_level=test_start_level,
#                                                distribution_mode=distribution_mode,
#                                                use_generated_assets=False,
#                                                use_backgrounds=False,
#                                                restrict_themes=True,
#                                                use_monochrome_assets=True,
#                                                normalize_rew=normalize_rew,
#                                                num_stack=1,
#                                                seed=0,
#                                                device=device,
#                                                mask_size=0,
#                                                mask_all=False)
# eval_envs_dic['test_eval']  = []
# for i in range(num_level):
#     eval_envs_dic['train_eval'].append(make_ProcgenEnvs(num_envs=1,
#                                                         env_name=env_name,
#                                                         start_level=test_start_level+i,
#                                                         num_levels=1,
#                                                         distribution_mode=distribution_mode,
#                                                         use_generated_assets=True,
#                                                         use_backgrounds=False,
#                                                         restrict_themes=True,
#                                                         use_monochrome_assets=True,
#                                                         rand_seed=0,
#                                                         mask_size=0,
#                                                         normalize_rew=False,
#                                                         mask_all=False,
#                                                         device=device))
# obs = eval_envs_dic['train_eval'].reset()
# obs_test = eval_envs_dic['test_eval'].reset()

# fig = plt.figure(figsize=(20, 20))
# columns = 2
# rows = 2
# for i in range(1, columns * rows + 1):
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(obs[i - 1].transpose(1, 2, 0))
# # plt.imshow(test_obs[0,:,:,:].transpose(1, 2, 0))
# plt.show()

# num_outputs = eval_envs_dic['train_eval'].action_space.n
# log_probs = (1/9) * torch.ones(num_processes, 9)
# dist = torch.distributions.Categorical(logits=log_probs)

# actor_critic = Policy(
#     eval_envs_dic['train_eval'].observation_space.shape,
#     eval_envs_dic['train_eval'].action_space,
#     base=ImpalaModel,
#     base_kwargs={'recurrent': False})
# actor_critic.to(device)

# # training agent
# agent = algo.PPO(
#     actor_critic,
#     0.2,
#     3,
#     8,
#     0.5,
#     0.01,
#     lr=0.0005,
#     eps=1e-05,
#     num_tasks=num_level,
#     attention_policy=False,
#     max_grad_norm=0.5,
#     weight_decay=0)

# # Load previous model
# continue_from_epoch = 3124
# save_dir = '/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_0_num_env_200_entro_0.01_gama_0.8_01-01-2023_21-07-10'
# if (continue_from_epoch >= 0) and save_dir != "":
#     save_path = save_dir
#     actor_critic_weighs = torch.load(
#         os.path.join(save_path, env_name + "-epoch-{}.pt".format(continue_from_epoch)), map_location=device)
#     actor_critic.load_state_dict(actor_critic_weighs['state_dict'])
#     # agent.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict'])


# seeds_train = torch.zeros(num_processes, 1)
# seeds_test = torch.zeros(num_processes, 1)
#
#
# for _ in range(n_steps):
#
#     act = dist.sample().numpy()
#     next_obs, rew, done, infos = eval_envs_dic['train_eval'].step(act)
#     obs = next_obs
#
#     for i, info in enumerate(infos):
#         seeds_train[i] = info["level_seed"]
#
#     output, inverse_indices, counts  = torch.unique(seeds_train, sorted=True, return_inverse=True,return_counts=True)
#     print(output.shape)
#
#     fig = plt.figure(figsize=(28, 28))
#     columns = 8
#     rows = 16
#     for i in range(1, columns * rows + 1):
#         ax = fig.add_subplot(rows, columns, i)
#         ax.title.set_text(str(i-1))
#         unique_obs = obs[(seeds_train == i-1).squeeze(), :, :, :][0]
#         plt.imshow(unique_obs.transpose(0, 2).transpose(0, 1))
#     plt.show()
#
#     next_obs_test, rew_test, done_test, infos_test = eval_envs_dic['test_eval'].step(act)
#     obs_test = next_obs_test
#
#     for i, info in enumerate(infos_test):
#         seeds_test[i] = info["level_seed"]
#
#     output_test, inverse_indices, counts = torch.unique(seeds_test, sorted=True, return_inverse=True, return_counts=True)
#     print(output_test.shape)
#
#     fig = plt.figure(figsize=(28, 28))
#     columns = 8
#     rows = 16
#     for i in range(1, columns * rows + 1):
#         ax = fig.add_subplot(rows, columns, i)
#         ax.title.set_text(str(test_start_level + i-1))
#         unique_obs_test = obs_test[(seeds_test == test_start_level + i-1).squeeze(), :, :, :][0]
#         plt.imshow(unique_obs_test.transpose(0, 2))
#     plt.show()
#
#     print(output.shape)


actor_critic.eval()
eval_dic_rew = {}
eval_dic_done = {}
eval_dic_seeds = {}
eval_dic_action = {}
num_steps = 512

episode_rewards = []
# for _ in range(num_level):
#     episode_rewards.append([])

episode_len_buffer = {}
episode_len_buffer['train_eval']  = []
for _ in range(num_level):
    episode_len_buffer['train_eval'].append([])
episode_len_buffer['test_eval']  = []
for _ in range(num_level):
    episode_len_buffer['test_eval'].append([])

episode_len_buffer_all = {}
episode_len_buffer_all['train_eval']  = deque(maxlen = num_level)
episode_len_buffer_all['test_eval']  = deque(maxlen = num_level)

episode_reward_buffer = {}
episode_reward_buffer['train_eval'] = []
for _ in range(num_level):
    episode_reward_buffer['train_eval'].append([])
episode_reward_buffer['test_eval'] = []
for _ in range(num_level):
    episode_reward_buffer['test_eval'].append([])

episode_reward_buffer_all = {}
episode_reward_buffer_all['train_eval']  = deque(maxlen = num_level)
episode_reward_buffer_all['test_eval']  = deque(maxlen = num_level)

num_episodes = {}
num_episodes['train_eval'] = []
for _ in range(num_level):
    num_episodes['train_eval'].append(0)
num_episodes['test_eval'] = []
for _ in range(num_level):
    num_episodes['test_eval'].append(0)

# num_episodes_all = {}
# num_episodes_all['train_eval']  = deque(maxlen = num_level)
# num_episodes_all['test_eval']  = deque(maxlen = num_level)


for env_name in EVAL_ENVS:
    # eval_dic_rew[env_name], eval_dic_done[env_name] = evaluate_procgen(actor_critic, eval_envs_dic,
    #                                                                                eval_disp_name,
    #                                                                                num_processes, device,
    #                                                                                num_steps, deterministic=True)
    eval_envs = eval_envs_dic[env_name]
    rew_batch = []
    done_batch = []
    seeds_batch = []
    action_batch = []
    if env_name == 'test_eval':
        num_level = num_level_test


    # if env_name == 'train_eval':
    #     min_level = 0
    #     max_level = num_level
    # else:
    #     min_level = test_start_level
    #     max_level = test_start_level + num_level
    #
    # levels = list(range(min_level,max_level))
    #
    # obs = eval_envs[min(levels)].reset()
    # for i in levels[1:]:
    #     one_obs = eval_envs[i].reset()
    #     obs = torch.cat((obs, one_obs), 0)

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_level, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_level, 1, device=device)

    eval_attn_masks  = torch.zeros(num_level, actor_critic.attention_size, device=device)
    eval_attn_masks1 = torch.zeros(num_level, 16, device=device)
    eval_attn_masks2 = torch.zeros(num_level, 32, device=device)
    eval_attn_masks3 = torch.zeros(num_level, 32, device=device)

    for t in range(num_steps):
        if obs.shape[0]>0:
            with torch.no_grad():
                if t == 0:
                    action = (4 * torch.ones(num_level,1)).int()
                else:
                    # eval_recurrent_hidden_states = eval_recurrent_hidden_states_all[levels]
                    # eval_masks = torch.ones(len(levels), 1, device=device)
                    _, action, _, eval_recurrent_hidden_states, _, _, _, _ = actor_critic.act(
                        obs.float().to(device),
                        eval_recurrent_hidden_states,
                        eval_masks,
                        attn_masks=eval_attn_masks,
                        attn_masks1=eval_attn_masks1,
                        attn_masks2=eval_attn_masks2,
                        attn_masks3=eval_attn_masks3,
                        deterministic=True,
                        reuse_masks=False)

                    # eval_recurrent_hidden_states_all[levels] = eval_recurrent_hidden_states
                # Observe reward and next obs
                next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
                # next_obs, reward, done, infos = eval_envs[min(levels)].step(action[0].cpu().numpy())
                # for i, ind in enumerate(levels[1:]):
                #     one_next_obs, one_reward, one_done, one_infos = eval_envs[ind].step(action[i+1].cpu().numpy())
                #     next_obs = torch.cat((next_obs, one_next_obs), 0)
                #     reward = np.concatenate((reward, one_reward), 0)
                #     done = np.concatenate((done, one_done), 0)
                #     infos.append(one_infos[0])

                    # if one_done:
                    #     episode_len_buffer[env_name].append(t)
                    #     episode_reward_buffer[env_name].append(one_reward)
                    #     num_episodes[env_name] += 1
                    #     levels.remove(one_infos[0]['prev_level_seed'])

                eval_masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=device)

                if 'env_reward' in infos[0]:
                    rew_batch.append([info['env_reward'] for info in infos])
                else:
                    rew_batch.append(reward)
                seeds = [info['level_seed'] for info in infos]
                seeds_batch.append(seeds)
                done_batch.append(done)
                action_batch.append(action.squeeze().cpu().numpy())

                # for i, info in enumerate(infos):
                #     eval_episode_len_buffer[i] += 1
                #     if done[i] == True:
                #         eval_episode_rewards.append(reward[i])
                #         eval_episode_len.append(eval_episode_len_buffer[i])
                #         eval_episode_len_buffer[i] = 0

                obs = next_obs

                if t < n_steps:
                    start_level = min(seeds)
                    output, inverse_indices, counts  = torch.unique(torch.tensor(seeds), sorted=True, return_inverse=True,return_counts=True)
                    print(output.shape)
                    print(start_level)

                    if env_name == 'test_eval':
                        fig = plt.figure(figsize=(28, 28))
                        columns = 8
                        rows = 16
                        for i in range(1, columns * rows + 1):
                            ax = fig.add_subplot(rows, columns, i)
                            ax.title.set_text(str(start_level + i - 1))
                            unique_obs = obs[(seeds == start_level + i - 1).squeeze(), :, :, :][0]
                            plt.imshow(unique_obs.transpose(0, 2).transpose(0, 1).cpu())
                        plt.show()
                    else:
                        fig = plt.figure(figsize=(28, 28))
                        columns = 20
                        rows = 20
                        for i in range(1, columns * rows + 1):
                            ax = fig.add_subplot(rows, columns, i)
                            ax.title.set_text(str(start_level + i - 1))
                            unique_obs = obs[(seeds == start_level + i - 1).squeeze(), :, :, :][0]
                            plt.imshow(unique_obs.transpose(0, 2).transpose(0, 1).cpu())
                        plt.show()


    eval_dic_rew[env_name] = np.array(rew_batch)
    eval_dic_done[env_name]  = np.array(done_batch)
    eval_dic_seeds[env_name]  = np.array(seeds_batch)
    eval_dic_action[env_name]  = np.array(action_batch)

    #evaluate statistics
    rew_batch = eval_dic_rew[env_name].T
    done_batch = eval_dic_done[env_name].T
    for i in range(num_level):
        for j in range(num_steps):
            episode_rewards.append(rew_batch[i][j])
            if done_batch[i][j]:
                episode_len_buffer[env_name][i].append(len(episode_rewards))
                episode_reward_buffer[env_name][i].append(np.sum(episode_rewards))
                episode_rewards = []
                num_episodes[env_name][i] += 1

        episode_len_buffer_all[env_name].append(np.mean(episode_len_buffer[env_name][i]))
        episode_reward_buffer_all[env_name].append(np.mean(episode_reward_buffer[env_name][i]))

print('train reward: {}, train length: {}, num_episodes: {}'.format(np.mean(episode_reward_buffer_all['train_eval']),np.mean(episode_len_buffer_all['train_eval']),np.mean(num_episodes['train_eval'])))
print('test reward: {}, test length: {}, num_episodes: {}'.format(np.mean(episode_reward_buffer_all['test_eval']),np.mean(episode_len_buffer_all['test_eval']),np.mean(num_episodes['test_eval'])))
print('done')



