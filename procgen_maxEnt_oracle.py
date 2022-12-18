
from a2c_ppo_acktr.procgen_wrappers import *
# from a2c_ppo_acktr.envs import make_vec_envs, make_ProcgenEnvs
import matplotlib.pyplot as plt
# from procgen import ProcgenEnv
from a2c_ppo_acktr.envs import make_ProcgenEnvs
from a2c_ppo_acktr.const_env import ProcgenConatEnvs
from a2c_ppo_acktr.model import Policy, MLPAttnBase, MLPHardAttnBase, MLPHardAttnReinforceBase, ImpalaModel
import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

from gym3 import VideoRecorderWrapper
from procgen import ProcgenEnv, ProcgenGym3Env
from gym import spaces

device = torch.device("cuda:{}".format(0))


env_name = "maze"
start_level = 1236
num_level = 1
distribution_mode = "easy"
seed = 0
normalize_rew = False
no_normalize = False
n_steps = 1

# device = torch.device("cuda:{}".format(0))

print('making envs...')
# Training envs
env = ProcgenGym3Env(num=1,
                          env_name=env_name,
                          start_level=start_level,
                          num_levels=num_level,
                          distribution_mode=distribution_mode,
                          render_mode="rgb_array",
                          use_generated_assets=True,
                          use_backgrounds=False,
                          restrict_themes=True,
                          use_monochrome_assets=True)

# test_env = ProcgenGym3Env(num=1,
#                           env_name=env_name,
#                           center_agent=False,
#                           use_generated_assets=True,
#                           use_backgrounds=False,
#                           restrict_themes=True,
#                           use_monochrome_assets=True,
#                           distribution_mode=distribution_mode,
#                           paint_vel_info=True,
#                           start_level=test_start_level,
#                           num_levels=num_level,
#                           render_mode="rgb_array"
#                           )

envs = VideoRecorderWrapper(env=env, directory="./videos", info_key="rgb", prefix=str(start_level), fps=5, render=True)

# print(envs.action_space)

rew, obs, first = envs.observe()
obs = obs['rgb']/255

# plot mazes
# plt.imshow(obs[0].transpose(0,2).cpu().numpy())
# plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
# # plt.savefig("test.png")
# plt.show()
action = np.array([7])

# for i in range(5):
# action = np.array([5])
# obs, reward, done, infos = envs.step(action)
#     # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
#     # plt.show()
#     # obs_sum += obs
# plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
# plt.show()

# for i in range(5):
# action = np.array([3])
# obs, reward, done, infos = envs.step(action)
# plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
# plt.show()
#     obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

# for i in range(6):
#     action = np.array([7])
#     obs, reward, done, infos = envs.step(action)
#     # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
#     # plt.show()
#     obs_sum += obs
# # plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# # plt.show()
#
# for i in range(6):
#     action = np.array([5])
#     obs, reward, done, infos = envs.step(action)
#     # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
#     # plt.show()
#     obs_sum += obs
# # plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# # plt.show()
#
# for i in range(2):
#     action = np.array([1])
#     obs, reward, done, infos = envs.step(action)
#     # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
#     # plt.show()
#     obs_sum += obs
# # plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# # plt.show()
#
# for i in range(5):
#     action = np.array([3])
#     obs, reward, done, infos = envs.step(action)
#     # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
#     # plt.show()
#     obs_sum += obs
# # plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# # plt.show()
#
# for i in range(5):
#     action = np.array([5])
#     obs, reward, done, infos = envs.step(action)
#     # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
#     # plt.show()
#     obs_sum += obs
# # plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# # plt.show()
#
# for i in range(2):
#     action = np.array([1])
#     obs, reward, done, infos = envs.step(action)
#     # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
#     # plt.show()
#     obs_sum += obs
# # plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# # plt.show()
#
# for i in range(5):
#     action = np.array([3])
#     obs, reward, done, infos = envs.step(action)
#     # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
#     # plt.show()
#     obs_sum += obs
# # plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# # plt.show()
#
# for i in range(5):
#     action = np.array([5])
#     obs, reward, done, infos = envs.step(action)
#     # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
#     # plt.show()
#     obs_sum += obs
# # plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# # plt.show()
#
# for i in range(2):
#     action = np.array([1])
#     obs, reward, done, infos = envs.step(action)
#     # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
#     # plt.show()
#     obs_sum += obs
# # plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# # plt.show()

# actor_critic = Policy(
#     envs.observation_space.shape,
#     envs.action_space,
#     base=ImpalaModel,
#     base_kwargs={'recurrent': True, 'hidden_size': 256})
# # base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
# actor_critic.to(device)
#
#
# saved_epoch = 3124
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_WOr_seed_0_num_env_5000_entro_0.01_04-11-2022_17-35-17"
# if (saved_epoch > 0) and save_dir != "":
#     save_path = save_dir
#     actor_critic_weighs = torch.load(
#         os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
#     actor_critic.load_state_dict(actor_critic_weighs['state_dict'])
#
#
# # obs = envs.reset()
# # obs = test_env.reset()
# eval_recurrent_hidden_states = torch.zeros(
#     num_level, actor_critic.recurrent_hidden_state_size, device=device)
# eval_masks = torch.zeros(num_level, 1, device=device)
#
# eval_attn_masks  = torch.zeros(num_level, actor_critic.attention_size, device=device)
# eval_attn_masks1 = torch.zeros(num_level, 16, device=device)
# eval_attn_masks2 = torch.zeros(num_level, 32, device=device)
# eval_attn_masks3 = torch.zeros(num_level, 32, device=device)

# actor_critic.eval()

done = np.zeros(1)
step = 0
reward = 0

obs = obs[0].transpose()
obs = np.transpose(obs, (0, 2, 1))
obs_sum = obs

while not done[0]:
    with torch.no_grad():


        min_r =  np.nonzero((obs[1] == 1))[0].min()
        max_r =  np.nonzero((obs[1] == 1))[0].max()

        min_c =  np.nonzero((obs[1] == 1))[1].min()
        max_c =  np.nonzero((obs[1] == 1))[1].max()

        if action[0] == 7:
            if obs[0][max_r+1,min_c] == 0:
                action = np.array([3])
            elif obs[0][min_r,max_c+1] == 0:
                action = np.array([7])
            elif obs[0][min_r-1, min_c] == 0:
                action = np.array([5])
            else:
                action = np.array([1])
        elif action[0] == 5:
            if obs[0][max_r,max_c+1] == 0:
                action = np.array([7])
            elif obs[0][min_r-1, min_c] == 0:
                action = np.array([5])
            elif obs[0][min_r, min_c-1] == 0:
                action = np.array([1])
            else:
                action = np.array([3])
        elif action[0] == 3:
            if obs[0][min_r, min_c-1] == 0:
                action = np.array([1])
            elif obs[0][max_r+1, min_c] == 0:
                action = np.array([3])
            elif obs[0][max_r,max_c+1] == 0:
                action = np.array([7])
            else:
                action = np.array([5])
        elif action[0] == 1:
            if obs[0][min_r-1, min_c] == 0:
                action = np.array([5])
            elif obs[0][min_r, min_c-1] == 0:
                action = np.array([1])
            elif obs[0][max_r+1, min_c] == 0:
                action = np.array([3])
            else:
                action = np.array([7])

        envs.act(action)
        rew, obs, first = envs.observe()
        obs = obs['rgb']/255
        # print(action[0])
        # plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
        # plt.show()
        obs = obs[0].transpose()
        obs = np.transpose(obs, (0, 2, 1))

        eval_masks = torch.tensor(
        [[0.0] if done_ else [1.0] for done_ in done],
        dtype=torch.float32,
        device=device)

        next_obs_sum = obs_sum + obs
        num_zero_obs_sum = (obs_sum[0] == 0).sum()
        num_zero_next_obs_sum = (next_obs_sum[0] == 0).sum()
        if num_zero_next_obs_sum < num_zero_obs_sum:
                reward += 1

        obs_sum = next_obs_sum
        print(reward)

print("done")
print("reward = {}".format(reward))
print("zero sum = {}".format((obs_sum[0] == 0).sum()))