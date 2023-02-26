
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

device = torch.device("cuda:{}".format(0))


env_name = "jumper"
start_level = 256
num_level = 1
distribution_mode = "easy"
seed = 0
normalize_rew = False
no_normalize = False
n_steps = 1
epsilon = 1e-5
epsilon_NN = 10

# device = torch.device("cuda:{}".format(0))

print('making envs...')
# Training envs
envs = make_ProcgenEnvs(num_envs=1,
                        env_name=env_name,
                        start_level=start_level,
                        num_levels=num_level,
                        distribution_mode=distribution_mode,
                        use_generated_assets=False,
                        use_backgrounds=True,
                        restrict_themes=False,
                        use_monochrome_assets=False,
                        rand_seed=seed,
                        mask_size=0,
                        normalize_rew=normalize_rew,
                        mask_all=False)

# print(envs.action_space)

obs = envs.reset()
# obs[0] = (obs[0][1,:,:] != 1.0)*obs[0]
# obs_sum = obs
# obs0 = obs.clone()
# diff_obs = torch.zeros_like(obs)
obs_stack = obs.clone()
# plot mazes
plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
plt.show()

action = np.array([5])
obs, reward, done, infos = envs.step(action)
# obs[0] = (obs[0][1,:,:] != 1.0)*obs[0]
# obs_sum += obs
# n_steps +=1
plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
plt.show()

norm2_dis = (obs_stack - obs).reshape(2,-1).pow(2).sum(1)
int_reward = norm2_dis.min()
print('inr reward: {}'.format(int_reward))
obs_stack = torch.cat((obs_stack,obs.clone()),0)
# init = int(max(0,n_steps-5))
# next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
# diff_obs_vec.append(next_diff_obs.clone())
# plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# # plt.savefig("test.png")
# plt.show()
# print((next_diff_obs-diff_obs).sum())
# int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
# print("int reward: {}".format(int_reward) )
# diff_obs = next_diff_obs

action = np.array([5])
obs, reward, done, infos = envs.step(action)
# obs_sum += obs
# n_steps +=1
# plt.imshow(((obs[0][1,:,:] != 1.0)*obs[0]).transpose(0,2).transpose(0,1).cpu().numpy())
plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
plt.show()
norm2_dis = (obs_stack - obs).reshape(2,-1).pow(2).sum(1)
int_reward = norm2_dis.min()
print('inr reward: {}'.format(int_reward))
obs_stack = torch.cat((obs_stack,obs.clone()),0)
# init = int(max(0,n_steps-5))
# next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
# diff_obs_vec.append(next_diff_obs.clone())
# plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# # plt.savefig("test.png")
# plt.show()
# print((next_diff_obs-diff_obs).sum())
# int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
# print("int reward: {}".format(int_reward) )
# diff_obs = next_diff_obs


action = np.array([5])
obs, reward, done, infos = envs.step(action)
# obs_sum += obs
# n_steps +=1
# plt.imshow(((obs[0][1,:,:] != 1.0)*obs[0]).transpose(0,2).transpose(0,1).cpu().numpy())
plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
plt.show()
norm2_dis = (obs_stack - obs).reshape(2,-1).pow(2).sum(1)
int_reward = norm2_dis.min()
print('inr reward: {}'.format(int_reward))
obs_stack = torch.cat((obs_stack,obs.clone()),0)
# init = int(max(0,n_steps-5))
# next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
# diff_obs_vec.append(next_diff_obs.clone())
# plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
# plt.show()
# print((next_diff_obs-diff_obs).sum())
# int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
# print("int reward: {}".format(int_reward) )
# diff_obs = next_diff_obs

action = np.array([5])
obs, reward, done, infos = envs.step(action)
# obs_sum += obs
# n_steps +=1
# plt.imshow(((obs[0][1,:,:] != 1.0)*obs[0]).transpose(0,2).transpose(0,1).cpu().numpy())
plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
plt.show()
norm2_dis = (obs_stack - obs).reshape(2,-1).pow(2).sum(1)
int_reward = norm2_dis.min()
print('inr reward: {}'.format(int_reward))
obs_stack = torch.cat((obs_stack,obs.clone()),0)
# init = int(max(0,n_steps-5))
# next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
# diff_obs_vec.append(next_diff_obs.clone())
# plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
# plt.show()
# print((next_diff_obs-diff_obs).sum())
# int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
# print("int reward: {}".format(int_reward) )
# diff_obs = next_diff_obs

action = np.array([5])
obs, reward, done, infos = envs.step(action)
# obs_sum += obs
# n_steps +=1
# plt.imshow(((obs[0][1,:,:] != 1.0)*obs[0]).transpose(0,2).transpose(0,1).cpu().numpy())
plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
plt.show()
norm2_dis = (obs_stack - obs).reshape(2,-1).pow(2).sum(1)
int_reward = norm2_dis.min()
print('inr reward: {}'.format(int_reward))
obs_stack = torch.cat((obs_stack,obs.clone()),0)
# init = int(max(0,n_steps-5))
# next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
# diff_obs_vec.append(next_diff_obs.clone())
# plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
# plt.show()
# print((next_diff_obs-diff_obs).sum())
# int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
# print("int reward: {}".format(int_reward) )
# diff_obs = next_diff_obs

action = np.array([5])
obs, reward, done, infos = envs.step(action)
# obs_sum += obs
# n_steps +=1
# plt.imshow(((obs[0][1,:,:] != 1.0)*obs[0]).transpose(0,2).transpose(0,1).cpu().numpy())
plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
plt.show()
norm2_dis = (obs_stack - obs).reshape(2,-1).pow(2).sum(1)
int_reward = norm2_dis.min()
print('inr reward: {}'.format(int_reward))
obs_stack = torch.cat((obs_stack,obs.clone()),0)
# init = int(max(0,n_steps-5))
# next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
# diff_obs_vec.append(next_diff_obs.clone())
# plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# # plt.savefig("test.png")
# plt.show()
# print((next_diff_obs-diff_obs).sum())
# int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
# print("int reward: {}".format(int_reward) )
# diff_obs = next_diff_obs

action = np.array([5])
obs, reward, done, infos = envs.step(action)
# obs_sum += obs
# n_steps +=1
# plt.imshow(((obs[0][1,:,:] != 1.0)*obs[0]).transpose(0,2).transpose(0,1).cpu().numpy())
plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
plt.show()
norm2_dis = (obs_stack - obs).reshape(2,-1).pow(2).sum(1)
int_reward = norm2_dis.min()
print('inr reward: {}'.format(int_reward))
obs_stack = torch.cat((obs_stack,obs.clone()),0)

action = np.array([5])
obs, reward, done, infos = envs.step(action)
# obs_sum += obs
# n_steps +=1
# plt.imshow(((obs[0][1,:,:] != 1.0)*obs[0]).transpose(0,2).transpose(0,1).cpu().numpy())
plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
plt.show()
norm2_dis = (obs_stack - obs).reshape(2,-1).pow(2).sum(1)
int_reward = norm2_dis.min()
print('inr reward: {}'.format(int_reward))
obs_stack = torch.cat((obs_stack,obs.clone()),0)

action = np.array([5])
obs, reward, done, infos = envs.step(action)
# obs_sum += obs
# n_steps +=1
# plt.imshow(((obs[0][1,:,:] != 1.0)*obs[0]).transpose(0,2).transpose(0,1).cpu().numpy())
plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
plt.show()
norm2_dis = (obs_stack - obs).reshape(2,-1).pow(2).sum(1)
int_reward = norm2_dis.min()
print('inr reward: {}'.format(int_reward))
obs_stack = torch.cat((obs_stack,obs.clone()),0)

action = np.array([5])
obs, reward, done, infos = envs.step(action)
# obs_sum += obs
# n_steps +=1
# plt.imshow(((obs[0][1,:,:] != 1.0)*obs[0]).transpose(0,2).transpose(0,1).cpu().numpy())
plt.imshow(obs[0].transpose(0,2).transpose(0,1).cpu().numpy())
plt.show()
norm2_dis = (obs_stack - obs).reshape(2,-1).pow(2).sum(1)
int_reward = norm2_dis.min()
print('inr reward: {}'.format(int_reward))
obs_stack = torch.cat((obs_stack,obs.clone()),0)

action = np.array([5])
obs, reward, done, infos = envs.step(action)
obs_sum += obs
n_steps +=1
# plt.imshow(obs[0].transpose(0,2).cpu().numpy())
init = int(max(0,n_steps-5))
next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
diff_obs_vec.append(next_diff_obs.clone())
plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
plt.show()
print((next_diff_obs-diff_obs).sum())
int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
print("int reward: {}".format(int_reward) )
diff_obs = next_diff_obs

action = np.array([5])
obs, reward, done, infos = envs.step(action)
obs_sum += obs
n_steps +=1
# plt.imshow(obs[0].transpose(0,2).cpu().numpy())
init = int(max(0,n_steps-5))
next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
diff_obs_vec.append(next_diff_obs.clone())
plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
plt.show()
print((next_diff_obs-diff_obs).sum())
int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
print("int reward: {}".format(int_reward) )
diff_obs = next_diff_obs

action = np.array([5])
obs, reward, done, infos = envs.step(action)
obs_sum += obs
n_steps +=1
# plt.imshow(obs[0].transpose(0,2).cpu().numpy())
init = int(max(0,n_steps-5))
next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
diff_obs_vec.append(next_diff_obs.clone())
plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
plt.show()
print((next_diff_obs-diff_obs).sum())
int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
print("int reward: {}".format(int_reward) )
diff_obs = next_diff_obs

action = np.array([5])
obs, reward, done, infos = envs.step(action)
obs_sum += obs
n_steps +=1
# plt.imshow(obs[0].transpose(0,2).cpu().numpy())
init = int(max(0,n_steps-5))
next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
diff_obs_vec.append(next_diff_obs.clone())
plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
plt.show()
print((next_diff_obs-diff_obs).sum())
int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
print("int reward: {}".format(int_reward) )
diff_obs = next_diff_obs


action = np.array([5])
obs, reward, done, infos = envs.step(action)
obs_sum += obs
n_steps +=1
# plt.imshow(obs[0].transpose(0,2).cpu().numpy())
init = int(max(0,n_steps-5))
next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
diff_obs_vec.append(next_diff_obs.clone())
plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
plt.show()
print((next_diff_obs-diff_obs).sum())
int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
print("int reward: {}".format(int_reward) )
diff_obs = next_diff_obs

action = np.array([5])
obs, reward, done, infos = envs.step(action)
obs_sum += obs
n_steps +=1
# plt.imshow(obs[0].transpose(0,2).cpu().numpy())
init = int(max(0,n_steps-5))
next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
diff_obs_vec.append(next_diff_obs.clone())
plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
plt.show()
print((next_diff_obs-diff_obs).sum())
int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
print("int reward: {}".format(int_reward) )
diff_obs = next_diff_obs

action = np.array([7])
obs, reward, done, infos = envs.step(action)
obs_sum += obs
n_steps +=1
# plt.imshow(obs[0].transpose(0,2).cpu().numpy())
init = int(max(0,n_steps-5))
next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
diff_obs_vec.append(next_diff_obs.clone())
plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
plt.show()
print((next_diff_obs-diff_obs).sum())
int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
print("int reward: {}".format(int_reward) )
diff_obs = next_diff_obs

action = np.array([7])
obs, reward, done, infos = envs.step(action)
obs_sum += obs
n_steps +=1
# plt.imshow(obs[0].transpose(0,2).cpu().numpy())
init = int(max(0,n_steps-5))
next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
diff_obs_vec.append(next_diff_obs.clone())
plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
plt.show()
print((next_diff_obs-diff_obs).sum())
int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
print("int reward: {}".format(int_reward) )
diff_obs = next_diff_obs

action = np.array([7])
obs, reward, done, infos = envs.step(action)
obs_sum += obs
n_steps +=1
# plt.imshow(obs[0].transpose(0,2).cpu().numpy())
init = int(max(0,n_steps-5))
next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
diff_obs_vec.append(next_diff_obs.clone())
plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
plt.show()
print((next_diff_obs-diff_obs).sum())
int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
print("int reward: {}".format(int_reward) )
diff_obs = next_diff_obs

action = np.array([7])
obs, reward, done, infos = envs.step(action)
obs_sum += obs
n_steps +=1
# plt.imshow(obs[0].transpose(0,2).cpu().numpy())
init = int(max(0,n_steps-5))
next_diff_obs = 1*(((obs_sum/n_steps-obs0)[0] > epsilon) + ((obs_sum/n_steps-obs0)[0] < -epsilon))
diff_obs_vec.append(next_diff_obs.clone())
plt.imshow((255*(next_diff_obs-diff_obs_vec[0])).transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
plt.show()
print((next_diff_obs-diff_obs).sum())
int_reward = 1*((next_diff_obs-diff_obs).sum() > epsilon_NN)
print("int reward: {}".format(int_reward) )
diff_obs = next_diff_obs

print("something")
# for i in range(5):
#     action = np.array([5])
#     obs, reward, done, infos = envs.step(action)
#     # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
#     # plt.show()
#     obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

# for i in range(5):
#     action = np.array([3])
#     obs, reward, done, infos = envs.step(action)
#     # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
#     # plt.show()
#     obs_sum += obs
# # plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# # plt.show()
#
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

action = np.array([2])

obs, _, done, infos = envs.step(action)

plt.imshow(obs[0].transpose(0,2).cpu().numpy())
# plt.savefig("test.png")
plt.show()


actor_critic = Policy(
    envs.observation_space.shape,
    envs.action_space,
    base=ImpalaModel,
    base_kwargs={'recurrent': True, 'hidden_size': 256})
# base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent})
actor_critic.to(device)


saved_epoch = 3124
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_WOr_seed_0_num_env_5000_entro_0.01_04-11-2022_17-35-17"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(
        os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic.load_state_dict(actor_critic_weighs['state_dict'])


# obs = envs.reset()
# obs = test_env.reset()
eval_recurrent_hidden_states = torch.zeros(
    num_level, actor_critic.recurrent_hidden_state_size, device=device)
eval_masks = torch.zeros(num_level, 1, device=device)

eval_attn_masks  = torch.zeros(num_level, actor_critic.attention_size, device=device)
eval_attn_masks1 = torch.zeros(num_level, 16, device=device)
eval_attn_masks2 = torch.zeros(num_level, 32, device=device)
eval_attn_masks3 = torch.zeros(num_level, 32, device=device)

# actor_critic.eval()

done = np.zeros(1)
step = 0
reward = 0

while not done[0]:
    with torch.no_grad():

        obs = torch.FloatTensor(obs)
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

        obs, _, done, infos = envs.step(action[0].cpu().numpy())
        print(action[0])
        # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
        # plt.show()

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        next_obs_sum = obs_sum + obs.cpu()
        num_zero_obs_sum = (obs_sum[0] == 0).sum()
        num_zero_next_obs_sum = (next_obs_sum[0] == 0).sum()
        if num_zero_next_obs_sum < num_zero_obs_sum:
                reward += 1

        obs_sum = next_obs_sum

print("done")
print("reward = {}".format(reward))
print("zero sum = {}".format((obs_sum[0] == 0).sum()))