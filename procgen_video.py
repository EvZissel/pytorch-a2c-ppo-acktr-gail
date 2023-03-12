from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.vec_env import VecVideoRecorder

from a2c_ppo_acktr.procgen_wrappers import *
# from a2c_ppo_acktr.envs import make_vec_envs, make_ProcgenEnvs
# import matplotlib.pyplot as plt
# from a2c_ppo_acktr.envs import make_ProcgenEnvs
# from a2c_ppo_acktr.const_env import ProcgenConatEnvs
import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
from a2c_ppo_acktr.model import Policy, MLPAttnBase, MLPHardAttnBase, MLPHardAttnReinforceBase, ImpalaModel
from a2c_ppo_acktr import algo, utils
from evaluation import evaluate_procgen
import matplotlib.pyplot as plt

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, FixedCategorical
from gym3 import VideoRecorderWrapper
from procgen import ProcgenEnv, ProcgenGym3Env
from gym import spaces
# from gym3.video_recorder import VideoRecorderWrapper

EVAL_ENVS = ['train_eval','test_eval']

# num_processes = 600
env_name = "jumper"
start_level = 0
num_level = 1
distribution_mode = "easy"
seed = 1
normalize_rew = False
no_normalize = True
# n_steps = 1

device = torch.device("cuda:{}".format(0))

class VideoRecorderprocess(VideoRecorderWrapper):
    def __init__(self, env, directory, info_key, prefix, fps ,render):
        super().__init__(env=env, directory=directory, info_key=info_key, prefix=prefix, fps=fps, render=render)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        obs = frame
        # indexes = (obs[:, :, 0] == 63)*(obs[:, :, 1] == 255)*(obs[:, :, 2] == 63)
        # obs[:, :, 0] = (obs[:, :, 0] * (1 - indexes))
        # obs[:, :, 1] = (obs[:, :, 1] * (1 - indexes))
        # obs[:, :, 2] = (obs[:, :, 2] * (1 - indexes))
        #
        # # obs[50:55,43:47,0] = 127
        # # obs[50:55,43:47,1] = 127
        # # obs[50:55,43:47,2] = 255
        # # obs[43:48,17:21,0] = 127
        # # obs[43:48,17:21,1] = 127
        # # obs[43:48,17:21,2] = 255
        # obs[9:13, 9:14, 0] = 127
        # obs[9:13, 9:14, 1] = 127
        # obs[9:13, 9:14, 2] = 255
        # obs = obs.astype(np.uint8)
        # r = obs[:, :, 0]
        # g = obs[:, :, 1]
        # b = obs[:, :, 2]
        # res_obs = np.zeros_like(obs)
        # res_obs[:, :, 0] = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # res_obs[:, :, 1] = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # res_obs[:, :, 2] = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return obs

# test_start_level = 201 #bi maze
test_start_level = 8
# test_env  = ProcgenConatEnvs(env_name=env_name,
#                              num_envs=num_level,
#                              start_level=test_start_level,
#                              distribution_mode=distribution_mode,
#                              use_generated_assets=True,
#                              use_backgrounds=False,
#                              restrict_themes=True,
#                              use_monochrome_assets=True,
#                              normalize_rew=normalize_rew,
#                              num_stack=1,
#                              seed=0,
#                              device=device,
#                              mask_size=0,
#                              mask_all=False)

test_env = ProcgenGym3Env(num=1,
                          env_name=env_name,
                          start_level=test_start_level,
                          num_levels=num_level,
                          distribution_mode=distribution_mode,
                          render_mode="rgb_array",
                          use_generated_assets=False,
                          center_agent=True,
                          use_backgrounds=True,
                          restrict_themes=False,
                          use_monochrome_assets=False,
                          rand_seed=seed)

test_env_full_obs = ProcgenGym3Env(num=1,
                          env_name=env_name,
                          start_level=test_start_level,
                          num_levels=num_level,
                          distribution_mode=distribution_mode,
                          render_mode="rgb_array",
                          use_generated_assets=True,
                          center_agent=False,
                          use_backgrounds=False,
                          restrict_themes=True,
                          use_monochrome_assets=True,
                          rand_seed=seed)

test_env = VideoRecorderprocess(env=test_env, directory="./videos", info_key="rgb", prefix=str(test_start_level), fps=5, render=True)

actor_critic_maxEnt = Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': True ,'hidden_size': 256})
actor_critic_maxEnt.to(device)

actor_critic1 = Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': False ,'hidden_size': 256})
actor_critic1.to(device)

actor_critic2 = Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': False ,'hidden_size': 256})
actor_critic2.to(device)

actor_critic3 = Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': False ,'hidden_size': 256})
actor_critic3.to(device)

actor_critic4= Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': False ,'hidden_size': 256})
actor_critic4.to(device)

# actor_critic5= Policy(
#     (3,64,64),
#     spaces.Discrete(15),
#     base=ImpalaModel,
#     base_kwargs={'recurrent': False ,'hidden_size': 256})
# actor_critic5.to(device)
#
# actor_critic6= Policy(
#     (3,64,64),
#     spaces.Discrete(15),
#     base=ImpalaModel,
#     base_kwargs={'recurrent': False ,'hidden_size': 256})
# actor_critic6.to(device)
#
# actor_critic7= Policy(
#     (3,64,64),
#     spaces.Discrete(15),
#     base=ImpalaModel,
#     base_kwargs={'recurrent': False ,'hidden_size': 256})
# actor_critic7.to(device)
#
# actor_critic8= Policy(
#     (3,64,64),
#     spaces.Discrete(15),
#     base=ImpalaModel,
#     base_kwargs={'recurrent': False ,'hidden_size': 256})
# actor_critic8.to(device)
#
# actor_critic9= Policy(
#     (3,64,64),
#     spaces.Discrete(15),
#     base=ImpalaModel,
#     base_kwargs={'recurrent': False ,'hidden_size': 256})
# actor_critic9.to(device)
#
# actor_critic10= Policy(
#     (3,64,64),
#     spaces.Discrete(15),
#     base=ImpalaModel,
#     base_kwargs={'recurrent': False ,'hidden_size': 256})
# actor_critic10.to(device)

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

# Load previous model
saved_epoch = 3050

# # save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/heist_seed_58967_num_env_200_entro_0.01_gama_0.5_05-02-2023_22-24-49_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/heist_seed_58967_num_env_200_entro_0.01_gama_0.5_05-02-2023_18-15-27_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_0_num_env_200_entro_0.01_gama_0.5_25-12-2022_00-56-16"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_2865_num_env_200_entro_0.01_gama_0.5_07-02-2023_15-10-02_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_2865_num_env_200_entro_0.01_gama_0.5_07-02-2023_19-36-39_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_2865_num_env_200_entro_0.01_gama_0.5_07-02-2023_19-40-00_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_58967_num_env_200_entro_0.01_gama_0.5_04-02-2023_15-46-16_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_3569_num_env_200_entro_0.01_gama_0.5_08-02-2023_23-48-07_original"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_3569_num_env_200_entro_0.01_gama_0.9_24-02-2023_19-59-46_original"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic_maxEnt.load_state_dict(actor_critic_weighs['state_dict'])


saved_epoch = 1524
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_0_num_env_200_entro_0.01_gama_0.999_24-01-2023_18-15-01_noRNN"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_1569_num_env_200_entro_0.01_gama_0.999_07-03-2023_23-20-30_noRNN_original"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic1.load_state_dict(actor_critic_weighs['state_dict'])

# saved_epoch = 3050
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/heist_seed_58965_num_env_200_entro_0.01_gama_0.999_03-02-2023_00-13-15_noRNN_original"
# if (saved_epoch > 0) and save_dir != "":
#     save_path = save_dir
#     actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
#     actor_critic2.load_state_dict(actor_critic_weighs['state_dict'])
#
# saved_epoch = 3050
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/heist_seed_58965_num_env_200_entro_0.01_gama_0.999_03-02-2023_00-13-15_noRNN_original"
# if (saved_epoch > 0) and save_dir != "":
#     save_path = save_dir
#     actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
#     actor_critic3.load_state_dict(actor_critic_weighs['state_dict'])
#
# saved_epoch = 3050
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/heist_seed_58965_num_env_200_entro_0.01_gama_0.999_03-02-2023_00-13-15_noRNN_original"
# if (saved_epoch > 0) and save_dir != "":
#     save_path = save_dir
#     actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
#     actor_critic4.load_state_dict(actor_critic_weighs['state_dict'])

# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_0_num_env_50_entro_0.01_gama_0.999_15-01-2023_15-54-15_noRNN"
# if (saved_epoch > 0) and save_dir != "":
#     save_path = save_dir
#     actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
#     actor_critic5.load_state_dict(actor_critic_weighs['state_dict'])
#
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_0_num_env_50_entro_0.01_gama_0.999_15-01-2023_15-55-07_noRNN"
# if (saved_epoch > 0) and save_dir != "":
#     save_path = save_dir
#     actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
#     actor_critic5.load_state_dict(actor_critic_weighs['state_dict'])
#
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_0_num_env_50_entro_0.01_gama_0.999_15-01-2023_15-55-35_noRNN"
# if (saved_epoch > 0) and save_dir != "":
#     save_path = save_dir
#     actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
#     actor_critic6.load_state_dict(actor_critic_weighs['state_dict'])
#
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_0_num_env_50_entro_0.01_gama_0.999_15-01-2023_15-56-03_noRNN"
# if (saved_epoch > 0) and save_dir != "":
#     save_path = save_dir
#     actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
#     actor_critic7.load_state_dict(actor_critic_weighs['state_dict'])
#
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_0_num_env_50_entro_0.01_gama_0.999_15-01-2023_15-56-31_noRNN"
# if (saved_epoch > 0) and save_dir != "":
#     save_path = save_dir
#     actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
#     actor_critic8.load_state_dict(actor_critic_weighs['state_dict'])
#
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_0_num_env_50_entro_0.01_gama_0.999_15-01-2023_16-06-03_noRNN"
# if (saved_epoch > 0) and save_dir != "":
#     save_path = save_dir
#     actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
#     actor_critic9.load_state_dict(actor_critic_weighs['state_dict'])
#
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_0_num_env_50_entro_0.01_gama_0.999_15-01-2023_16-06-10_noRNN"
# if (saved_epoch > 0) and save_dir != "":
#     save_path = save_dir
#     actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
#     actor_critic10.load_state_dict(actor_critic_weighs['state_dict'])

# obs = test_env.reset()

eval_recurrent_hidden_states = torch.zeros(
    1, actor_critic_maxEnt.recurrent_hidden_state_size, device=device)
eval_masks = torch.zeros(1, 1, device=device)

eval_attn_masks  = torch.zeros(1, actor_critic_maxEnt.attention_size, device=device)
eval_attn_masks1 = torch.zeros(1, 16, device=device)
eval_attn_masks2 = torch.zeros(1, 32, device=device)
eval_attn_masks3 = torch.zeros(1, 32, device=device)

down_sample_avg = nn.AvgPool2d(1, stride=1)
actor_critic_maxEnt.eval()
# actor_critic1.eval()
# actor_critic2.eval()
# actor_critic3.eval()
# actor_critic4.eval()
rew, obs, first = test_env.observe()
rew_full_obs, obs_full_obs, first_full_obs = test_env_full_obs.observe()
# states = test_env.callmethod("get_state")
# test_env2.callmethod("set_state", states)
# rew2, obs2, first2 = test_env2.observe()



obs = obs['rgb'].transpose(0, 3, 1, 2)
obs2 = obs_full_obs['rgb'].transpose(0, 3, 1, 2)
# obs0 = obs2[0]
# indexes = (obs0[0, :, :] == 63) * (obs0[1, :, :] == 255) * (obs0[2, :, :] == 63)
# obs0[0, :, :]  = (obs0[0, :, :] * (~indexes))
# obs0[1, :, :]  = (obs0[1, :, :] * (~indexes))
# obs0[2, :, :]  = (obs0[2, :, :] * (~indexes))
# obs2 = torch.tensor(obs2)
# myobj = plt.imshow(obs2[0].transpose(0, 2).transpose(0, 1))
# plt.show()

obs = torch.FloatTensor(obs / 255)
obs2 = torch.FloatTensor(obs2 / 255)
myobj = plt.imshow(obs2[0][0])
plt.show()
# obs = torch.FloatTensor(obs.transpose(0, 3, 1, 2) / 255)
obs2 = down_sample_avg(obs2)
obs_sum = obs2[0][0]
Oracle = (obs_sum == 0).sum()
print('Oracle: {}'.format(Oracle))
obs_stack = obs.clone()

# myobj = plt.imshow(obs_sum>0.0001)
# plt.show()

# Change reward location
# obs = obs['rgb']
# indexes = np.stack([(obs[:,:,2] == 255), (obs[:,:,2] == 255), (obs[:,:,2]== 255)], axis=2)
# obs = obs*(1-indexes)
# # obs[50:55,43:47,0] = 127
# # obs[50:55,43:47,1] = 127
# # obs[50:55,43:47,2] = 255
# # obs[43:48,17:21,0] = 127
# # obs[43:48,17:21,1] = 127
# # obs[43:48,17:21,2] = 255
# obs[9:13,9:14,0] = 127
# obs[9:13,9:14,1] = 127
# obs[9:13,9:14,2] = 255

# obs =obs.transpose(0, 3, 1, 2) / 255.0
# obs = np.expand_dims(obs, axis=0).transpose(0, 3, 1, 2) / 255.0
# hidden_state = np.zeros((1, 1))
# hidden_state = torch.FloatTensor(hidden_state).to(device=device)


done = np.zeros(1)
step = 0
iter = 0
beta = 0.5
moving_average_prob1 = 0
moving_average_prob2 = 0
steps_remaining = 30
step_count = 0
max_ent_step = 0
ent_step_count = 0
ent_last_step_count = 0
novel = True
int_reward_sum = 0

# m = FixedCategorical(torch.tensor([ 0.55, 0.25, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.0125, 0.0125, 0.0125, 0.0125]))
m = FixedCategorical(torch.tensor([ 0.55, 0.25, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]))
# m = FixedCategorical(torch.tensor([ 0.55, 0.25, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025]))
# m = m = FixedCategorical(torch.tensor([ 0.55, 0.25, 0.045, 0.025, 0.025, 0.025, 0.015, 0.015, 0.0125, 0.0125, 0.0125, 0.0125]))
rand_policy = FixedCategorical(torch.tensor([ 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 1-14*0.067]))
maxEnt_steps = 0
while not done[0] and iter<1000:
# while iter<1000:
    iter +=1
    with torch.no_grad():
        # obs = torch.FloatTensor(obs).to(device=device)
        # hidden_state = torch.FloatTensor(hidden_state).to(device=device)
        # mask = torch.FloatTensor(1 - done).to(device=device)
        # dist, value, hidden_state = policy(obs, hidden_state, mask)
        # act = dist.sample()
        # log_prob_act = dist.log_prob(act)
        # plt.imshow(u.transpose(1, 2, 0))
        # plt.show()


        _, action1, _, dist_probs1, _, _, _, _, _ = actor_critic1.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=True,
            reuse_masks=False)

        # dist_probs1[:, 1] += dist_probs1[:, 0]
        # dist_probs1[:, 1] += dist_probs1[:, 2]
        # dist_probs1[:, 0] = 0
        # dist_probs1[:, 2] = 0
        #
        # dist_probs1[:, 7] += dist_probs1[:, 6]
        # dist_probs1[:, 7] += dist_probs1[:, 8]
        # dist_probs1[:, 6] = 0
        # dist_probs1[:, 8] = 0
        # pure_action1 = dist_probs1.max(1)[1].unsqueeze(1)
        # prob_pure_action1 = dist_probs1.max(1)[0].unsqueeze(1)
        #
        # if (prob_pure_action1 > 0.9):
        #     moving_average_prob1 = (1-beta)*moving_average_prob1 + beta*prob_pure_action1
        # else:
        #     moving_average_prob1 = 0
        #
        _, action2, _, dist_probs2, _, _, _, _, _ = actor_critic2.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=True,
            reuse_masks=False)

        # dist_probs2[:, 1] += dist_probs2[:, 0]
        # dist_probs2[:, 1] += dist_probs2[:, 2]
        # dist_probs2[:, 0] = 0
        # dist_probs2[:, 2] = 0
        #
        # dist_probs2[:, 7] += dist_probs2[:, 6]
        # dist_probs2[:, 7] += dist_probs2[:, 8]
        # dist_probs2[:, 6] = 0
        # dist_probs2[:, 8] = 0
        # pure_action2 = dist_probs2.max(1)[1].unsqueeze(1)
        # prob_pure_action2 = dist_probs2.max(1)[0].unsqueeze(1)
        #
        # if (prob_pure_action2 > 0.9):
        #     moving_average_prob2 = (1 - beta) * moving_average_prob2 + beta * prob_pure_action2
        # else:
        #     moving_average_prob2 = 0

        _, action3, _, dist_probs3, _, _, _, _, _ = actor_critic3.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=True,
            reuse_masks=False)

        # dist_probs3[:, 1] += dist_probs3[:, 0]
        # dist_probs3[:, 1] += dist_probs3[:, 2]
        # dist_probs3[:, 0] = 0
        # dist_probs3[:, 2] = 0
        #
        # dist_probs3[:, 7] += dist_probs3[:, 6]
        # dist_probs3[:, 7] += dist_probs3[:, 8]
        # dist_probs3[:, 6] = 0
        # dist_probs3[:, 8] = 0
        # pure_action3 = dist_probs3.max(1)[1].unsqueeze(1)
        # prob_pure_action3 = dist_probs3.max(1)[0].unsqueeze(1)

        _, action4, _, dist_probs4, _, _, _, _, _ = actor_critic4.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=True,
            reuse_masks=False)

        # dist_probs4[:, 1] += dist_probs4[:, 0]
        # dist_probs4[:, 1] += dist_probs4[:, 2]
        # dist_probs4[:, 0] = 0
        # dist_probs4[:, 2] = 0
        #
        # dist_probs4[:, 7] += dist_probs4[:, 6]
        # dist_probs4[:, 7] += dist_probs4[:, 8]
        # dist_probs4[:, 6] = 0
        # dist_probs4[:, 8] = 0
        # pure_action4 = dist_probs4.max(1)[1].unsqueeze(1)
        # prob_pure_action4 = dist_probs4.max(1)[0].unsqueeze(1)
        #
        # _, _, _, dist_probs5, _, _, _, _, _ = actor_critic5.act(
        #     obs.float().to(device),
        #     eval_recurrent_hidden_states,
        #     eval_masks,
        #     attn_masks=eval_attn_masks,
        #     attn_masks1=eval_attn_masks1,
        #     attn_masks2=eval_attn_masks2,
        #     attn_masks3=eval_attn_masks3,
        #     deterministic=True,
        #     reuse_masks=False)
        #
        # dist_probs5[:, 1] += dist_probs5[:, 0]
        # dist_probs5[:, 1] += dist_probs5[:, 2]
        # dist_probs5[:, 0] = 0
        # dist_probs5[:, 2] = 0
        #
        # dist_probs5[:, 7] += dist_probs5[:, 6]
        # dist_probs5[:, 7] += dist_probs5[:, 8]
        # dist_probs5[:, 6] = 0
        # dist_probs5[:, 8] = 0
        # pure_action5 = dist_probs5.max(1)[1].unsqueeze(1)
        # prob_pure_action5 = dist_probs5.max(1)[0].unsqueeze(1)
        #
        # _, _, _, dist_probs6, _, _, _, _, _ = actor_critic6.act(
        #     obs.float().to(device),
        #     eval_recurrent_hidden_states,
        #     eval_masks,
        #     attn_masks=eval_attn_masks,
        #     attn_masks1=eval_attn_masks1,
        #     attn_masks2=eval_attn_masks2,
        #     attn_masks3=eval_attn_masks3,
        #     deterministic=True,
        #     reuse_masks=False)
        #
        # dist_probs6[:, 1] += dist_probs6[:, 0]
        # dist_probs6[:, 1] += dist_probs6[:, 2]
        # dist_probs6[:, 0] = 0
        # dist_probs6[:, 2] = 0
        #
        # dist_probs6[:, 7] += dist_probs6[:, 6]
        # dist_probs6[:, 7] += dist_probs6[:, 8]
        # dist_probs6[:, 6] = 0
        # dist_probs6[:, 8] = 0
        # pure_action6 = dist_probs6.max(1)[1].unsqueeze(1)
        # prob_pure_action6 = dist_probs6.max(1)[0].unsqueeze(1)
        #
        # _, _, _, dist_probs7, _, _, _, _, _ = actor_critic7.act(
        #     obs.float().to(device),
        #     eval_recurrent_hidden_states,
        #     eval_masks,
        #     attn_masks=eval_attn_masks,
        #     attn_masks1=eval_attn_masks1,
        #     attn_masks2=eval_attn_masks2,
        #     attn_masks3=eval_attn_masks3,
        #     deterministic=True,
        #     reuse_masks=False)
        #
        # dist_probs7[:, 1] += dist_probs7[:, 0]
        # dist_probs7[:, 1] += dist_probs7[:, 2]
        # dist_probs7[:, 0] = 0
        # dist_probs7[:, 2] = 0
        #
        # dist_probs7[:, 7] += dist_probs7[:, 6]
        # dist_probs7[:, 7] += dist_probs7[:, 8]
        # dist_probs7[:, 6] = 0
        # dist_probs7[:, 8] = 0
        # pure_action7 = dist_probs7.max(1)[1].unsqueeze(1)
        # prob_pure_action7 = dist_probs7.max(1)[0].unsqueeze(1)
        #
        # _, _, _, dist_probs8, _, _, _, _, _ = actor_critic8.act(
        #     obs.float().to(device),
        #     eval_recurrent_hidden_states,
        #     eval_masks,
        #     attn_masks=eval_attn_masks,
        #     attn_masks1=eval_attn_masks1,
        #     attn_masks2=eval_attn_masks2,
        #     attn_masks3=eval_attn_masks3,
        #     deterministic=True,
        #     reuse_masks=False)
        #
        # dist_probs8[:, 1] += dist_probs8[:, 0]
        # dist_probs8[:, 1] += dist_probs8[:, 2]
        # dist_probs8[:, 0] = 0
        # dist_probs8[:, 2] = 0
        #
        # dist_probs8[:, 7] += dist_probs8[:, 6]
        # dist_probs8[:, 7] += dist_probs8[:, 8]
        # dist_probs8[:, 6] = 0
        # dist_probs8[:, 8] = 0
        # pure_action8 = dist_probs8.max(1)[1].unsqueeze(1)
        # prob_pure_action8 = dist_probs8.max(1)[0].unsqueeze(1)
        #
        # _, _, _, dist_probs9, _, _, _, _, _ = actor_critic9.act(
        #     obs.float().to(device),
        #     eval_recurrent_hidden_states,
        #     eval_masks,
        #     attn_masks=eval_attn_masks,
        #     attn_masks1=eval_attn_masks1,
        #     attn_masks2=eval_attn_masks2,
        #     attn_masks3=eval_attn_masks3,
        #     deterministic=True,
        #     reuse_masks=False)
        #
        # dist_probs9[:, 1] += dist_probs9[:, 0]
        # dist_probs9[:, 1] += dist_probs9[:, 2]
        # dist_probs9[:, 0] = 0
        # dist_probs9[:, 2] = 0
        #
        # dist_probs9[:, 7] += dist_probs9[:, 6]
        # dist_probs9[:, 7] += dist_probs9[:, 8]
        # dist_probs9[:, 6] = 0
        # dist_probs9[:, 8] = 0
        # pure_action9 = dist_probs9.max(1)[1].unsqueeze(1)
        # prob_pure_action9 = dist_probs9.max(1)[0].unsqueeze(1)
        #
        # _, _, _, dist_probs10, _, _, _, _, _ = actor_critic10.act(
        #     obs.float().to(device),
        #     eval_recurrent_hidden_states,
        #     eval_masks,
        #     attn_masks=eval_attn_masks,
        #     attn_masks1=eval_attn_masks1,
        #     attn_masks2=eval_attn_masks2,
        #     attn_masks3=eval_attn_masks3,
        #     deterministic=True,
        #     reuse_masks=False)
        #
        # dist_probs10[:, 1] += dist_probs10[:, 0]
        # dist_probs10[:, 1] += dist_probs10[:, 2]
        # dist_probs10[:, 0] = 0
        # dist_probs10[:, 2] = 0
        #
        # dist_probs10[:, 7] += dist_probs10[:, 6]
        # dist_probs10[:, 7] += dist_probs10[:, 8]
        # dist_probs10[:, 6] = 0
        # dist_probs10[:, 8] = 0
        # pure_action10 = dist_probs10.max(1)[1].unsqueeze(1)
        # prob_pure_action10 = dist_probs10.max(1)[0].unsqueeze(1)

        _, action_maxEnt, _, dist_probs_maxEnt, eval_recurrent_hidden_states, _, _, _, _ = actor_critic_maxEnt.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=True,
            reuse_masks=False)

        # dist_probs_maxEnt[:, 1] += dist_probs_maxEnt[:, 0]
        # dist_probs_maxEnt[:, 1] += dist_probs_maxEnt[:, 2]
        # dist_probs_maxEnt[:, 0] = 0
        # dist_probs_maxEnt[:, 2] = 0
        #
        # dist_probs_maxEnt[:, 7] += dist_probs_maxEnt[:, 6]
        # dist_probs_maxEnt[:, 7] += dist_probs_maxEnt[:, 8]
        # dist_probs_maxEnt[:, 6] = 0
        # dist_probs_maxEnt[:, 8] = 0
        # pure_action_maxEnt = dist_probs_maxEnt.max(1)[1].unsqueeze(1)
        # prob_pure_action_maxEnt = dist_probs_maxEnt.max(1)[0].unsqueeze(1)

    # next_obs, rew, done, info = test_env.step(act.cpu().numpy())
    # action = pure_action4
    action = action_maxEnt
    # action = rand_policy.sample().unsqueeze(1)
    # # if (pure_action1 != pure_action2) or (moving_average_prob1 < 0.98) or (moving_average_prob2 < 0.98):
    # # if (moving_average_prob1 > 0.9):
    # ent_step_count += 1
    maxEnt_steps -= 1

    if (maxEnt_steps<=0) and action1 == action2 == action3 == action4:
    # if novel and pure_action1 == pure_action2 == pure_action3 == pure_action4:
    # if novel and pure_action1 == pure_action2 == pure_action3 == pure_action4 == pure_action5 == pure_action6 == pure_action7 == pure_action8 == pure_action9 == pure_action10:
    # if novel:
        # if step_count > 5:
        action = action1
        maxEnt_steps = m.sample() + 1
    #         max_ent_step +=1
    #         step_count  = max_ent_step
    # #     step_count += 1
    # else:
    #     ent_last_step_count = ent_step_count
    #     ent_step_count = 0


    # if iter < 100:
    #     action = pure_action1
    test_env.act(action_maxEnt[0].cpu().numpy())
    test_env_full_obs.act(action_maxEnt[0].cpu().numpy())
    # steps_remaining -= 1



    eval_masks = torch.tensor(
        [[0.0] if done_ else [1.0] for done_ in done],
        dtype=torch.float32,
        device=device)

    rew, obs, first = test_env.observe()
    done[0] = first
    step += 1

    # states = test_env.callmethod("get_state")
    # test_env2.callmethod("set_state", states)
    rew2, obs2, first2 = test_env_full_obs.observe()

    obs = obs['rgb'].transpose(0, 3, 1, 2)
    obs2 = obs2['rgb'].transpose(0, 3, 1, 2)

    # obs0 = obs2[0]
    # indexes = (obs0[0, :, :] == 63) * (obs0[1, :, :] == 255) * (obs0[2, :, :] == 63)
    # obs0[0, :, :] = (obs0[0, :, :] * (~indexes))
    # obs0[1, :, :] = (obs0[1, :, :] * (~indexes))
    # obs0[2, :, :] = (obs0[2, :, :] * (~indexes))
    # obs2 = torch.tensor(obs2)
    # myobj = plt.imshow(obs2[0].transpose(0, 2).transpose(0, 1))
    # plt.show()

    obs = torch.FloatTensor(obs / 255)
    obs2 = torch.FloatTensor(obs2 / 255)
    obs2 = down_sample_avg(obs2)


    # norm2_dis = (obs_stack - obs).reshape(obs_stack.size(0),-1).pow(2).sum(1)
    # int_reward = 1 * (norm2_dis.min() > 100)
    # print('int reward: {}'.format(int_reward))
    # obs_stack = torch.cat((obs_stack, obs.clone()), 0)
    # int_reward_sum += int_reward
    # if iter == 256:
    #     obs_stack = obs.clone()



    next_obs_sum = obs_sum + obs2[0][0]
    num_zero_obs_sum = (obs_sum == 0).sum()
    num_zero_next_obs_sum = (next_obs_sum == 0).sum()
    if num_zero_obs_sum - num_zero_next_obs_sum > 0:
        novel = True
        obs_sum = obs_sum + obs2[0][0]
    else:
        novel = False
    int_reward_sum +=  num_zero_obs_sum - num_zero_next_obs_sum

    # myobj = plt.imshow(obs_sum>0.0001)
    # plt.show()




    # print(f"step {step} reward {rew} first {first}")
    print(f"step {step} reward {rew} first {first} action {action} if action1 {(action4 == action1 == action2 == action3)} novel {novel} int_reward_sum {int_reward_sum}")
    print("debug")
    # print(f" prob1 {prob_pure_action1} prob2 {prob_pure_action2} prob3 {prob_pure_action3} prob4 {prob_pure_action4}")

    # # Change reward location
    # obs = obs['rgb']
    # indexes = np.stack([(obs[:, :, 2] == 255), (obs[:, :, 2] == 255), (obs[:, :, 2] == 255)], axis=2)
    # obs = obs * (1 - indexes)
    # # obs[50:55, 43:47, 0] = 127
    # # obs[50:55, 43:47, 1] = 127
    # # obs[50:55, 43:47, 2] = 255
    # # obs[43:48, 17:21, 0] = 127
    # # obs[43:48, 17:21, 1] = 127
    # # obs[43:48, 17:21, 2] = 255
    # obs[9:13, 9:14, 0] = 127
    # obs[9:13, 9:14, 1] = 127
    # obs[9:13, 9:14, 2] = 255

    # obs = np.expand_dims(obs, axis=0).transpose(0, 3, 1, 2) / 255.0
    # obs = obs.transpose(0, 3, 1, 2)

print("done")