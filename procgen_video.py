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
env_name = "maze"
start_level = 0
num_level = 1
distribution_mode = "easy"
seed = 0
normalize_rew = False
no_normalize = True
# n_steps = 1

device = torch.device("cuda:{}".format(0))

class VideoRecorderprocess(VideoRecorderWrapper):
    def __init__(self, env, directory, ob_key, prefix, fps):
        super().__init__(env=env, directory=directory, ob_key=ob_key, prefix=prefix, fps=fps)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        obs = frame
        # indexes = np.stack([(obs[:, :, 2] == 255), (obs[:, :, 2] == 255), (obs[:, :, 2] == 255)], axis=2)
        # obs = (obs * (1 - indexes)).astype(np.uint8)
        # # obs[50:55,43:47,0] = 127
        # # obs[50:55,43:47,1] = 127
        # # obs[50:55,43:47,2] = 255
        # # obs[43:48,17:21,0] = 127
        # # obs[43:48,17:21,1] = 127
        # # obs[43:48,17:21,2] = 255
        # obs[9:13, 9:14, 0] = 127
        # obs[9:13, 9:14, 1] = 127
        # obs[9:13, 9:14, 2] = 255
        obs = obs.astype(np.uint8)

        return obs

test_start_level = 4965
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

test_env = VideoRecorderWrapper(env=test_env, directory="./videos", info_key="rgb", prefix=str(test_start_level), fps=5, render=True)

actor_critic = Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': True ,'hidden_size': 256})
actor_critic.to(device)

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
saved_epoch = 9372
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_0_num_env_1000_entro_0.01_15-11-2022_09-52-09"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(
        os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic.load_state_dict(actor_critic_weighs['state_dict'])

# obs = test_env.reset()
eval_recurrent_hidden_states = torch.zeros(
    num_level, actor_critic.recurrent_hidden_state_size, device=device)
eval_masks = torch.zeros(num_level, 1, device=device)

eval_attn_masks  = torch.zeros(num_level, actor_critic.attention_size, device=device)
eval_attn_masks1 = torch.zeros(num_level, 16, device=device)
eval_attn_masks2 = torch.zeros(num_level, 32, device=device)
eval_attn_masks3 = torch.zeros(num_level, 32, device=device)

actor_critic.eval()
rew, obs, first = test_env.observe()
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
# myobj = plt.imshow(obs[0].transpose(1, 2, 0))
# plt.show()
iter = 0
while not done[0] and iter<500:
    iter +=1
    with torch.no_grad():
        # obs = torch.FloatTensor(obs).to(device=device)
        # hidden_state = torch.FloatTensor(hidden_state).to(device=device)
        # mask = torch.FloatTensor(1 - done).to(device=device)
        # dist, value, hidden_state = policy(obs, hidden_state, mask)
        # act = dist.sample()
        # log_prob_act = dist.log_prob(act)
        # plt.imshow(obs[0].transpose(1, 2, 0))
        # plt.show()

        # myobj.set_data(obs[0].transpose(1, 2, 0))
        # plt.show()
        obs = obs['rgb']
        obs = torch.FloatTensor(obs.transpose(0, 3, 1, 2)/255)
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

    # next_obs, rew, done, info = test_env.step(act.cpu().numpy())
    test_env.act(action[0].cpu().numpy())

    eval_masks = torch.tensor(
        [[0.0] if done_ else [1.0] for done_ in done],
        dtype=torch.float32,
        device=device)

    rew, obs, first = test_env.observe()
    done[0] = first
    step += 1
    print(f"step {step} reward {rew} first {first} action {action[0]}")

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