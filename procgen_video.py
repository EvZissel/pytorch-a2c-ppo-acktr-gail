from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.vec_env import VecVideoRecorder
import cv2
import os

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
from PIL import Image, ImageFont, ImageDraw

EVAL_ENVS = ['train_eval','test_eval']

# num_processes = 600
env_name = "maze"
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

        self.is_maxEnt = True
        self.action = None

    def set_is_maxEnt(self, is_maxEnt, action):
        self.is_maxEnt = is_maxEnt
        self.action = action


    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        obs = frame
        mouse_crnter = (obs == [100,122,123])
        x_start = np.where(mouse_crnter.sum(2) == 3)[1][0]
        y_start = np.where(mouse_crnter.sum(2) == 3)[0][0]
        x_end = x_start
        y_end = y_start
        if self.action == 0 or self.action == 1 or self.action == 2:
            x_start = x_start - 8
            x_end   = x_start - 20
        elif self.action == 3:
            y_start = y_start + 8
            y_end   = y_start + 20
        elif self.action == 5:
            y_start = y_start - 8
            y_end   = y_start - 20
        elif self.action == 6 or self.action == 7 or self.action == 8:
            x_start = x_start + 8
            x_end   = x_start + 20


        image = Image.fromarray(obs)
        draw = ImageDraw.Draw(image)

        font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
        font = ImageFont.truetype(font_path, size=20)

        text = "ExpGen: "
        if self.is_maxEnt:
            text += "MaxEnt"
            fill = "red"
            color = (255, 0, 0)
        else:
            text += "Reward Policy"
            fill = "blue"
            color = (0, 0, 255)
        draw.text((186, 6), text, fill=fill, align ="left", font=font)

        na = np.array(image)
        # na = cv2.arrowedLine(na, (x_start, y_start), (x_end, y_end), color, 2)

        # obs = obs.astype(np.uint8)
        # myobj = plt.imshow(obs)
        # plt.show()

        return na

# test_start_level = 21i maze
test_start_level = 204
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
#                              mask_all=False=4

test_env = ProcgenGym3Env(num=1,
                          env_name=env_name,
                          start_level=test_start_level,
                          num_levels=num_level,
                          distribution_mode=distribution_mode,
                          render_mode="rgb_array",
                          use_generated_assets=False,
                          center_agent=False,
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
                          use_generated_assets=False,
                          center_agent=False,
                          use_backgrounds=True,
                          restrict_themes=False,
                          use_monochrome_assets=False,
                          rand_seed=seed)

test_env = VideoRecorderprocess(env=test_env, directory="./videos", info_key="rgb", prefix=str(test_start_level), fps=1, render=True)

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

actor_critic5= Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': False ,'hidden_size': 256})
actor_critic5.to(device)

actor_critic6= Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': False ,'hidden_size': 256})
actor_critic6.to(device)

actor_critic7= Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': False ,'hidden_size': 256})
actor_critic7.to(device)

actor_critic8= Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': False ,'hidden_size': 256})
actor_critic8.to(device)

actor_critic9= Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': False ,'hidden_size': 256})
actor_critic9.to(device)

actor_critic10= Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': False ,'hidden_size': 256})
actor_critic10.to(device)

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
#
# # # save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/heist_seed_58967_num_env_200_entro_0.01_gama_0.5_05-02-2023_22-24-49_original"
# # save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/heist_seed_58967_num_env_200_entro_0.01_gama_0.5_05-02-2023_18-15-27_original"
# # save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_0_num_env_200_entro_0.01_gama_0.5_25-12-2022_00-56-16"
# # save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_2865_num_env_200_entro_0.01_gama_0.5_07-02-2023_15-10-02_original"
# # save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_2865_num_env_200_entro_0.01_gama_0.5_07-02-2023_19-36-39_original"
# # save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_2865_num_env_200_entro_0.01_gama_0.5_07-02-2023_19-40-00_original"
# # save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_58967_num_env_200_entro_0.01_gama_0.5_04-02-2023_15-46-16_original"
# # save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_3569_num_env_200_entro_0.01_gama_0.5_08-02-2023_23-48-07_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_7945_num_env_200_entro_0.01_gama_0.99_06-04-2023_16-01-01" #hard
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_1234_num_env_200_entro_0.01_gama_0.5_11-04-2023_12-21-22"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_1234_num_env_200_entro_0.01_gama_0.5_24-04-2023_13-36-20_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/heist_seed_1234_num_env_200_entro_0.01_gama_0.5_24-04-2023_13-40-28_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_4567_num_env_200_entro_0.01_gama_0.99_28-04-2023_13-38-59_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_1234_num_env_200_entro_0.01_gama_0.5_02-05-2023_17-29-21_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_1234_num_env_200_entro_0.01_gama_0.5_02-05-2023_17-43-16_original" #last one
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/dodgeball_seed_1234_num_env_200_entro_0.01_gama_0.9_02-05-2023_09-43-13_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/bigfish_seed_1234_num_env_200_entro_0.01_gama_0.99_02-05-2023_10-29-48_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/miner_seed_1234_num_env_200_entro_0.01_gama_0.9_01-05-2023_12-56-50_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/starpilot_seed_1234_num_env_200_entro_0.01_gama_0.9_20-05-2023_19-55-30_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/heist_seed_1375_num_env_200_entro_0.01_gama_0.999_03-02-2023_00-18-51_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_1234_num_env_200_entro_0.01_gama_0.5_18-08-2023_14-14-32_original"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_1234_num_env_200_entro_0.01_gama_0.9_16-09-2023_17-36-37_original_L2eval"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic_maxEnt.load_state_dict(actor_critic_weighs['state_dict'])


saved_epoch = 1524
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_0_num_env_200_entro_0.01_gama_0.999_24-01-2023_18-15-01_noRNN"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_23456_num_env_200_entro_0.01_gama_0.999_10-04-2023_18-12-26_noRNN" #hard
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_1547_num_env_200_entro_0.01_gama_0.999_10-04-2023_15-29-37_noRNN"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_63957_num_env_200_entro_0.01_gama_0.999_01-02-2023_13-55-44_noRNN_original" #last one
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_9465_num_env_200_entro_0.01_gama_0.999_07-03-2023_23-40-30_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/bigfish_seed_1258_num_env_200_entro_0.01_gama_0.999_02-05-2023_10-39-29_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/starpilot_seed_1_num_env_200_entro_0.01_gama_0.999_19-05-2023_00-35-58_noRNN_original"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic1.load_state_dict(actor_critic_weighs['state_dict'])

saved_epoch = 1524
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_25478_num_env_200_entro_0.01_gama_0.999_10-04-2023_15-30-04_noRNN"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_25698_num_env_200_entro_0.01_gama_0.999_01-02-2023_14-02-43_noRNN_original" #last one
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_8312_num_env_200_entro_0.01_gama_0.999_07-03-2023_23-39-20_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/miner_seed_111555_num_env_200_entro_0.01_gama_0.999_30-04-2023_23-57-06_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/starpilot_seed_1234_num_env_200_entro_0.01_gama_0.999_19-05-2023_13-15-17_noRNN_original"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic2.load_state_dict(actor_critic_weighs['state_dict'])

saved_epoch = 1524
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_44678_num_env_200_entro_0.01_gama_0.999_10-04-2023_15-30-49_noRNN"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_55664_num_env_200_entro_0.01_gama_0.999_27-04-2023_12-10-47_noRNN_original" #last one
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_7985_num_env_200_entro_0.01_gama_0.999_07-03-2023_23-38-04_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/miner_seed_2457_num_env_200_entro_0.01_gama_0.999_30-04-2023_23-58-09_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/starpilot_seed_74185_num_env_200_entro_0.01_gama_0.999_19-05-2023_13-20-03_noRNN_original"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic3.load_state_dict(actor_critic_weighs['state_dict'])

saved_epoch = 1524
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_9745_num_env_200_entro_0.01_gama_0.999_10-04-2023_15-31-16_noRNN"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_89456_num_env_200_entro_0.01_gama_0.999_27-04-2023_12-10-19_noRNN_original" #last one
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_6487_num_env_200_entro_0.01_gama_0.999_07-03-2023_23-25-59_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/miner_seed_33441_num_env_200_entro_0.01_gama_0.999_01-05-2023_00-01-11_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/starpilot_seed_97415_num_env_200_entro_0.01_gama_0.999_19-05-2023_13-23-13_noRNN_original"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic4.load_state_dict(actor_critic_weighs['state_dict'])

saved_epoch = 1524
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_5896_num_env_200_entro_0.01_gama_0.999_11-04-2023_12-22-03_noRNN"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_1296_num_env_200_entro_0.01_gama_0.999_05-02-2023_17-47-44_noRNN_original" #last one
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_5194_num_env_200_entro_0.01_gama_0.999_07-03-2023_23-37-24_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/miner_seed_441112_num_env_200_entro_0.01_gama_0.999_01-05-2023_00-04-01_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/starpilot_seed_2845_num_env_200_entro_0.01_gama_0.999_19-05-2023_13-15-55_noRNN_original"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic5.load_state_dict(actor_critic_weighs['state_dict'])

# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_6334_num_env_200_entro_0.01_gama_0.999_11-04-2023_12-22-26_noRNN"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_26987_num_env_200_entro_0.01_gama_0.999_05-02-2023_17-48-04_noRNN_original" #last one
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_4589_num_env_200_entro_0.01_gama_0.999_07-03-2023_23-25-12_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/miner_seed_5454_num_env_200_entro_0.01_gama_0.999_01-05-2023_00-07-54_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/starpilot_seed_3865_num_env_200_entro_0.01_gama_0.999_19-05-2023_13-16-34_noRNN_original"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic6.load_state_dict(actor_critic_weighs['state_dict'])

# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_8877_num_env_200_entro_0.01_gama_0.999_20-04-2023_22-31-25_noRNN"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_3875_num_env_200_entro_0.01_gama_0.999_05-02-2023_17-48-25_noRNN_original" #last one
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_3978_num_env_200_entro_0.01_gama_0.999_07-03-2023_23-24-08_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/miner_seed_654321_num_env_200_entro_0.01_gama_0.999_01-05-2023_00-09-11_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/starpilot_seed_4752_num_env_200_entro_0.01_gama_0.999_19-05-2023_13-17-25_noRNN_original"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic7.load_state_dict(actor_critic_weighs['state_dict'])

# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_97645_num_env_200_entro_0.01_gama_0.999_20-04-2023_22-32-27_noRNN"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_4976_num_env_200_entro_0.01_gama_0.999_05-02-2023_17-48-46_noRNN_original" #last one
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/jumper_seed_1569_num_env_200_entro_0.01_gama_0.999_07-03-2023_23-20-30_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/miner_seed_771100_num_env_200_entro_0.01_gama_0.999_01-05-2023_00-10-11_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/starpilot_seed_5746_num_env_200_entro_0.01_gama_0.999_19-05-2023_13-18-55_noRNN_original"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic8.load_state_dict(actor_critic_weighs['state_dict'])

# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_110110_num_env_200_entro_0.01_gama_0.999_20-04-2023_22-32-54_noRNN"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_1_num_env_200_entro_0.01_gama_0.999_07-05-2023_14-02-06_noRNN_original" #last one
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/miner_seed_1_num_env_200_entro_0.01_gama_0.999_07-05-2023_14-05-16_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/starpilot_seed_6974_num_env_200_entro_0.01_gama_0.999_19-05-2023_13-19-25_noRNN_original"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic9.load_state_dict(actor_critic_weighs['state_dict'])

# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_27854_num_env_200_entro_0.01_gama_0.999_20-04-2023_22-33-48_noRNN"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_2_num_env_200_entro_0.01_gama_0.999_07-05-2023_14-02-23_noRNN_original" #last one
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/miner_seed_2_num_env_200_entro_0.01_gama_0.999_07-05-2023_14-05-33_noRNN_original"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/starpilot_seed_8975_num_env_200_entro_0.01_gama_0.999_19-05-2023_13-20-42_noRNN_original"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic10.load_state_dict(actor_critic_weighs['state_dict'])


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
# myobj = plt.imshow(obs2[0][0])
# plt.show()
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

# m = FixedCategorical(torch.tensor([ 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.087]))
# m = FixedCategorical(torch.tensor([ 0.55, 0.25, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.0125, 0.0125, 0.0125, 0.0125]))
m = FixedCategorical(torch.tensor([ 0.55, 0.25, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025])) #worked
# m = FixedCategorical(torch.tensor([ 0.55, 0.25, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025]))
# m = m = FixedCategorical(torch.tensor([ 0.55, 0.25, 0.045, 0.025, 0.025, 0.025, 0.015, 0.015, 0.0125, 0.0125, 0.0125, 0.0125]))
rand_policy = FixedCategorical(torch.tensor([ 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 1-14*0.067]))
maxEnt_steps = 0
history = []
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

        value1, action1, _, dist_probs1, _, _, _, _, _ = actor_critic1.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=False,
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
        value2, action2, _, dist_probs2, _, _, _, _, _ = actor_critic2.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=False,
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

        # if (prob_pure_action2 > 0.9):
        #     moving_average_prob2 = (1 - beta) * moving_average_prob2 + beta * prob_pure_action2
        # else:
        #     moving_average_prob2 = 0

        value3, action3, _, dist_probs3, _, _, _, _, _ = actor_critic3.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=False,
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

        value4, action4, _, dist_probs4, _, _, _, _, _ = actor_critic4.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=False,
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

        _, action5, _, dist_probs5, _, _, _, _, _ = actor_critic5.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=False,
            reuse_masks=False)

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

        _, action6, _, dist_probs6, _, _, _, _, _ = actor_critic6.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=False,
            reuse_masks=False)

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
        _, action7, _, dist_probs7, _, _, _, _, _ = actor_critic7.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=False,
            reuse_masks=False)
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
        _, action8, _, dist_probs8, _, _, _, _, _ = actor_critic8.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=False,
            reuse_masks=False)
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
        _, action9, _, dist_probs9, _, _, _, _, _ = actor_critic9.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=False,
            reuse_masks=False)
        #
        # # dist_probs9[:, 1] += dist_probs9[:, 0]
        # # dist_probs9[:, 1] += dist_probs9[:, 2]
        # # dist_probs9[:, 0] = 0
        # # dist_probs9[:, 2] = 0
        # #
        # # dist_probs9[:, 7] += dist_probs9[:, 6]
        # # dist_probs9[:, 7] += dist_probs9[:, 8]
        # # dist_probs9[:, 6] = 0
        # # dist_probs9[:, 8] = 0
        # # pure_action9 = dist_probs9.max(1)[1].unsqueeze(1)
        # # prob_pure_action9 = dist_probs9.max(1)[0].unsqueeze(1)

        _, action10, _, dist_probs10, _, _, _, _, _ = actor_critic10.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=False,
            reuse_masks=False)
        # #
        # # dist_probs10[:, 1] += dist_probs10[:, 0]
        # # dist_probs10[:, 1] += dist_probs10[:, 2]
        # # dist_probs10[:, 0] = 0
        # # dist_probs10[:, 2] = 0
        # #
        # # dist_probs10[:, 7] += dist_probs10[:, 6]
        # # dist_probs10[:, 7] += dist_probs10[:, 8]
        # # dist_probs10[:, 6] = 0
        # # dist_probs10[:, 8] = 0
        # # pure_action10 = dist_probs10.max(1)[1].unsqueeze(1)
        # # prob_pure_action10 = dist_probs10.max(1)[0].unsqueeze(1)
        #
        # _, action10, _, dist_probs10, _, _, _, _, _ = actor_critic10.act(
        #     obs.float().to(device),
        #     eval_recurrent_hidden_states,
        #     eval_masks,
        #     attn_masks=eval_attn_masks,
        #     attn_masks1=eval_attn_masks1,
        #     attn_masks2=eval_attn_masks2,
        #     attn_masks3=eval_attn_masks3,
        #     deterministic=True,
        #     reuse_masks=False)

        value_maxEnt, action_maxEnt, _, dist_probs_maxEnt, eval_recurrent_hidden_states, _, _, _, _ = actor_critic_maxEnt.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=False,
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

    action_hist = torch.zeros(15)
    action_hist[action1] += 1
    action_hist[action2] += 1
    action_hist[action3] += 1
    action_hist[action4] += 1
    action_hist[action5] += 1
    action_hist[action6] += 1
    action_hist[action7] += 1
    action_hist[action8] += 1
    action_hist[action9] += 1
    action_hist[action10] += 1

    # 1-NN
    cardinal_left = action_hist[0] + action_hist[1] + action_hist[2]
    cardinal_right = action_hist[6] + action_hist[7] + action_hist[8]
    # cardinal_up = action_hist[2] + action_hist[5] + action_hist[8]
    cardinal_up =  action_hist[5]
    # cardinal_down = action_hist[0] + action_hist[3] + action_hist[6]
    cardinal_down =  action_hist[3]
    cardinal_fire =  action_hist[9]

    directions = torch.tensor([cardinal_up, cardinal_right, cardinal_down, cardinal_left, cardinal_fire])
    cardinal_index = torch.argmax(directions)
    # cardinal_value = directions[cardinal_index]
    # lookup = torch.tensor([5, 7, 3, 1, 9])
    # action_NN = torch.tensor(lookup[cardinal_index]).unsqueeze(0).unsqueeze(0)

    if action1[0] == 0 or action1[0] == 1 or action1[0] == 2:
        action_cardinal = torch.tensor([3])
    elif  action1[0] == 6 or action1[0] == 7 or action1[0] == 8:
        action_cardinal = torch.tensor([1])
    elif action1[0] == 5:
        action_cardinal = torch.tensor([0])
    elif action1[0] == 3:
        action_cardinal = torch.tensor([2])
    else:
        action_cardinal = torch.tensor([4])

    cardinal_value = directions[action_cardinal]
    # cardinal_index = torch.argmax(action_hist)
    # cardinal_value = action_hist[cardinal_index]
    # action_NN = torch.tensor(cardinal_index).unsqueeze(0).unsqueeze(0)

    # next_obs, rew, done, info = test_env.step(act.cpu().numpy())
    # action = pure_action4
    action = action_maxEnt
    # action = rand_policy.sample().unsqueeze(1)
    # # if (pure_action1 != pure_action2) or (moving_average_prob1 < 0.98) or (moving_average_prob2 < 0.98):
    # # if (moving_average_prob1 > 0.9):
    # ent_step_count += 1
    maxEnt_steps -= 1
    is_maxEnt = True

    # if (maxEnt_steps <= 0) and pure_action1 == pure_action2 == pure_action3 == pure_action4:
    # if (maxEnt_steps <= 0) and torch.var(torch.cat((value1,value2,value3,value4))) < 0.02:
    if (maxEnt_steps <= 0) and cardinal_value >= 8:
    # if novel and pure_action1 == pure_action2 == pure_action3 == pure_action4:
    # if novel and pure_action1 == pure_action2 == pure_action3 == pure_action4 == pure_action5 == pure_action6 == pure_action7 == pure_action8 == pure_action9 == pure_action10:
    # if novel:
        # if step_count > 5:
        # action = action_NN
        # action = action_NN
        action = action1
        is_maxEnt = False
    else:
        maxEnt_steps = m.sample()
    #         max_ent_step +=1
    #         step_count  = max_ent_step
    # #     step_count += 1
    # else:
    #     ent_last_step_count = ent_step_count
    #     ent_step_count = 0


    # if iter < 100:
    #     action = pure_action1
    # test_env.set_is_maxEnt(is_maxEnt, action[0])
    test_env.set_is_maxEnt(True, action[0])
    test_env.act(action_maxEnt[0].cpu().numpy())
    test_env_full_obs.act(action_maxEnt[0].cpu().numpy())

    history.append({'action': action[0].cpu().numpy(), 'is_maxEnt': is_maxEnt})
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
    print(f"t={step} is_maxEnt={is_maxEnt} action={action.item()} cardinal_value={cardinal_value} maxEnt_steps={maxEnt_steps} int_reward_sum={int_reward_sum}")
    print(f"a1={action1.item()}, a2={action2.item()}, a3={action3.item()}, a4={action4.item()}, a5={action5.item()}, a6={action6.item()}, a7={action7.item()}, a8={action8.item()}, maxEnt={action_maxEnt.item()}")
    # print(f"v1={value1.item()}, v2={value2.item()}, v3={value3.item()}, v4={value4.item()}, v_maxEnt={value_maxEnt.item()}")
    # print(f"variance={torch.var(torch.cat((value1,value2,value3,value4)))}")
    if first:
        print(f"reward {rew}")
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

if not first:
    print("** failed to reach goal **")

print("done")

#
# env_replay = ProcgenGym3Env(num=1,
#                           env_name=env_name,
#                           start_level=test_start_level,
#                           num_levels=num_level,
#                           distribution_mode=distribution_mode,
#                           render_mode="rgb_array",
#                           use_generated_assets=False,
#                           center_agent=False,
#                           use_backgrounds=True,
#                           restrict_themes=False,
#                           use_monochrome_assets=False,
#                           rand_seed=seed)
#
# env_recorder = VideoRecorderprocess(env=env_replay, directory="./videos", info_key="rgb", prefix=str(test_start_level), fps=1.5, render=True)
# for i in range(len(history) - 1):
#     action = history[i]['action']
#     transition = history[i + 1]
#     env_recorder.set_is_maxEnt(transition['is_maxEnt'], transition['action'])
#     env_recorder.act(action)
#     env_recorder.observe()