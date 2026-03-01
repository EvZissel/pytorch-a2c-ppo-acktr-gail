from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.vec_env import VecVideoRecorder
import cv2
import os

from a2c_ppo_acktr.procgen_wrappers import *
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


env_name = "maze"
distribution_mode = "easy"
num_level = 1
seed = 0
normalize_rew = False
no_normalize = True


device = torch.device("cuda:{}".format(0))
class VideoRecorderprocess(VideoRecorderWrapper):
    def __init__(self, env, directory, info_key, prefix, fps ,render):
        super().__init__(env=env, directory=directory, info_key=info_key, prefix=prefix, fps=fps, render=render)


    def _process_frame(self, frame: np.ndarray) -> np.ndarray: # overwrite the function to add text to the frame
        # obs = frame
        # mouse_crnter = (obs == [100,122,123])
        # x_start = np.where(mouse_crnter.sum(2) == 3)[1][0]
        # y_start = np.where(mouse_crnter.sum(2) == 3)[0][0]
        # x_end = x_start
        # y_end = y_start
        # if self.action == 0 or self.action == 1 or self.action == 2:
        #     x_start = x_start - 8
        #     x_end   = x_start - 20
        # elif self.action == 3:
        #     y_start = y_start + 8
        #     y_end   = y_start + 20
        # elif self.action == 5:
        #     y_start = y_start - 8
        #     y_end   = y_start - 20
        # elif self.action == 6 or self.action == 7 or self.action == 8:
        #     x_start = x_start + 8
        #     x_end   = x_start + 20


        # image = Image.fromarray(obs)
        # draw = ImageDraw.Draw(image)
        #
        # font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
        # font = ImageFont.truetype(font_path, size=20)
        #
        # text = "ExpGen: "
        # if self.is_maxEnt:
        #     text += "MaxEnt"
        #     fill = "red"
        #     color = (255, 0, 0)
        # else:
        #     text += "Reward Policy"
        #     fill = "blue"
        #     color = (0, 0, 255)
        # draw.text((186, 6), text, fill=fill, align ="left", font=font)
        #
        # na = np.array(image)
        # na = cv2.arrowedLine(na, (x_start, y_start), (x_end, y_end), color, 2)
        #
        # # obs = obs.astype(np.uint8)
        # # myobj = plt.imshow(obs)
        # # plt.show()
        #
        # # na = frame
        return frame


test_start_level = 218
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

test_env = VideoRecorderprocess(env=test_env, directory="./videos", info_key="rgb", prefix=str(test_start_level), fps=1, render=True)


actor_critic = Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': False ,'hidden_size': 256})
actor_critic.to(device)


# Load previous model
saved_epoch = 1524
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_63957_num_env_200_entro_0.01_gama_0.999_01-02-2023_13-55-44_noRNN_original" #last one
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic.load_state_dict(actor_critic_weighs['state_dict'])

eval_recurrent_hidden_states = torch.zeros(
    1, 256, device=device)
eval_masks = torch.zeros(1, 1, device=device)

eval_attn_masks  = torch.zeros(1, actor_critic.attention_size, device=device)
down_sample_avg = nn.AvgPool2d(1, stride=1)

rew, obs, first = test_env.observe()


obs = obs['rgb'].transpose(0, 3, 1, 2)

obs = torch.FloatTensor(obs / 255)


done = np.zeros(1)
step = 0
iter = 0
step_count = 0
rew_sum = 0

while not done[0] and iter<1000:
    iter +=1
    with torch.no_grad():

        value1, action1, _, dist_probs1, _, _, _, _, _ = actor_critic.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            deterministic=False,
            reuse_masks=False)

        test_env.act(action1[0].cpu().numpy())

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        rew, next_obs, first = test_env.observe()
        rew_sum += rew
        done[0] = first
        step += 1

        next_obs = next_obs['rgb']
        next_obs = torch.FloatTensor(next_obs.transpose(0, 3, 1, 2) / 255)


    if first:
        print(f"reward {rew}")


