from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.vec_env import VecVideoRecorder

from a2c_ppo_acktr.procgen_wrappers import *
from a2c_ppo_acktr.envs import make_ProcgenEnvs
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
# n_steps = 1

device = torch.device("cuda:{}".format(0))
# device = 'CPU'

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
        # obs = obs.astype(np.uint8)

        return obs

test_start_level = 14
envs = make_ProcgenEnvs(num_envs=1,
                        env_name=env_name,
                        start_level=test_start_level,
                        num_levels=num_level,
                        distribution_mode=distribution_mode,
                        use_generated_assets=True,
                        use_backgrounds=False,
                        restrict_themes=True,
                        use_monochrome_assets=True,
                        rand_seed=seed,
                        center_agent=False,
                        mask_size=0,
                        normalize_rew=normalize_rew,
                        mask_all=True,
                        device=device)

envs_full = make_ProcgenEnvs(num_envs=1,
                        env_name=env_name,
                        start_level=test_start_level,
                        num_levels=num_level,
                        distribution_mode=distribution_mode,
                        use_generated_assets=True,
                        use_backgrounds=False,
                        restrict_themes=True,
                        use_monochrome_assets=True,
                        rand_seed=seed,
                        center_agent=False,
                        mask_size=0,
                        normalize_rew=normalize_rew,
                        mask_all=False,
                        device=device)

actor_critic1 = Policy(
    (3,64,64),
    spaces.Discrete(15),
    base=ImpalaModel,
    base_kwargs={'recurrent': True ,'hidden_size': 256})
actor_critic1.to(device)


saved_epoch = 3050
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/miner_seed_0_num_env_200_entro_0.01_gama_0.999_24-01-2023_22-37-10_noRNN"
# save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/dodgeball_seed_2867_num_env_200_entro_0.01_gama_0.999_19-04-2023_16-41-24_noRNN_original"
save_dir = "/home/ev/Desktop/pytorch-a2c-ppo-acktr-gail/ppo_log/maze_seed_77665_num_env_128_entro_0.01_gama_0.999_19-05-2023_11-40-03_mask_all"
if (saved_epoch > 0) and save_dir != "":
    save_path = save_dir
    actor_critic_weighs = torch.load(os.path.join(save_path, env_name + "-epoch-{}.pt".format(saved_epoch)), map_location=device)
    actor_critic1.load_state_dict(actor_critic_weighs['state_dict'])


# obs = test_env.reset()
eval_recurrent_hidden_states = torch.zeros(
    num_level, 256, device=device)
eval_masks = torch.zeros(num_level, 1, device=device)

eval_attn_masks  = torch.zeros(num_level, actor_critic1.attention_size, device=device)
eval_attn_masks1 = torch.zeros(num_level, 16, device=device)
eval_attn_masks2 = torch.zeros(num_level, 32, device=device)
eval_attn_masks3 = torch.zeros(num_level, 32, device=device)

# actor_critic_maxEnt.eval()
actor_critic1.eval()
# actor_critic2.eval()
# actor_critic3.eval()
# actor_critic4.eval()
# rew, obs, first = test_env.observe()
obs = envs.reset()
obs_full = envs_full.reset()

# obs = obs['rgb']
# obs = torch.FloatTensor(obs.transpose(0, 3, 1, 2) / 255)
# obs_sum = obs
# obs_sum2 = torch.zeros_like(obs)
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
myobj = plt.imshow(obs[0].transpose(0, 2).transpose(1, 0))
plt.show()
myobj = plt.imshow(obs_full[0].cpu().transpose(0, 2).transpose(1, 0))
plt.show()
iter = 0
rew_sum = 0
# beta = 0.5
# moving_average_prob1 = 0
# moving_average_prob2 = 0
# steps_remaining = 30
# step_count = 0
# max_ent_step = 0
# ent_step_count = 0
# ent_last_step_count = 0
# novel = True
#
# # m = FixedCategorical(torch.tensor([ 0.55, 0.25, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.0125, 0.0125, 0.0125, 0.0125]))
# # m = FixedCategorical(torch.tensor([ 0.55, 0.25, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]))
# m = FixedCategorical(torch.tensor([ 0.75, 0.15, 0.05, 0.05]))
#     # m = FixedCategorical(torch.tensor([ 1.0, 0.0]))
# # m = FixedCategorical(torch.tensor([ 0.55, 0.25, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025]))
# # m = m = FixedCategorical(torch.tensor([ 0.55, 0.25, 0.045, 0.025, 0.025, 0.025, 0.015, 0.015, 0.0125, 0.0125, 0.0125, 0.0125]))
# # rand_policy = FixedCategorical(torch.tensor([ 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.0]))
#
# indices_row = torch.tensor([3, 9, 16, 22, 28, 35, 41, 48, 54, 61])
# indices_cal = torch.tensor([3, 10, 16, 22, 28, 35, 42, 48, 54, 61])
#
# maxEnt_steps = 0
# int_reward_sum = 0
# int_reward_sum2 = 0
# rew_sum = 0
# down_sample_avg = nn.AvgPool2d(3, stride=3)
while not done[0] and iter<500:
    iter +=1
    with torch.no_grad():

        _, action1, _, dist_probs1, _, _, _, _, _ = actor_critic1.act(
            obs.float().to(device),
            eval_recurrent_hidden_states,
            eval_masks,
            attn_masks=eval_attn_masks,
            attn_masks1=eval_attn_masks1,
            attn_masks2=eval_attn_masks2,
            attn_masks3=eval_attn_masks3,
            deterministic=False,
            reuse_masks=False)

    obs, reward, done, infos = envs.step(action1.squeeze(0).cpu().numpy())
    obs_full, _, _, _ = envs_full.step(action1.squeeze(0).cpu().numpy())

    # myobj = plt.imshow(obs[0].transpose(0, 2).transpose(1, 0))
    # plt.show()


    eval_masks = torch.tensor(
        [[0.0] if done_ else [1.0] for done_ in done],
        dtype=torch.float32,
        device=device)

    # rew, next_obs, first = test_env.observe()
    rew_sum += reward
    done[0] = done
    step += 1


    print(f"step {step} reward {reward}  first {done} action1 {action1} reward_sum {rew_sum}")