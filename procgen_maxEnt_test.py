
from a2c_ppo_acktr.procgen_wrappers import *
# from a2c_ppo_acktr.envs import make_vec_envs, make_ProcgenEnvs
import matplotlib.pyplot as plt
# from procgen import ProcgenEnv
from a2c_ppo_acktr.envs import make_ProcgenEnvs
from a2c_ppo_acktr.const_env import ProcgenConatEnvs
import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


env_name = "maze"
start_level = 3415
num_level = 1
distribution_mode = "easy"
seed = 0
normalize_rew = False
no_normalize = False
n_steps = 1

# device = torch.device("cuda:{}".format(0))

print('making envs...')
# Training envs
envs = make_ProcgenEnvs(num_envs=1,
                        env_name=env_name,
                        start_level=start_level,
                        num_levels=num_level,
                        distribution_mode=distribution_mode,
                        use_generated_assets=True,
                        use_backgrounds=False,
                        restrict_themes=True,
                        use_monochrome_assets=True,
                        rand_seed=seed,
                        mask_size=0,
                        normalize_rew=normalize_rew,
                        mask_all=False)

print(envs.action_space)

obs = envs.reset()
obs_sum = obs
# plot mazes
plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
plt.savefig("test.png")
# plt.show()

for i in range(5):
    action = np.array([5])
    obs, reward, done, infos = envs.step(action)
    # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    # plt.show()
    obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

for i in range(5):
    action = np.array([3])
    obs, reward, done, infos = envs.step(action)
    # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    # plt.show()
    obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

for i in range(6):
    action = np.array([7])
    obs, reward, done, infos = envs.step(action)
    # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    # plt.show()
    obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

for i in range(6):
    action = np.array([5])
    obs, reward, done, infos = envs.step(action)
    # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    # plt.show()
    obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

for i in range(2):
    action = np.array([1])
    obs, reward, done, infos = envs.step(action)
    # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    # plt.show()
    obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

for i in range(5):
    action = np.array([3])
    obs, reward, done, infos = envs.step(action)
    # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    # plt.show()
    obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

for i in range(5):
    action = np.array([5])
    obs, reward, done, infos = envs.step(action)
    # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    # plt.show()
    obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

for i in range(2):
    action = np.array([1])
    obs, reward, done, infos = envs.step(action)
    # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    # plt.show()
    obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

for i in range(5):
    action = np.array([3])
    obs, reward, done, infos = envs.step(action)
    # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    # plt.show()
    obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

for i in range(5):
    action = np.array([5])
    obs, reward, done, infos = envs.step(action)
    # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    # plt.show()
    obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

for i in range(2):
    action = np.array([1])
    obs, reward, done, infos = envs.step(action)
    # plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    # plt.show()
    obs_sum += obs
# plt.imshow(obs_sum[0].transpose(0,2).cpu().numpy())
# plt.show()

print("done")