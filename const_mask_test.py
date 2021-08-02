from a2c_ppo_acktr.procgen_wrappers import *
# from a2c_ppo_acktr.envs import make_vec_envs, make_ProcgenEnvs
import matplotlib.pyplot as plt
from procgen import ProcgenEnv
from scipy import signal

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, FixedCategorical


num_processes=128
env_name="maze"
start_level =0
num_level=128
distribution_mode="easy"
seed=0
normalize_rew=False
no_normalize=False

envs = ProcgenEnv(num_envs=num_processes,
                  env_name=env_name,
                  start_level=start_level,
                  num_levels=num_level,
                  distribution_mode=distribution_mode,
                  use_generated_assets=True,
                  use_backgrounds=False,
                  restrict_themes=True,
                  use_monochrome_assets=True)

envs = VecExtractDictObs(envs, "rgb")
if normalize_rew:
    envs = VecNormalize(envs, ob=False)  # normalizing returns, but not the img frames.
envs = TransposeFrame(envs)
envs = MaskFloatFrame(envs, l=0)
# envs = ScaledFloatFrame(envs)

obs = envs.reset()

brown = np.stack([191 * np.ones((64, 64)), 127 *  np.ones((64, 64)), 63 *  np.ones((64, 64))], axis=0)
brown = np.expand_dims(brown, axis=0)
brown = np.repeat(brown, obs.shape[0], axis=0)
# result = signal.convolve2d(obs[0], kernel, boundary='fill', mode='same')
indexes = np.stack([(obs[:,1,:,:] == 255),(obs[:,1,:,:] == 255),(obs[:,1,:,:] == 255)], axis=1)
l = 8

indexes_l = np.zeros(indexes.shape)
for i in range(1,l+1):
    indexes_l[i:]  = indexes[:-i]


# indexes = np.where(obs[0][1] == 255)
# row_min = max(indexes[0].min()-l, 0)
# row_max = min(indexes[0].max()+l,64-1)
#
# cal_min = max(indexes[1].min()-l, 0)
# cal_max = min(indexes[1].max()+l,64-1)
#
# mask = np.zeros([64,64])
#
# mask[row_min:row_max,cal_min:cal_max] = 1
# mask = np.expand_dims(mask, axis=0)
#
# obs_0 = obs[0] * mask + brown * (1-mask)

obs = obs * indexes + brown * (1-indexes)

# plt.imshow((obs[0:25]/255).transpose(1, 2, 0))
fig = plt.figure(figsize=(20, 20))
columns = 5
rows = 5
for i in range(1, columns * rows + 1):
    fig.add_subplot(rows, columns, i)
    plt.imshow((obs[i]/255).transpose(1, 2, 0))

plt.show()

print("debug")
