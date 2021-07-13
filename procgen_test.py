


from a2c_ppo_acktr.procgen_wrappers import *
from a2c_ppo_acktr.envs import make_vec_envs, make_ProcgenEnvs
import matplotlib.pyplot as plt

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, FixedCategorical



num_processes=128
env_name="maze"
start_level =0
num_level=128
distribution_mode="easy"
seed=0
normalize_rew=False
no_normalize=False
n_steps = 10

envs = make_ProcgenEnvs(num_envs=num_processes,
                        env_name=env_name,
                        start_level=start_level,
                        num_levels=num_level,
                        distribution_mode=distribution_mode,
                        use_generated_assets=True,
                        use_backgrounds=False,
                        restrict_themes=True,
                        use_monochrome_assets=True,
                        rand_seed=seed,
                        normalize_rew= normalize_rew,
                        no_normalize = no_normalize)

obs = envs.reset()

# fig = plt.figure(figsize=(20, 20))
# columns = 2
# rows = 2
# for i in range(1, columns * rows + 1):
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(obs[i - 1].transpose(1, 2, 0))
# # plt.imshow(test_obs[0,:,:,:].transpose(1, 2, 0))
# plt.show()

num_outputs = envs.action_space.n
log_probs = (1/9) * torch.ones(num_processes, 9)
dist = torch.distributions.Categorical(logits=log_probs)


for _ in range(n_steps):

    fig = plt.figure(figsize=(20, 20))
    columns = 2
    rows = 2
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(obs[i - 1].transpose(1, 2, 0))
    # plt.imshow(test_obs[0,:,:,:].transpose(1, 2, 0))
    plt.show()

    act = dist.sample().numpy()
    next_obs, rew, done, info = envs.step(act)
    obs = next_obs
