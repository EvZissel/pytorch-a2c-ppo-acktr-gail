
from a2c_ppo_acktr.procgen_wrappers import *
# from a2c_ppo_acktr.envs import make_vec_envs, make_ProcgenEnvs
import matplotlib.pyplot as plt
# from procgen import ProcgenEnv
from a2c_ppo_acktr.envs import make_ProcgenEnvs
from a2c_ppo_acktr.const_env import ProcgenConatEnvs
from a2c_ppo_acktr.model import Policy, MLPAttnBase, MLPHardAttnBase, MLPHardAttnReinforceBase, ImpalaModel
from evaluation import maxEnt_oracle
import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

device = torch.device("cuda:{}".format(0))


env_name = "maze"
start_level = 0
num_level = 1
distribution_mode = "easy"
seed = 0
normalize_rew = False
no_normalize = False
n_steps = 1
max_train_envs = 200

reward_seeds = []

# device = torch.device("cuda:{}".format(0))

print('making envs...')
# Training envs
for i in range(max_train_envs):
    envs = make_ProcgenEnvs(num_envs=1,
                            env_name=env_name,
                            start_level=start_level+i,
                            num_levels=num_level,
                            distribution_mode=distribution_mode,
                            use_generated_assets=False,
                            use_backgrounds=False,
                            restrict_themes=True,
                            use_monochrome_assets=True,
                            rand_seed=seed,
                            mask_size=0,
                            normalize_rew=normalize_rew,
                            mask_all=False)

    # print(envs.action_space)

    obs = envs.reset()
    obs_sum = obs
    # plot mazes
    plt.imshow(obs[0].transpose(0,2).cpu().numpy())
    # plt.savefig("test.png")
    plt.show()

    action = torch.full((1,1), 5)
    done = torch.full((1,1), 0)
    reward = 0

    while not done[0]:
        with torch.no_grad():

            action = maxEnt_oracle(obs,action)

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

    reward_seeds.append(reward)


print("done")
# print("reward = {}".format(reward))
# print("zero sum = {}".format((obs_sum[0] == 0).sum()))