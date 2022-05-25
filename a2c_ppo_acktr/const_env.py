from .procgen_wrappers import *
from procgen import ProcgenEnv
from .envs import VecPyTorch

import random

class ProcgenConatEnvs(object):
    def __init__(self,
                 env_name,
                 num_envs=1,
                 start_level=0,
                 distribution_mode='easy',
                 use_generated_assets=True,
                 use_backgrounds=False,
                 restrict_themes=True,
                 use_monochrome_assets=True,
                 normalize_rew=True,
                 num_stack=1,
                 seed=0,
                 device="cpu",
                 mask_size=0,
                 mask_all=False,
                 **kwargs):

        self.num_envs = num_envs
        # randomlist = []
        # for i in range(num_envs):
        #     n = random.randint(start_level, 2 ** 31 - 1)
        #     # randomlist.append(i)
        #     randomlist.append(n)
        # print(randomlist)
        self.env_list = []
        for i in range(num_envs):
            self.env_list.append(ProcgenEnv(num_envs=1,
                                          env_name=env_name,
                                          start_level=start_level + i,
                                          num_levels=1,
                                          distribution_mode=distribution_mode,
                                          use_generated_assets=use_generated_assets,
                                          use_backgrounds=use_backgrounds,
                                          restrict_themes=restrict_themes,
                                          use_monochrome_assets=use_monochrome_assets,
                                          rand_seed=seed))

            self.env_list[i] = VecExtractDictObs(self.env_list[i], "rgb")
            self.env_list[i] = VecFrameStack(self.env_list[i], num_stack)
            if normalize_rew:
                self.env_list[i] = VecNormalize(self.env_list[i],ob=False)  # normalizing returns, but not the img frames.
            self.env_list[i] = TransposeFrame(self.env_list[i])
            # envs = MaskFloatFrame(envs,l=mask_size)
            self.env_list[i] = VecPyTorch(self.env_list[i], device)
            if mask_size > 0:
                self.env_list[i] = MaskFrame(self.env_list[i], l=mask_size, device=device)
            if mask_all:
                self.env_list[i] = MaskAllFrame(self.env_list[i])
            self.env_list[i] = ScaledFloatFrame(self.env_list[i])

        self.action_space = self.env_list[0].action_space
        self.observation_space = self.env_list[0].observation_space


    def reset(self):
        full_obs_test = self.env_list[0].reset()
        for i in range(1,self.num_envs):
            obs_test = self.env_list[i].reset()
            full_obs_test = torch.cat((full_obs_test, obs_test), 0)

        return full_obs_test

    def close(self):
        for i in range(self.num_envs):
            self.env_list[i].close()
        return

    def step(self,action):
        full_obs_test, full_rew_test, full_done_test, full_info_test = self.env_list[0].step(action[0:1])
        for i in range(1,self.num_envs):
            next_obs_test, rew_test, done_test, info_test = self.env_list[i].step(action[i:i+1])
            full_obs_test = torch.cat((full_obs_test, next_obs_test), 0)
            full_rew_test = np.concatenate((full_rew_test, rew_test), 0)
            full_done_test = np.concatenate((full_done_test, done_test), 0)
            full_info_test.append(info_test[0])

        return full_obs_test, full_rew_test, full_done_test, full_info_test
