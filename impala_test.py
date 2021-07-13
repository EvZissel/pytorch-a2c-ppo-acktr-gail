import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.model import NNBase
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.envs import make_ProcgenEnvs

class Flatten(nn.Module):
    def forward(self, x):
        # return x.view(x.size(0), -1)
        return x.reshape(x.size(0), -1)

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))


        self.conv1 = init_(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1))
        self.conv2 = init_(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x



class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv = init_(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaModel(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256):
        super(ImpalaModel, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            ImpalaBlock(in_channels=num_inputs, out_channels=16),
            ImpalaBlock(in_channels=16, out_channels=32),
            ImpalaBlock(in_channels=32, out_channels=32), nn.ReLU(), Flatten(),
            nn.Linear(in_features=32 * 8 * 8, out_features=256),nn.ReLU())


        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, attn_masks=None, reuse_masks=False):

        x = self.main(inputs)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs, None, None

envs = make_ProcgenEnvs(num_envs=3,
                      env_name='maze',
                      start_level=0,
                      num_levels=128,
                      distribution_mode='easy',
                      use_generated_assets=True,
                      use_backgrounds=False,
                      restrict_themes=True,
                      use_monochrome_assets=True,
                      rand_seed=0,
                      normalize_rew= False,
                      no_normalize = False)


obs = envs.reset()
network = ImpalaModel(3, recurrent=False)
x = ImpalaModel.main(obs)
print("debug")
