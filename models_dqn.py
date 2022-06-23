#!/usr/bin/env python3

import torch
import torch.nn as nn
from a2c_ppo_acktr.model import NNBase
from a2c_ppo_acktr.utils import init
import numpy as np

class Deep_feature(nn.Module):
    def __init__(self, input_shape, feature_dimension, num_actions):
        super(Deep_feature, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), feature_dimension),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)


class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)


class DQN(NNBase):
    def __init__(self, input_shape, num_actions, zero_ind, recurrent=False, hidden_size=64):
        super(DQN, self).__init__(recurrent, input_shape[0], hidden_size)
        self.input_shape = input_shape
        self.num_actions = num_actions

        num_inputs = input_shape[0]
        self.zero_ind = zero_ind
        if recurrent:
            num_inputs = hidden_size

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.ReLU(), nn.Linear(hidden_size, self.num_actions)
        )

    def forward(self, x, rnn_hxs, masks, attn_masks):
        x = attn_masks * x
        if self.zero_ind:
            x = torch.cat((torch.zeros(x.size()[1]-2), torch.ones(2)),0).cuda() * x
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        return self.layers(x), rnn_hxs

class DQN_softAttn(NNBase):
    def __init__(self, input_shape, num_actions, zero_ind, recurrent=False, hidden_size=64, target=False):
        super(DQN_softAttn, self).__init__(recurrent, input_shape[0], hidden_size)
        self.input_shape = input_shape
        self.num_actions = num_actions

        num_inputs = input_shape[0]
        self.zero_ind = zero_ind
        self.target = target
        if recurrent:
            num_inputs = hidden_size

        # # should be double precision for finite difference
        # self.layers = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size).double(), nn.ReLU(), nn.Linear(hidden_size, self.num_actions).double()
        # )
        # self.input_attention = nn.Parameter(torch.ones(input_shape).double(), requires_grad=True)
        # self.gru.double()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.ReLU(), nn.Linear(hidden_size, self.num_actions)
        )
        self.input_attention = nn.Parameter(torch.ones(input_shape), requires_grad=True)


    def forward(self, x, rnn_hxs, masks):
        # with torch.backends.cudnn.flags(enabled=False):
        if self.target:
            x = (torch.sigmoid(self.input_attention.data) > 0.5).to(self.input_attention.dtype) * x
        else:
            # x = self.activation(self.input_attention) * x
            x = torch.sigmoid(self.input_attention) * x
        if self.zero_ind:
            x = torch.cat((torch.zeros(x.size()[1]-2), torch.ones(2)),0).cuda() * x
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.layers(x), rnn_hxs


class DQN_softAttn_L2grad(NNBase):
    def __init__(self, input_shape, num_actions, zero_ind, recurrent=False, hidden_size=64, target=False):
        super(DQN_softAttn_L2grad, self).__init__(recurrent, input_shape[0], hidden_size)
        self.input_shape = input_shape
        self.num_actions = num_actions

        num_inputs = input_shape[0]
        self.zero_ind = zero_ind
        self.target = target
        if recurrent:
            num_inputs = hidden_size

        # should be double precision for finite difference
        # self.layers = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size).double(), nn.ReLU(), nn.Linear(hidden_size, self.num_actions).double()
        # )
        # self.input_attention = nn.Parameter(torch.ones(input_shape).double(), requires_grad=True)
        # self.gru.double()

        self.layer_1 = nn.Linear(num_inputs, hidden_size)
        nn.init.orthogonal_(self.layer_1.weight)
        nn.init.zeros_(self.layer_1.bias)

        self.activation = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, self.num_actions)
        nn.init.orthogonal_(self.layer_2.weight)
        nn.init.zeros_(self.layer_2.bias)

        self.input_attention = nn.Parameter(torch.ones(input_shape), requires_grad=True)


    def forward(self, x, rnn_hxs, masks):
        # with torch.backends.cudnn.flags(enabled=False):
        if self.target:
            x = (torch.sigmoid(self.input_attention.data) > 0.5).to(self.input_attention.dtype) * x
        else:
            # x = self.activation(self.input_attention) * x
            x = torch.sigmoid(self.input_attention) * x
        if self.zero_ind:
            x = torch.cat((torch.zeros(x.size()[1] - 2), torch.ones(2)), 0).cuda() * x
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        out_1 = self.activation(self.layer_1(x))
        out_2 = self.layer_2(out_1)

        return out_2, out_1, rnn_hxs


class DQN_RNNLast(NNBase):
    def __init__(self, input_shape, num_actions, zero_ind, recurrent=False, hidden_size=64, target=False):
        super(DQN_RNNLast, self).__init__(recurrent, hidden_size, hidden_size)
        self.input_shape = input_shape
        self.num_actions = num_actions

        num_inputs = input_shape[0]
        self.zero_ind = zero_ind
        self.target = target
        # if recurrent:
        #     num_inputs = hidden_size

        # should be double precision for finite difference
        # self.layers = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size).double(), nn.ReLU(), nn.Linear(hidden_size, self.num_actions).double()
        # )
        # self.input_attention = nn.Parameter(torch.ones(input_shape).double(), requires_grad=True)
        # self.gru.double()

        self.layer_1 = nn.Linear(num_inputs, hidden_size)
        nn.init.orthogonal_(self.layer_1.weight)
        nn.init.zeros_(self.layer_1.bias)

        self.activation = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        nn.init.orthogonal_(self.layer_2.weight)
        nn.init.zeros_(self.layer_2.bias)

        self.layer_last = nn.Linear(hidden_size, self.num_actions)
        nn.init.orthogonal_(self.layer_last.weight)
        nn.init.zeros_(self.layer_last.bias)

        # self.input_attention = nn.Parameter(torch.ones(input_shape[0],input_shape[0]), requires_grad=True)
        self.input_attention = nn.Parameter(torch.ones(input_shape[0]), requires_grad=True)


    def forward(self, x, rnn_hxs, masks):
        # with torch.backends.cudnn.flags(enabled=False):
        if self.target:
            x = (torch.sigmoid(self.input_attention.data) > 0.5).to(self.input_attention.dtype) * x
        else:
            # x = self.activation(self.input_attention) * x
            x = torch.sigmoid(self.input_attention) * x
            # x = torch.transpose(torch.matmul(torch.sigmoid(self.input_attention), torch.transpose(x,1,0)),1,0)
        if self.zero_ind:
            x = torch.cat((torch.zeros(x.size()[1] - 2), torch.ones(2)), 0).cuda() * x

        out_1 = self.activation(self.layer_1(x))
        out_2 = self.layer_2(out_1)

        if self.is_recurrent:
            out_2, rnn_hxs = self._forward_gru(out_2, rnn_hxs, masks)

        out = self.layer_last(out_2)

        return out, out_2, rnn_hxs


class DQN_RNNLast_analytic(NNBase):
    def __init__(self, input_shape, num_actions, zero_ind, recurrent=False, hidden_size=64, target=False):
        super(DQN_RNNLast_analytic, self).__init__(recurrent, hidden_size, hidden_size)
        self.input_shape = input_shape
        self.num_actions = num_actions

        num_inputs = input_shape[0]
        self.zero_ind = zero_ind
        self.target = target


        self.layer_1 = nn.Linear(num_inputs, hidden_size)
        nn.init.orthogonal_(self.layer_1.weight)
        nn.init.zeros_(self.layer_1.bias)

        self.activation = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        nn.init.orthogonal_(self.layer_2.weight)
        nn.init.zeros_(self.layer_2.bias)

        self.layer_last = nn.Linear(hidden_size, self.num_actions)
        nn.init.orthogonal_(self.layer_last.weight)
        nn.init.zeros_(self.layer_last.bias)

        # self.input_attention = nn.Parameter(torch.ones(input_shape[0],input_shape[0]), requires_grad=True)
        if self.zero_ind:
            self.input_attention = nn.Parameter(torch.cat((-1000000*torch.ones(input_shape[0] - 2), 1000000*torch.ones(2)), 0), requires_grad=False)
        else:
            self.input_attention = nn.Parameter(1000000*torch.ones(input_shape[0]), requires_grad=False)
        self.input_attention_sig = nn.Parameter(torch.ones(input_shape[0]), requires_grad=True)
        self.input_attention_sig.data = torch.sigmoid(self.input_attention).data


    def forward(self, x, rnn_hxs, masks):
        # with torch.backends.cudnn.flags(enabled=False):
        x[:, 6] /= 10
        if self.target:
            x = (torch.sigmoid(self.input_attention.data) > 0.5).to(self.input_attention.dtype) * x
        else:
            # x = self.activation(self.input_attention) * x
            self.input_attention_sig.data = torch.sigmoid(self.input_attention).data
            x = self.input_attention_sig * x
            # x = torch.transpose(torch.matmul(torch.sigmoid(self.input_attention), torch.transpose(x,1,0)),1,0)
        # if self.zero_ind:
        #    x  = torch.cat((torch.zeros(x.size()[1] - 2), torch.ones(2)), 0).cuda() * x

        out_1 = self.activation(self.layer_1(x))
        out_2 = self.layer_2(out_1)

        if self.is_recurrent:
            out_2, rnn_hxs = self._forward_gru(out_2, rnn_hxs, masks)

        out = self.layer_last(out_2)

        return out, out_2, rnn_hxs


class DQN_Attention(NNBase):
    def __init__(self, input_shape, num_actions, recurrent=False, hidden_size=64):
        super(DQN_Attention, self).__init__(recurrent, input_shape[0], hidden_size)
        self.input_shape = input_shape
        self.num_actions = num_actions

        num_inputs = input_shape[0]
        if recurrent:
            num_inputs = hidden_size

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 64), nn.ReLU(), nn.Linear(64, self.num_actions)
        )
        # self.input_attention = nn.Parameter(torch.cat((-1000000*torch.ones(input_shape[0]-2), torch.ones(2)),0), requires_grad=True)
        self.input_attention = nn.Parameter(torch.ones(input_shape), requires_grad=True)

    def forward(self, x, rnn_hxs, masks, attn_masks=None, reuse_masks=False):
        # if self.is_recurrent:
        #     x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        # return self.layers(x), rnn_hxs

        probs = torch.sigmoid(self.input_attention.repeat([x.shape[0], 1]))
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        sampled_mask = probs.sample() / (1 - probs.probs)
        # sampled_mask = sampled_mask.repeat([x.shape[0], 1])
        new_attn_masks = attn_masks if reuse_masks else sampled_mask
        attn_masks = new_attn_masks * (1 - masks) + attn_masks * masks  # masks=1 in first step of episode
        attn_log_probs = probs.log_prob((attn_masks > 0).float()).sum(dim=1).reshape([x.shape[0], 1])
        x = attn_masks * x

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.layers(x), rnn_hxs, attn_log_probs, attn_masks

    def attn_log_probs(self, attn_masks):
        probs = torch.sigmoid(self.input_attention)
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        attn_log_probs = probs.log_prob((attn_masks>0).float()).sum(dim=1).reshape([attn_masks.shape[0], 1])
        return attn_log_probs


class DQN_Attention_Wvalue(NNBase):
    def __init__(self, input_shape, num_actions, recurrent=False, hidden_size=64):
        super(DQN_Attention_Wvalue, self).__init__(recurrent, input_shape[0], hidden_size)
        self.input_shape = input_shape
        self.num_actions = num_actions

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        num_inputs = input_shape[0]
        if recurrent:
            num_inputs = hidden_size

        self.layers = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)), nn.ReLU(), init_(nn.Linear(64, self.num_actions))
        )
        # self.input_attention = nn.Parameter(torch.cat((-1000000*torch.ones(input_shape[0]-2), torch.ones(2)),0), requires_grad=True)
        self.input_attention = nn.Parameter(torch.ones(input_shape), requires_grad=True)
        self.value = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(), init_(nn.Linear(hidden_size, 1)))

    def forward(self, x, rnn_hxs, masks, attn_masks=None, reuse_masks=False):
        # if self.is_recurrent:
        #     x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        # return self.layers(x), rnn_hxs

        probs = torch.sigmoid(self.input_attention.repeat([x.shape[0], 1]))
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        sampled_mask = probs.sample() / (1 - probs.probs)
        # sampled_mask = sampled_mask.repeat([x.shape[0], 1])
        new_attn_masks = attn_masks if reuse_masks else sampled_mask
        attn_masks = new_attn_masks * (1 - masks) + attn_masks * masks  # masks=1 in first step of episode
        attn_log_probs = probs.log_prob((attn_masks > 0).float()).sum(dim=1).reshape([x.shape[0], 1])
        x = attn_masks * x

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.layers(x), self.value(x), rnn_hxs, attn_log_probs, attn_masks

    def attn_log_probs(self, attn_masks):
        probs = torch.sigmoid(self.input_attention)
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        attn_log_probs = probs.log_prob((attn_masks>0).float()).sum(dim=1).reshape([attn_masks.shape[0], 1])
        return attn_log_probs