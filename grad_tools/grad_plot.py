import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class GradPlot():
    def __init__(self, optimizer):
        self._optim = optimizer
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def state_dict(self):
        '''
        Return the optimizer parameters
        '''

        return self._optim.state_dict()

    def plot_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        mean_flat_grad = self._merge_grad(grads, has_grads)
        mean_grad = self._unflatten_grad(mean_flat_grad, shapes[0])
        self._set_grad(mean_grad)
        F_norms_all = []
        F_norms_gru = []
        F_norms_actor = []
        F_norms_critic = []
        F_norms_cat = []

        for g in grads:
            F_norms_all.append((g-mean_flat_grad).norm(2))
            g = self._unflatten_grad(g, shapes[0])
            g_gru = self._flatten_grad(g[0:4])
            mean_gru = self._flatten_grad(mean_grad[0:4])
            F_norms_gru.append((g_gru - mean_gru).norm(2))
            g_actor = self._flatten_grad(g[4:8])
            mean_actor = self._flatten_grad(mean_grad[4:8])
            F_norms_actor.append((g_actor - mean_actor).norm(2))
            g_critic = self._flatten_grad(g[8:14])
            mean_critic  = self._flatten_grad(mean_grad[8:14])
            F_norms_critic.append((g_critic - mean_critic).norm(2))
            g_cat = self._flatten_grad(g[14:])
            mean_cat  = self._flatten_grad(mean_grad[14:])
            F_norms_cat.append((g_cat - mean_cat).norm(2))


        return mean_flat_grad, grads, shapes[0], F_norms_all, F_norms_gru, F_norms_actor, F_norms_critic, F_norms_cat


    def _merge_grad(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        plot_grad, num_task = copy.deepcopy(grads), len(grads)

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        # return the mean of the gradient
        merged_grad[shared] = torch.stack([g[shared]
                                           for g in plot_grad]).mean(dim=0)
        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in plot_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad))
            has_grads.append(self._flatten_grad(has_grad))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class GradPlotDqn():
    def __init__(self, optimizer):
        self._optim = optimizer
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def state_dict(self):
        '''
        Return the optimizer parameters
        '''

        return self._optim.state_dict()

    def load_state_dict(self, optimizer_state_dict):
        '''
        Load optimizer state
        '''

        return self._optim.load_state_dict(optimizer_state_dict)

    def plot_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        mean_flat_grad = self._merge_grad(grads, has_grads)
        mean_grad = self._unflatten_grad(mean_flat_grad, shapes[0])
        self._set_grad(mean_grad)
        F_norms_all = []
        F_norms_gru = []
        F_norms_dqn = []

        for g in grads:
            F_norms_all.append((g - mean_flat_grad).norm(2))
            g = self._unflatten_grad(g, shapes[0])
            g_gru = self._flatten_grad(g[0:4])
            mean_gru = self._flatten_grad(mean_grad[0:4])
            F_norms_gru.append((g_gru - mean_gru).norm(2))
            g_dqn = self._flatten_grad(g[4:])
            mean_dqn = self._flatten_grad(mean_grad[4:])
            F_norms_dqn.append((g_dqn - mean_dqn).norm(2))

        return mean_flat_grad, grads, shapes[0], F_norms_all, F_norms_gru, F_norms_dqn

    def _merge_grad(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        plot_grad, num_task = copy.deepcopy(grads), len(grads)

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        # return the mean of the gradient
        merged_grad[shared] = torch.stack([g[shared]
                                           for g in plot_grad]).mean(dim=0)
        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in plot_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad))
            has_grads.append(self._flatten_grad(has_grad))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class GradPlot_moons():
    def __init__(self, optimizer):
        self._optim = optimizer
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def state_dict(self):
        '''
        Return the optimizer parameters
        '''

        return self._optim.state_dict()

    def plot_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        mean_flat_grad = self._merge_grad(grads, has_grads)
        mean_grad = self._unflatten_grad(mean_flat_grad, shapes[0])
        self._set_grad(mean_grad)

        return mean_flat_grad, grads, shapes[0]

    def _merge_grad(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        plot_grad, num_task = copy.deepcopy(grads), len(grads)

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        # return the mean of the gradient
        merged_grad[shared] = torch.stack([g[shared]
                                           for g in plot_grad]).mean(dim=0)
        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in plot_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad))
            has_grads.append(self._flatten_grad(has_grad))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    grad_plot_adam = GradPlot(optim.Adam(net.parameters()))
    grad_plot_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    grad_plot_adam.plot_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    adam = optim.Adam(net.parameters())
    adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    total_loss = (loss1 + loss2)/2
    total_loss.backward()
    for p in net.parameters():
        print(p.grad)

    # seperated shared network test
    #
    # torch.manual_seed(4)
    # x, y = torch.randn(2, 3), torch.randn(2, 4)
    # net = MultiHeadTestNet()
    # y_pred_1, y_pred_2 = net(x)
    # pc_adam = Grad_plot(optim.Adam(net.parameters()))
    # pc_adam.zero_grad()
    # loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    # loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)
    #
    # pc_adam.plot_backward([loss1, loss2])
    # for p in net.parameters():
    #     print(p.grad)
