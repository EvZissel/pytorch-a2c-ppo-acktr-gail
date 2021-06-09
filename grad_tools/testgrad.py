import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class TestGrad():
    def __init__(self, optimizer, max_grad_norm=0.0, noise_ratio=0.0, use_median=False, quantile=-1, alpha=1.0, beta=1.0):
        self._optim = optimizer
        self._max_grad_norm = max_grad_norm
        self._noise_ratio = noise_ratio
        self._use_median = use_median
        self._quantile = quantile
        self._alpha = alpha
        self._beta = beta
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

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads, special_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads, special_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, special_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        special = torch.stack(special_grads).prod(0)
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        grad_norms = []
        for g_i in pc_grad:
            grad_norms.append(g_i.norm())
        noise_std = np.minimum(np.max(np.array(grad_norms)), self._max_grad_norm) * self._noise_ratio
        signs = copy.deepcopy(grads)
        for i, g_i in enumerate(grads):
            signs[i] = torch.sign(g_i)
        scores = torch.abs(torch.stack([g[shared] for g in signs]).mean(dim=0))
        active_elements = scores >= self._beta

        # print((scores >= 1.0).float().mean())
        # analysis
        # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        # train_grad = torch.stack([g[shared] for g in grads[:-3]]).mean(dim=0)
        # val_grad = []
        # # half = len(train_grad) // 2
        # update = True
        # for i in range(3):
        #     val_grad.append(cos(train_grad[:], grads[-(i+1)][:]))
        #     update = update and (val_grad[-1] > 0)
        # print(val_grad, update)

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        stacked_grads = torch.stack([(g * active_elements)[shared] for g in grads])
        if self._use_median:
            merged_grad[shared] = torch.median(stacked_grads, dim=0)[0]
        elif self._quantile > 0:
            signs = torch.sign(stacked_grads[0]) # all grads have same signs
            merged_grad[shared] = torch.quantile(stacked_grads.abs(), self._quantile, dim=0) * signs
        else:
            merged_grad[shared] = stacked_grads.mean(dim=0)
        if noise_std > 0:
            merged_grad += torch.normal(torch.zeros_like(grads[0]), noise_std)
        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        standard_grad = torch.stack([g[shared] for g in grads]).mean(dim=0)
        alpha = special * self._alpha
        # if update == False:
        #     print('no update')
        #     return 0 * standard_grad
        return alpha * merged_grad + (1-alpha) * standard_grad

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

        grads, shapes, has_grads, special_grads = [], [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad, special_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            special_grads.append(self._flatten_grad(special_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads, special_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
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
        - special_grad: a list of mask represent whether the parameter should be applied a special/standard gradient
        '''

        grad, shape, has_grad, special_grad = [], [], [], []
        for group in self._optim.param_groups:
            apply_special_grad_to_group = True
            if "special_grad" in group:
                apply_special_grad_to_group = group['special_grad']
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if apply_special_grad_to_group:
                    special_grad.append(torch.ones_like(p).to(p.device))
                else:
                    special_grad.append(torch.zeros_like(p).to(p.device))
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad, special_grad


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
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)
    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)
