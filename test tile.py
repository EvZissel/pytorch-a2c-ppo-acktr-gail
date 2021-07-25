import numpy as np
import torch
import torch.nn as nn


att_size = [3, 3]
input_attention = nn.Parameter(torch.rand(att_size), requires_grad=True)
B = torch.ones(2, 2)
mask = torch.kron(input_attention, B)
mask = torch.kron(input_attention, B).repeat([128,3,1,1])
print("done")