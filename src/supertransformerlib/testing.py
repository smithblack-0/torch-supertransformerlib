import torch
from torch import nn


class test(nn.Module):
    class subtest:
        pass
    def __init__(self):
        super().__init__()
    def forward(self):
        return torch.rand([10])


instance = test()
instance = torch.jit.script(instance)
