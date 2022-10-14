import torch
from torch import nn
from typing import Dict

@torch.jit.script
class test:
    def __call__(self):
        tensor = torch.randn([5])
        return torch.reshape(tensor, [5])

instance = test()
instance = torch.jit.script(instance)
instance()