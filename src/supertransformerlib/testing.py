from typing import Optional, Union, Dict, List, overload
import torch
from torch import nn


class EnsembleSpace(nn.Module):
    def __getattr__(self, item: str):
        if item == "bob":
            return 4
        return super().__getattr__(item)
    def __init__(self):
        super().__init__()
        self.item = 3
        self.kernel = nn.Parameter(torch.randn([5, 3]))
        self.kernel2 = self.kernel + 1
    def forward(self):

        return self.item, self.bob



instance = EnsembleSpace()
instance = torch.jit.script(instance)
print(instance())