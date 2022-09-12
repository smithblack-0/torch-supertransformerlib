from typing import Optional, Union, Dict, List, overload
import torch
from torch import nn
from torch.nn import Module

@torch.jit.script
class conf:
    def __init__(self, item: int):
        self.item = item

class base_layer(nn.Module):
    @torch.jit.export
    def modify(self, item : conf):
        self.value = item.item
        for child in self.children():
            if isinstance(child, base_layer):
                child.modify(item)

    def __init__(self):
        super().__init__()
        self.is_base = True
        self.value = 3

class inheriting_layer(base_layer):
    def __init__(self):
        super().__init__()
        self.sublayer = base_layer()
    def forward(self):
       return self.sublayer.value, self.value


instance = inheriting_layer()
instance = torch.jit.script(instance)
print(instance())
instance.modify(conf(6))
print(instance())