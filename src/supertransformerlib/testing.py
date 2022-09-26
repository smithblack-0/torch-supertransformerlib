from typing import Optional, Union, Dict, List, overload
import torch
from torch import nn
from torch.nn import Module
import dataclasses


@torch.jit.script
def selector(tensor, a, b)->torch.Tensor:
    return tensor[a, b]

tensor = torch.randn([300, 300])
a = torch.randint(0, 300, [10, 10])
b = torch.randint(0, 300, [10, 10])
print(selector(tensor, a, b))