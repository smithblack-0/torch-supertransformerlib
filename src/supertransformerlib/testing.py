from typing import Optional, Union, Dict, List, overload
import torch
from torch import nn
from torch.nn import Module
import dataclasses

@torch.jit.script
class test_set:

    def __init__(self,
                 tensor: torch.Tensor,
                 integer: int,
                 ):

        self.tensor = tensor
        self.integer = integer

@torch.jit.script
def test_instancing():
    instance = test_set(torch.randn([3, 4]), 5)
    instance.integer = 7
    instance.tensor = torch.randn([6, 7])
    return instance.tensor, instance.integer


print(test_instancing())