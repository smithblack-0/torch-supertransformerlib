"""
A module for feedforward and support functions to
chill in

"""
from typing import Optional, Union, List

import torch
from torch import nn

import src.supertransformerlib
from src.supertransformerlib import Basics

class _FeedForward:
    """
    Responsible for actually executing the forward call
    of the feedforward mechanism. Is a closure of sorts.

    Not user servicable. User should use the factory
    class instead.
    """
    def __init__(self,
                 ff1: Basics.LinearFactory.Type,
                 ff2: Basics.LinearFactory.Type
                 ):
        self.ff1 = ff1
        self.ff2 = ff2
    def __call__(self, tensor: torch.Tensor)->torch.Tensor:
        tensor = tensor.unsqueeze(0).transpose(0, -2).squeeze(-2)  # Transfer the item channel out of the wayh
        tensor = self.ff1(tensor)  # (item, ..., (config), (ensemble), internal)
        tensor = torch.relu(tensor)
        tensor = self.ff2(tensor)  # (item,..., (config) , (ensemble), embedding)
        tensor = tensor.unsqueeze(-2).transpose(0, -2).squeeze(0)  # Transfer item back into position
        return tensor

torch.jit.script(_FeedForward)

class FeedForwardFactory(nn.Module):
    """
    A feedforward factory layer.

    Creates passable feedforward layers for use in
    autodifferentiation and other useful model tricks.
    Allows parallel ensembles as illistrated on
    class LinearFactory

    Displays the type in the standard location.
    """
    Type = _FeedForward
    def __init__(self,
                 d_model: int,
                 d_internal: Optional[int] = None,
                 d_output: Optional[int] = None,
                 parallel: Optional[Union[torch.Tensor, List[int], int]] = None,
                 ):
        """
        :param d_model: The model width
        :param d_internal: The internel width. By default 2048
        :param d_output: The output widht. If left as none, same as d_model
        :param parallel: The parallel kernels dimensions.
        :param dynamics: The dynamic width. None means off.
        """
        super().__init__()

        if d_internal is None:
            d_internal = 2048
        if d_output is None:
            d_output = d_model

        self.ff1 = Basics.LinearFactory(d_model, d_internal, parallel)
        self.ff2 = Basics.LinearFactory(d_internal, d_output, parallel)
    def forward(self)->_FeedForward:
        return _FeedForward(
            self.ff1(),
            self.ff2()
        )