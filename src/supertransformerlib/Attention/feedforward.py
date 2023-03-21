"""
A module for feedforward and support functions to
chill in
"""
from typing import Optional, Union, List

import torch
from torch import nn
from supertransformerlib.Basics import linear

class Feedforward(nn.Module):
    """
    The feedforward layer. Capable of being built in a
    parallel kernel ensemble.
    """
    def __init__(self,
                 d_model: int,
                 d_internal: Optional[int] = None,
                 d_output: Optional[int] = None,
                 parallel: Optional[Union[torch.Tensor, List[int], int]] = None,
                 activation: nn.Module = nn.ReLU()
                 ):
        """

        :param d_model: The input embedding dimension
        :param d_internal: The size of the internal embedding space. May be none defaulting to 2048
        :param d_output: The size of the output embedding dim. If none, defaults to d_model
        :param parallel: The shape of the parallel kernel. May be none
        :param activation: The activation layer. Defaults to relu.
        """

        super().__init__()

        if d_internal is None:
            d_internal = 2048
        if d_output is None:
            d_output = d_model

        self.ff1 = linear.Linear(d_model, d_internal, parallel)
        self.ff2 = linear.Linear(d_internal, d_output, parallel)
        self.activation = activation


    def forward(self, tensor: torch.Tensor)->torch.Tensor:
        """
        Peforms the feedforward operation
        """
        tensor = tensor.movedim(-2, 0)  # Transfer the item channel out of the wayh
        tensor = self.ff1(tensor)  # (item, ..., (config), (ensemble), internal)
        tensor = self.activation(tensor)
        tensor = self.ff2(tensor)  # (item,..., (config) , (ensemble), embedding)
        tensor = tensor.movedim(0, -2)  # Transfer item back into position
        return tensor
