"""
A module for feedforward and support functions to
chill in

"""
from typing import Optional, Union, List

import torch
from torch import nn
from src.supertransformerlib import Basics


class FeedForward(nn.Module):
    """
    Responsible for executing the feedforward
    mechanism when requested. Will return
    the same input embedding as the output
    embedding if not explictly told not to.

    Ensembe capable.
    """
    def __init__(self,
                 d_model: int,
                 d_internal: Optional[int] = None,
                 d_output: Optional[int] = None,
                 parallel: Optional[Union[torch.Tensor, List[int], int]] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        :param d_model: The model width
        :param d_internal: The internel width. By default 2048
        :param d_output: The output widht. If left as none, same as d_model
        :param parallel: The parallel kernels dimensions.
        :param dynamics: The dynamic width. None means off.
        :param dtype: The torch dtype
        :param device: The torch device.
        """
        super().__init__()

        if d_internal is None:
            d_internal = 2048
        if d_output is None:
            d_output = d_model

        self.hyperspatial_projector = Basics.Linear(d_model, d_internal, parallel,
                                                    dtype, device)
        self.restorative_projector = Basics.Linear(d_internal, d_output, parallel,
                                                   dtype, device)

    def forward(self, tensor: torch.Tensor)->torch.Tensor:
        """ Move the item channel out of the way, perform feedforward, move back"""
        tensor = tensor.movedim(-2, 0)
        tensor = self.hyperspatial_projector(tensor)
        tensor = torch.relu(tensor)
        tensor = self.restorative_projector(tensor)
        tensor = tensor.movedim(0, -2)
        return tensor
