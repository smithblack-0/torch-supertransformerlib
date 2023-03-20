"""LSTM Controller."""
import torch
import supertransformerlib
from torch import nn
from torch.nn import Parameter
from typing import Optional
import numpy as np

class FeedforwardController(nn.Module):
    """
    A NTM controller based on a feedforward
    layer, capable of being used in a parallel
    ensemble.
    """
    def __init__(self,
                 num_inputs: int,
                 num_hidden: int,
                 num_outputs: int,
                 ensemble_shape: Optional[supertransformerlib.Core.StandardShapeType]
                 ):

        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.feedforward = supertransformerlib.Basics.Feedforward(
            num_inputs,
            num_hidden,
            num_outputs,
            ensemble_shape
        )
    def forward(self, tensor: torch.Tensor)->torch.Tensor:
        return self.feedforward(tensor)

