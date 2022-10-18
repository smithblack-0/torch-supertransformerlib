"""

The interconnections module

Responsible for holding the neural configuration and available connective
bindings for a particular situation.


"""

import torch
from torch import nn



class ConvolutionalKernel(nn.Module):
    """
    A particular ConvolutionalKernel.

    A convolutional kernel is a specification of connections
    of the current element to nearby elements which tells
    us how they are connected, in terms of a dilation,
    start, and end for each dimension. This layer keeps to this
    trend.

    """