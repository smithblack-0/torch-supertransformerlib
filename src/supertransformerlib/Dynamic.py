"""

A collection of functions designed to
aid in the design of more intelligent
algorithms.

Contains query based batch retrieval
tools, frameworks for ACT, and more.

"""


import torch
from torch import nn


class Batch():
    """
    A feature for retrieving important s
    """

class BatchSampler(nn.Module):
    """
    Accepts as an input a iterator or dataset which is expected to
    yield full_input, label pair. It will take responsibility for breaking
    them up into batches.
    """