"""

A place for the midbrain resource access
system to be defined.

"""

import torch
import Adaptive
from torch import nn





class AttentionResource:
    """
    A mechanism usable for looking up information
    from some sort of backend.
    """
    #
    #
    #
    def __init__(self,
                 content: torch.Tensor,
                 query_project_kernel: torch.Tensor,
                 key_project_kernel: torch.Tensor,
                 value_project_kernel: torch.Tensor,
                 dehead_projector: torch.Tensor,
                 ):
        self.content = content
        self.query_project_kernel = query_project_kernel
    def __call__