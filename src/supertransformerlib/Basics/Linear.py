"""

Definitions for the version of linear used
throughout the library.

"""

import torch
from torch import nn


def forward(tensor: torch.Tensor,
            kernel: torch.Tensor,
            bias: Optional[torch.Tensor] = None
            )