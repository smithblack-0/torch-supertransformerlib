"""

A module containing the logic needed to impliment
a working throwahead queue.

The throwaway queue is designed to work more efficiently with
the unique challenges of machine learning than something like
residual bypass.

"""

import torch
from typing import List, Optional
from src.supertransformerlib import Core



class ThrowaheadQueue:
    """
    The throwahead queue is designed to allow for
    directly placing values N units in advance. Dequeuing
    advances the entire queue as normal. Enqueuing, meanwhile,
    involves providing enqueue weights for each slot in the
    queue. The tensor you are enqueuing is multiplied by the
    the weights and added to the appropriate portion of the queue.

    Broadcasting is supported, and is defined from left to right. This
    means that if you defined a queue of length 10 and shape [4, 5],
    you could update it using a weights tensor of shape [10, 4] and
    a content tensor of shape [4, 5]
    """

    def dequeue(self)->torch.Tensor:
        # Get the next entry, then advanced the queue
        # and put a fresh page on the end.

        output = self.queue[0]
        new_page = torch.zeros_like(output)
        self.queue = torch.roll(self.queue, -1, 0)
        self.queue[-1] = new_page
        return output

    def enqueue(self, throwahead_weights: torch.Tensor, tensor: torch.Tensor):
        # Goes ahead and enqueues a particular tensor at the appropriate probabilities
        while throwahead_weights.dim() < self.queue.dim():
            throwahead_weights = throwahead_weights.unsqueeze(-1)
        update = throwahead_weights*tensor
        self.queue = self.queue + update

    def __init__(self,
                 length: int,
                 shape: Core.StandardShapeType,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None):

        shape = Core.standardize_shape(shape, "shape")
        shape_as_list: List[int] = shape.tolist() # Required by torchscript
        shape_as_list = [length] + shape_as_list
        self.queue = torch.zeros(shape_as_list, dtype=dtype, device=device)
