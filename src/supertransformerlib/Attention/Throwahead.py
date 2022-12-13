"""
A module for the throwahead implimentation to reside.

Throwahead uses the throwahead queue to predict, then
throw, information into future timesteps. It also
dequeues information which might be useful for this layer.

"""

import torch
from torch import nn
from typing import Optional
from src.supertransformerlib import Core
from src.supertransformerlib import Structures


class ThrowaheadException(Core.ValidationError):
    def __init__(self, reason: str):
        type = "ThrowaheadException"
        super().__init__(type, reason)


class _QueueFactory(nn.Module):
    """
    A small helper class. It will, when
    provided with the batch shape,
    go ahead and make a functional
    throwahead queue for further usage.

    It produces queues, not virtual layers.
    """
    def __init__(self,
                 queue_length: int,
                 element_shape: torch.Tensor,
                 parallel: Optional[torch.Tensor],
                 dtype: Optional[torch.dtype],
                 device: Optional[torch.device]):

        super().__init__()

        if parallel is not None:
            partial_shape = torch.concat([parallel, element_shape], dim=-1)
        else:
            partial_shape = element_shape

        self.queue_length = queue_length
        self.partial = partial_shape
        self.dtype = dtype
        self.device = device

    def forward(self, batch_shape: torch.Tensor)->Structures.ThrowaheadQueue:
        total_shape = torch.concat([batch_shape, self.partial], dim=-1)
        ThrowaheadQueue = Structures.ThrowaheadQueue(self.queue_length,
                                                     total_shape,
                                                     self.dtype,
                                                     self.device)
        return ThrowaheadQueue


class _Throwahead:
    """
    The implementation of throwahead. It performs
    the enqueuing and dequeuing actions and interacts
    with the data structure.
    """

    def __init__(self,
                 queue: Structures.ThrowaheadQueue,
                 ):
        self.queue = queue

    def __call__(self, tensor: torch.Tensor, enqueue_weights: torch.Tensor)->torch.Tensor:
        self.queue.enqueue(enqueue_weights, tensor)
        output = self.queue.dequeue()
        return output

torch.jit.script(_Throwahead)

class ThrowaheadFactory(nn.Module):
    """
    The factory class for this technique. It consists of storing based on
    the throwahead probability a particular tensor, which will in turn pop
    out N calls in the future based on the probabilities.

    It is expected that this will be used per batch, meaning
    a factory class is required. Throwahead was developed under the
    assumption skip connections are useful.
    """
    def __init__(self,
                 queue_size: int,
                 elements_shape: Core.StandardShapeType,
                 parallel: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None):
        """

        :param queue_size: The size of the throwahead queue. Larger size requires more space, but is
               more effective at forwarding information to future layers.
        :param elements_shape: The shape of the elements dimension. For example, [64] for embedding or
               [256, 256] for an image.
        :param parallel: The number of parallel ensembles. For example, 4 or [3, 4]
        :param dtype:
        :param device:
        """

        super().__init__()

        elements_shape = Core.standardize_shape(elements_shape, "elements_shape")

        if parallel is not None:
            parallel = Core.standardize_shape(parallel, "parallel")

        self.queueFactory = _QueueFactory(queue_size,
                                          elements_shape,
                                          parallel,
                                          dtype,
                                          device)


    def forward(self, batch_shape: Core.StandardShapeType)->_Throwahead:
        """
        :param batch_shape: The shape of the elements which should be handled independently.
        """
        batch_shape = Core.standardize_shape(batch_shape, "batch_shape")
        queue = self.queueFactory(batch_shape)
        return _Throwahead(queue)

