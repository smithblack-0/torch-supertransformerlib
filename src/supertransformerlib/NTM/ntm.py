from typing import Optional, Tuple, List

import torch
from torch import nn

from supertransformerlib import Basics
from supertransformerlib import Core
from supertransformerlib.NTM.indexer import Indexer
from supertransformerlib.NTM.reset import WeightsResetter


class MemDefaults(nn.Module):
    """
    A NTM extension layer designed to contain within it the default state for
    the memory units across the ensemble, along with any and all logic which
    could demand interaction with such a entity.
    """

    def __init__(self,
                 memory_size: int,
                 memory_width: int,
                 ensemble_shape: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        Initialize the memory defaults, by creating a parameter which possesses
        the memory size and width and including any ensemble directives

        :param memory_size: The number of memory elements
        :param memory_width: The width of the memory embeddings
        :param ensemble_shape: The shape of any ensemble. Can be int, list[int], or 1d tensor
        :param dtype: The dtype
        :param device: The device
        """

        if ensemble_shape is not None:
            shape: List[int] = Core.standardize_shape(ensemble_shape).tolist()
            shape = shape + [memory_size, memory_width]
        else:
            shape = [memory_size, memory_width]

        memory_default = torch.zeros(shape, dtype=dtype, device=device)
        torch.nn.init.kaiming_uniform_(memory_default)
        self.memory_default = memory_default

    @torch.jit.export
    def make_memory(self,
                    batch_shape: Core.StandardShapeType)->torch.Tensor:
        """
        Makes a memory unit compatible with the batch by broadcasting across
        the batch shape.

        :param batch_shape: The shape of the batch, quite literally. Can be a int, a list of ints, or a 1d
                            tensor
        :return: A memory tensor of shape batch_shape... x (...ensemble) x memory_size x memory_width
        """
        memory_default = self.memory_default
        shape: List[int] = Core.standardize_shape(batch_shape).tolist()
        batch_len = len(shape)
        expand_shape = shape + [-1]*self.memory_default.dim()
        for _ in range(batch_len):
            memory_default = memory_default.unsqueeze(0)
        memory = memory_default.expand(expand_shape)
        return memory
    @torch.jit.export
    def reset_memory(self,
                     memory: torch.Tensor,
                     reset_probabilities: torch.Tensor) -> torch.Tensor:
        """
        A section for resetting memories to their default values. This uses the
        reset probabilities tensor. This is done as an extrapolation between their
        current and reset values

        :param memory: A memory tensor, of shape
                        (...batch_shape) x (ensemble_shape...) x memory_size x memory_width.

                        It is literally the memories
        :param reset_probabilities: The reset probabilities, which is a float tensor of shape
                        (...batch_shape) x (ensemble_shape...) x memory_size with values
                        between 0 and 1. 0 indicates do not reset, 1 indicates completely
                        reset
        :return: The memory tensor, with the indicated elements reset
        """
        reset_probabilities = reset_probabilities.unsqueeze(-1)
        reset_values = self.memory_default.expand_as(memory)
        updated_memory = memory * (1 - reset_probabilities) + reset_values * reset_probabilities
        return updated_memory

class MemManager(nn.Module):
    """
    A layer to create empty memory and/or
    read/write parameter blocks, and manage
    resetting them when appropriate for new batches.

    * Make memory block in the first place
    * Make weight blocks in the first place
    * Reset entire memory to default values
    * Reset only weight channels when appropriate to default values
    *

    The layer will accept a control state alongside
    """

class Resetter(nn.Module):
    """
    A layer designed to reset a NTM memory unit and the
    associated read and write parameters back to
    default values, or even make it in the first
    place.
    """

class Reader(nn.Module):
    """
    A collection of read heads designed to fetch information out of a
    NTM datastructure. This is performed by using the prior read
    weights, the control state, and the current memory.

    Multiple read heads are contained within the structure and all
    work alongside the input to handle the particular problem being
    worked upon.

    The output of the reader will be the read result and the
    updated reader_weights
    """

    def __init__(self,
                 memory_size: int,
                 memory_width: int,
                 num_heads: int,
                 control_width: int,
                 shift_kernel_width: int,
                 head_creation_mode: Optional[str] = "project",
                 head_merge_mode: Optional[str] = "weighted_sum",
                 ensemble_shape: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,


                 ):
        """

        :param memory_size: The size of the auxilary memory in terms of memory units
        :param memory_width: The width of the memory embeddings
        :param num_heads: The number of heads to make
        :param control_width: The width of the control tensor embedding
        :param shift_kernel_width: How wide to make the shift kernel. Notably,
                                 shifts are defined to be symmetric around 0
        :param ensemble_shape: The shape of any ensemble action which is occuring
        :param head_creation_mode: The control tensor can have heads placed on it by two
                                    means. We could either do a linear projection, or a reshape
                                    action. This decides which to use. The two modes are
                                    "reshape" or "project".
        :param head_merge_mode: The read tensors will consist of a weighted sum of embeddings with
                                one per head. These have to be merged together before the results
                                are returned. Two options exist for this. They are "project" which
                                uses a linear projection, "weight" which makes a weighted
                                sum between each head, and "sum" which simply sums across the
                                appropriate dimension. These each require progressively decreasing
                                numbers of parameters, and the default is weight
        :param dtype: The dtype of the kernels
        :param device: The device of the kernels
        """

        # Store natural parameters.

        self.memory_size = memory_size
        self.memory_width = memory_width
        self.num_heads = num_heads
        self.control_width = control_width
        self.shift_kernel_width = shift_kernel_width
        self.ensemble_shape = ensemble_shape
        self.head_creation_mode = head_creation_mode

        super().__init__()

        # We need to make the layers responsible for
        # creating and merging heads. Heads will need to
        # be created on the control stte tensor, and when
        # merging we will be merging the memory tensor stack

        self.create_heads = Basics.MakeHead(control_width, num_heads,
                                            mode=head_creation_mode,
                                            parallel=ensemble_shape,
                                            dtype=dtype,
                                            device=device)
        self.merge_heads = Basics.ReductiveMergeHeads(memory_width,
                                                      num_heads,
                                                      mode=head_merge_mode,
                                                      parallel=ensemble_shape,
                                                      dtype=dtype,
                                                      device=device)

        # Define the NTM indexing mechanism. Notice how we are expecting to
        # recieve control tensors of head width, not control width. Also, note
        # that the ensemble content must have the head width added to it to
        # create the proper parallel kernels. This ensures the index generation
        # is entirely indepedent between heads: There is no reuse of parameters as in
        # a convolution.

        if ensemble_shape is not None:
            ensemble_shape = Core.standardize_shape(ensemble_shape, "ensemble_shape")
            ensemble_shape = torch.concat([ensemble_shape, torch.tensor([num_heads])], dim=-1)
        else:
            ensemble_shape = torch.tensor([num_heads])

        self.make_read_weights = Indexer(
                                    memory_size,
                                    memory_width,
                                    self.create_heads.head_width,
                                    shift_kernel_width,
                                    ensemble_shape,
                                    dtype=dtype,
                                    device=device)


    def forward(self,
                control_state: torch.Tensor,
                memory: torch.Tensor,
                read_weights: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:

        """
        Performs the read operation

        :param control_state: A ... x control_width tensor used to control actions
        :param memory: A ... x mem_size x mem_width memory tensor
        :param read_weights: The prior read weights from last time, a ... x heads x mem_size probability weights tensor.
                            Notably, this can be left out, in which case it makes a default that fits
                            the batch shape.
        :return:
            A ...  x mem_width tensor of memory read output
            A ... x heads x mem_size probability weights tensor indicating the current probability results.
        """

        # Create head states on the control state and memory tensors
        # Note that we have to add a virtual items dimension onto the
        # control tensor and then remove it

        control_state = self.create_heads(control_state.unsqueeze(-2)).squeeze(-2) #
        memory = memory.unsqueeze(-3)

        # Make the new read weights
        weights = self.make_read_weights(control_state,
                                         memory,
                                         read_weights
                                         )
        # Do the read, merge the heads
        headed_output = torch.matmul(weights.unsqueeze(-2), memory)
        output = self.merge_heads(headed_output).squeeze(-2)

        # return
        return output, weights

class Writer(nn.Module):
    """
    A collection of write heads designed to insert information back
    into an NTM memory datastructure, while recursively keeping
    in mind the prior actions.
    NTM datastructure. This is performed by using the prior read
    weights, the control state, and the current memory.

    Multiple read heads are contained within the structure and all
    work alongside the input to handle the particular problem being
    worked upon.

    The output of the reader will be the read result and the
    updated reader_weights
    """