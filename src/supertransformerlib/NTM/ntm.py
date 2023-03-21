from typing import Optional, Tuple

import torch
from torch import nn

from supertransformerlib import Basics
from supertransformerlib import Core
from supertransformerlib.NTM.indexer import Indexer
from supertransformerlib.NTM.reset import WeightsResetter

class NTMBase(nn.Module):
    """
    The base class for an NTM reader/writer/resetter, it
    contains the logic needed in order to create
    head handling layers.
    """
    def make_head_creator(self,
                          input_width: int,
                          num_heads: int,
                          creation_mode: str
                          )->nn.Module:
        task = "making head generator"
        if creation_mode == "reshape":
            if input_width % num_heads != 0:
                msg = f"""\
                The head width is the control width divided by the number of heads.

                It is the case the current head creation specification is not allowed.
                The mode is 'reshape' and we are asking for a number of heads equal
                to {num_heads}. However, this does not divide cleanly into {input_width}
                """
                msg = Core.dedent(msg)
                raise Core.Errors.ValidationError("HeadWidthProblem", msg, task)
            head_width = input_width // num_heads
            return Core.Reshape(input_width, [num_heads, head_width])
        elif creation_mode == "project":
            head_width = input_width // num_heads
            if head_width < 1:
                msg = f"""\
                The head width is the control width divided by the number of heads.

                It is the case the current head width, after completing this process,
                is less than one which is not allowed             
                """
                msg = Core.dedent(msg)
                raise Core.Errors.ValidationError("HeadWidthProblem", msg, task)
        else:
            msg = f"""\
            creation mode was not among "reshape" or "project"
            """
            msg = Core.dedent(msg)
            raise ValueError(msg)
    def make_head_collapser



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
                 ensemble_shape: Optional[Core.StandardShapeType] = None,
                 head_creation_mode: Optional[str] = "project",
                 head_merge_mode: Optional[str] =
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

        # We need to make a layer responsible for reshaping control state
        # to possess the same number of heads as the weights tensor. This
        # is accomplished here by creating either a reshape or linear
        # projection layer.



            self.create_heads = Basics.Linear(control_width, [num_heads, head_width], ensemble_shape)

        # Merging the results back together has a few options associated with it.

        # Define the indexer used to make the read weights, and other
        # related interaction mechanisms

        self.weight_resetter = WeightsResetter(control_width,
                                                       memory_size,
                                                       memory_width,
                                                       ensemble_shape,
                                                       dtype = dtype,
                                                       device = device)
        self.make_read_weights = Indexer(memory_size,
                                    memory_width,
                                    control_width,
                                    shift_kernel_width,
                                    ensemble_shape,
                                    dtype=dtype,
                                    device=device)


    def forward(self,
                control_state: torch.Tensor,
                memory: torch.Tensor,
                prior_weights: Optional[torch.Tensor]=None)->Tuple[torch.Tensor, torch.Tensor]:

        """
        Performs the read operation

        :param control_state: A ... x control_width tensor used to control actions
        :param memory: A ... x mem_size x mem_width memory tensor
        :param prior_weights: The prior weights from last time, a ... x mem_size probability weights tensor.
                            Notably, this can be left out, in which case it makes a default that fits
                            the batch shape.
        :return:
            A ...  x mem_width tensor of memory read output
            A ... x mem_size probability weights tensor indicating the current probability results.
        """


        if prior_weights is None:
            # Make a default prior weights tensor if it does not exists.
            batch_dims = control_state.dim() - 1 - self.ensemble_shape.shape[-1]
            batch_shape = control_state.shape[:batch_dims]
            prior_weights = self.weight_resetter.setup_new_weights(batch_shape)
        else:
            # Reset the prior weights if we think it might help
            prior_weights = self.weight_resetter(control_state, prior_weights)

        # Make the weights, then perform the weighted sum and return

        weights = self.make_read_weights(control_state,
                                         memory,
                                         prior_weights) # should be ... x (ensemble) x mem_size
        output = torch.matmul(weights.unsqueeze(-1), memory)
        return output, weights
