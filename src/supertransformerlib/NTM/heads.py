"""NTM Read and Write Heads."""
from typing import Tuple, Optional

import torch
from torch import nn
from .. import Core, Basics
from .indexer import Indexer
from .reset import WeightsResetter


class WriteHead(nn.Module):
    """
    The write head for the NTM mechanism, the default kind.

    It contains pretty much the same logic as the read head plus
    the addtional requirements to create the write features.
    """
    def __init__(self,
                 memory_size: int,
                 memory_width: int,
                 control_width: int,
                 shift_kernel_width: int,
                 ensemble_shape: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):

        self.memory_size = memory_size
        self.memory_width = memory_width
        self.control_width = control_width
        self.shift_kernel_width = shift_kernel_width
        self.ensemble_shape = Core.standardize_shape(ensemble_shape, "ensemble_shape")

        super().__init__()

        # Define the reset manager. This is responsible for resetting
        # weights back to default if the model thinks it is a
        # good idea.

        self.weight_resetter = WeightsResetter(control_width,
                                                       memory_size,
                                                       memory_width,
                                                       ensemble_shape,
                                                       dtype = dtype,
                                                       device = device)

        # Define the make write weights manager. This will
        # make the write weights when called upon

        self.make_write_weights = Indexer(memory_size,
                                    memory_width,
                                    control_width,
                                    shift_kernel_width,
                                    ensemble_shape,
                                    dtype=dtype,
                                    device=device)

        # Define the  creators for the erase and memory tensor
        # generators

        self.project_erase_tensor = Basics.Linear(control_width,
                                                 memory_width,
                                                 ensemble_shape)
        self.project_add_tensor = Basics.Linear(control_width,
                                               memory_width,
                                               ensemble_shape)

    def forward(self,
                control_state: torch.Tensor,
                memory: torch.Tensor,
                prior_weights: Optional[torch.Tensor]=None,
                force_reset: Optional[torch.Tensor] = None)->Tuple[torch.Tensor, torch.Tensor]:

        """
        Performs the read operation, and forces a reset on batch
        elements for which the reset mask ends up evaluation to true.

        Otherwise, the model handles everything.

        :param control_state: A ... x control_width tensor used to control actions
        :param memory: A ... x mem_size x mem_width memory tensor
        :param prior_weights: The prior weights from last time, a ... x mem_size probability weights tensor.
                            Notably, this can be left out, in which case it makes a default that fits
                            the batch shape.
        :param force_reset: An optional bool tensor of batch shape. Each element should index a batch.
                            A value of true will mean to reset the weights on that portion to their
                            default before developing the indices.

                            This options use case is with chunked batches. If batches have been
                            chunked and then merged together, one batch may end when another has
                            not. In this case, we want to only reset the batches that have ended.
        :return:
            A ... x mem_size x mem_width tensor of memory
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

        # If a forced reset is active, do it

        if force_reset is not None:
            prior_weights = self.weight_resetter.force_reset(force_reset, )

        # Make the weights, then perform the weighted sum and return

        weights = self.make_read_weights(control_state,
                                         memory,
                                         prior_weights) # should be ... x (ensemble) x mem_size
        add_tensor = self.project_add_tensor(control_state) #... x (ensemble) x mem_width
        erase_tensor = torch.sigmoid(self.project_erase_tensor(control_state)) #... x (ensemble) x mem_width

        # Go update the memory based on the results. Do this by erasing according to
        # the erase head results, and adding according to the writing results
        memory = memory*(1-weights.unsqueeze(-1)*erase_tensor.unsqueeze(-2))
        memory = memory + weights.unsqueeze(-1) * add_tensor.unsqueeze(-2)

        # Return the memory and weights
        return memory, weights

