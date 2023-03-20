import torch
from torch import nn
from .. import Basics
from .. import Core
from typing import Tuple, Optional, List


class MemoryResetter(nn.Module):
    """
      A class designed to reset memory and weights for a Neural Turing Machine

      It accepts a control_state tensor and a memory tensor. It then proceeds
      to reset the memory to default values if the learnable reset probability
      is high. It does this as an interpolation between current memory and
      default memory
      """

    def __init__(self,
                 control_width: int,
                 memory_size: int,
                 memory_width: int,
                 ensemble_shape: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the MemoryResetter with the required parameters.

        :param control_width: The width of the control tensor embeddings
        :param memory_size: The number of memory elements to setup
        :param memory_width: The width of the memory embeddings
        :param ensemble_shape: Any ensemble shaping.
        """
        super().__init__()

        self.control_width = control_width
        self.memory_size = memory_size
        self.memory_width = memory_width

        # Set up the reset projector

        self.control_projector = Basics.Linear(control_width,
                                               1,
                                               ensemble_shape)
        # Set up the default parameters. Include the extra dimensions for kernels
        # Jump through a few extra hoops to satisfy torchscript.

        memory_shape = [memory_size, memory_width]

        if ensemble_shape is not None:
            ensemble_shape = Core.standardize_shape(ensemble_shape,
                                                                        "ensemble_shape").tolist()
            memory_shape = ensemble_shape + memory_shape

        memory_default = torch.zeros(memory_shape, dtype=dtype, device=device)
        torch.nn.init.xavier_normal_(memory_default)
        self.memory_default = nn.Parameter(memory_default)

    @torch.jit.export
    def force_reset(self,
                    batch_mask: torch.Tensor,
                    memories: torch.Tensor
                    )->torch.Tensor:
        """
        Forces a reset back to default for mask elements marked
        true. This is designed to be utilized to reset batches
        whose subsections are out of sync, such that one batch
        may end while another one is still running.

        :param batch_mask: A bool tensor shaped like the batch dimension
                        with true indicating a reset is demanded.
        :param memories: The current memories.
        :return: A group of memory units that has been reset where so indicated.
        """
        batch_dims = batch_mask.dim()
        other_dims = memories.dim() - batch_dims
        default_memories = self.memory_default
        for _ in range(batch_dims):
            default_memories = default_memories.unsqueeze(0)
        for _ in range(other_dims):
            batch_mask = batch_mask.unsqueeze(-1)

        memory = torch.where(batch_mask, default_memories, memories)
        return memory


    @torch.jit.export
    def setup_new_memory(self,batch_shape: Core.StandardShapeType)->torch.Tensor:
        """
        Creates a new default memory tensor by expanding the default to match the
        batch shape.

        :param batch_shape: The shape of the batch to set this up for
        :return: A ... x x memory_size x memory_width tensor with possible ensemble dimensions after
                the batch dimensions
        """

        batch_shape_tensor = Core.standardize_shape(batch_shape, "batch_shape")
        batch_shape: List[int] = batch_shape_tensor.tolist() # Must explictly annotate as list for jit.
        expansion_shape = batch_shape + [-1]*self.memory_default.dim()
        default_memory = self.memory_default
        for _ in range(len(batch_shape)):
            default_memory = default_memory.unsqueeze(0)
        default_memory = default_memory.expand(expansion_shape)
        return default_memory

    def forward(self,
                 control_state: torch.Tensor,
                 memory: torch.Tensor,
                 )->torch.Tensor:
        """
        Resets the memory to it's default configuration if the learnable
        reset projection indicates to do so.

        :param control_state: A ... x control_width tensor which is projected to decide whether
                             to reset or not
        :param memory: The prior memory tensor, of shape ... x mem_num x embedding
        :return: A interpolated memory tensor of shape ... x mem_num x mem_embedding
        """

        reset_probability = torch.sigmoid(self.control_projector(control_state))
        batch_dims = memory.dim() - self.memory_default.dim()
        default_memories = self.memory_default
        for _ in range(batch_dims):
            default_memories = default_memories.unsqueeze(0)
        memory = memory*(1-reset_probability.unsqueeze(-1)) + default_memories*reset_probability.unsqueeze(-1)
        return memory


class WeightsResetter(nn.Module):
    """
    A class designed to reset the read weights for a Neural Turning Machine.

    The class will accept a control state, weights
    tensor and create a reset probability, which it
    will then use to extrapolate between the default
    state and the current state.
    """

    @torch.jit.export
    def setup_new_weights(self, batch_shape: Core.StandardShapeType) -> torch.Tensor:
        """
        Creates a new default weights tensor by expanding the default to match the
        batch shape.

        :param batch_shape: The shape of the batch to set this up for
        :return: A ... x x memory_size x memory_width tensor with possible ensemble dimensions after
                the batch dimensions
        """
        batch_shape_tensor = Core.standardize_shape(batch_shape, "batch_shape")
        batch_shape: List[int] = batch_shape_tensor.tolist() # Must explictly annotate as list for jit
        expansion_shape = batch_shape + [-1] * self.weights_default.dim()
        default_weights = torch.softmax(self.weights_default, dim=-1)
        for _ in range(len(batch_shape)):
            default_weights = default_weights.unsqueeze(0)
        default_weights = default_weights.expand(expansion_shape)
        return default_weights

    @torch.jit.export
    def force_reset(self,
                    batch_mask: torch.Tensor,
                    weights: torch.Tensor)->torch.Tensor:
        """
        Forces a reset back to default for mask elements marked
        true. This is designed to be utilized to reset batches
        whose subsections are out of sync, such that one batch
        may end while another one is still running.

        :param batch_mask: A bool tensor shaped like the batch dimension
                        with true indicating a reset is demanded.
        :param weights: The current weights.
        :return: A group of weights that has been reset where so indicated.
        """

        batch_dims = batch_mask.dim()
        other_dims = weights.dim() - batch_dims
        default_weights = torch.softmax(self.weights_default, dim=-1)
        for _ in range(batch_dims):
            default_weights = default_weights.unsqueeze(0)
        for _ in range(other_dims):
            batch_mask = batch_mask.unsqueeze(-1)
        weights = torch.where(batch_mask, default_weights, weights)
        return weights

    def __init__(self,
                 control_width: int,
                 memory_size: int,
                 memory_width: int,
                 ensemble_shape: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the MemoryResetter with the required parameters.

        :param control_width: The width of the control tensor embeddings
        :param memory_size: The number of memory elements to setup
        :param memory_width: The width of the memory embeddings
        :param ensemble_shape: Any ensemble shaping.
        """
        super().__init__()

        self.control_width = control_width
        self.memory_size = memory_size
        self.memory_width = memory_width

        # Set up the reset projector

        self.control_projector = Basics.Linear(control_width,
                                               1,
                                               ensemble_shape)
        # Set up the default parameters. Include the extra dimensions for kernels
        # Jump through a few extra hoops to satisfy torchscript.

        weights_shape = [memory_size]

        if ensemble_shape is not None:
            ensemble_shape = Core.standardize_shape(ensemble_shape,
                                                    "ensemble_shape").tolist()
            weights_shape = ensemble_shape + weights_shape

        weights_default = torch.zeros(weights_shape, dtype=dtype, device=device)
        torch.nn.init.normal_(weights_default)
        self.weights_default = nn.Parameter(weights_default)

    def forward(self,
                 control_state: torch.Tensor,
                 weights: torch.Tensor,
                 )->torch.Tensor:
        """
        Resets the weights to their default configuration if the learnable
        reset projection indicates to do so. Do so using a extrapolation
        based on a probability.

        :param control_state: A ... x control_width tensor used to get the reset probability
        :param weights: The prior weights tensor, of shape ... x mem_num
        :return: A interpolated ... x mem_num weights tensor.
        """
        reset_probability = torch.sigmoid(self.control_projector(control_state))
        batch_dims = weights.dim() - self.weights_default.dim()
        default_weights = torch.softmax(self.weights_default, dim=-1)
        for _ in range(batch_dims):
            default_weights = default_weights.unsqueeze(0)
        weights = weights*(1-reset_probability) + default_weights*reset_probability
        return weights