from typing import Optional, List

import torch
import uuid
from torch import nn
from supertransformerlib import Core


class DefaultParameterLayer(nn.Module):
    """
    A NTM extension layer designed to contain within it the default
    state for some sort of parameter and to be manipulatable to create,
    interpolate, and reset batch elements to as fine a granularity as is provided

    It also contains a unique id which identifies what parameter id
    it is corrolated with.
    """
    def __init__(self,
                 parameter: nn.Parameter
                 ):
        super().__init__()
        self.ident = str(uuid.uuid1())
        self.default_parameter = parameter

    @torch.jit.export
    def make_batch(self,
                   batch_shape: Core.StandardShapeType
                   ):
        """
        :param batch_shape: The shape of the batch, in terms of an int, a list of ints, or a 1d tensor
        :return: A batch consisting of a broadcasted defaults
        """
        broadcast_shape: List[int] = Core.standardize_shape(batch_shape, "batch_shape").tolist()
        expansion_length = len(broadcast_shape)
        broadcast_shape += [-1] * self.default_parameter.dim()
        defaults = self.default_parameter
        for _ in range(expansion_length):
            defaults = defaults.unsqueeze(0)
        tensor = defaults.expand(broadcast_shape)
        return tensor

    @torch.jit.export
    def reset_to_parameters(self,
                            reset_probability: torch.Tensor,
                            tensor: torch.Tensor) -> torch.Tensor:
        """
        A small helper method, this will accept a fully expanded tensor and
        it's unbroadcasted defaults, then perform linear interpolation between them using the
        reset probabilities. A value of 0 will mean do not reset, while 1
        means completely reset

        :param reset_probability: A float tensor of values between 0..1. The rank of this tensor can
                                  only be greater than or equal to the rank of parameter 'tensor', and
                                  the dimensions here must match the initial dimensions of 'tensor'
        :param tensor:          A data tensor which we wish to interpolate with.
        :return: An interpolated tensor between the tensor and the defaults, mediated by the reset probability
        """
        defaults = self.default_parameter
        reset_values = defaults.expand_as(tensor)
        while reset_probability.dim() < reset_values.dim():
            reset_probability = reset_probability.unsqueeze(-1)
        updated_tensor = tensor * (1 - reset_probability) + reset_values * reset_probability
        return updated_tensor

    @torch.jit.export
    def force_reset_to_defaults(self,
                                reset_mask: torch.Tensor,
                                tensor: torch.Tensor)->torch.Tensor:
        """
        Forces a reset to default where the reset mask is marked as true

        :param reset_mask: A mask which matches tensor's dimensions on the initial dimensions. Elements
                            marked true will be reset to defaults
        :param tensor: The tensor to reset
        :return: A tensor which has had elements replaced with the mask where appropriate
        """
        defaults = self.default_parameter
        reset_values = defaults.expand_as(tensor)
        while reset_mask.dim() < reset_values.dim():
            reset_mask = reset_mask.unsqueeze(-1)
        updated_tensor = torch.where(reset_mask, reset_values, tensor)
        return updated_tensor


def make_memory_parameter(
        memory_size: int,
        memory_width: int,
        ensemble_shape: Optional[Core.StandardShapeType] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
        )->DefaultParameterLayer:

    """
    Creates a functional DefaultParameterLayer for representing a memory
    parameter, which is capable of handling resetting to defaults.
    """

    shape = [memory_size, memory_width]
    if ensemble_shape is not None:
        ensemble_shape_list: List[int] = Core.standardize_shape(ensemble_shape, "ensemble_shape").tolist()
        shape = ensemble_shape_list + shape

    parameter = torch.zeros(shape, dtype = dtype, device=device)
    torch.nn.init.kaiming_uniform_(parameter)
    parameter = nn.Parameter(parameter)
    return DefaultParameterLayer(parameter)

def make_weights_parameter(memory_size: int,
                           num_heads: int,
                           ensemble_shape: Optional[Core.StandardShapeType] = None,
                           dtype: Optional[torch.dtype] = None,
                           device: Optional[torch.device] = None
                           ) -> DefaultParameterLayer:
    """

    Creates a functional weights layer to contain the default weights
    values and to be responsible for resetting the weights.

    :param memory_size: The size of the built memory
    :param num_heads: The number of heads the memory will manage
    :param ensemble_shape: The shape of the ensemble, if used
    :param dtype: The dtype
    :param device: The device.
    :return:
    """
    shape = [num_heads, memory_size]
    if ensemble_shape is not None:
        ensemble_shape_list: List[int] = Core.standardize_shape(ensemble_shape, "ensemble_shape").tolist()
        shape = ensemble_shape_list + shape

    parameter = torch.zeros(shape, dtype = dtype, device=device)
    torch.nn.init.kaiming_uniform_(parameter)
    parameter = nn.Parameter(parameter)
    return DefaultParameterLayer(parameter)