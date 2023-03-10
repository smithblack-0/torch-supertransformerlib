"""
A small module containing a custom
padding function for circular padding.
"""


from typing import Optional, List

import torch
from . import errors as Errors
from . import string_util
class PaddingException(Errors.ValidationError):
    """
    An exception to raise when something
    goes wrong when padding
    """
    def __init__(self,
                 reason: str,
                 task: Optional[str] = None,
                 padding: Optional[List[int]] = None,
                 tensor: Optional[torch.Tensor] = None
                 ):
        type = "PaddingException"
        self.padding = padding
        self.tensor = tensor
        super().__init__(type, reason, task)


def _pad_circular(tensor: torch.Tensor, paddings: List[int]):
    """
    Performs circular padding to any dimension and to any
    degree while satisfying the torch padding paradynm.

    Circlular padding creates a new tensor which wraps the
    old tensor around in order to introduce extra dimensions.

    Highly efficient, performing only one new assign.

    :param tensor: The tensor to pad
    :param paddings: The paddings, starting from
        the last dimension, in terms of [start, end, ...]
    :return: A circularly padded tensor.
    """

    paddings = torch.tensor(paddings, dtype=torch.int64)
    pairings = paddings.view(paddings.shape[0]//2, 2)
    ordered_pairings = torch.flip(pairings, dims=[0])
    start_padding, end_padding = ordered_pairings[..., 0], ordered_pairings[..., 1]

    interesting_length = ordered_pairings.shape[0]
    dim_lengths = torch.tensor(tensor.shape[-interesting_length:], dtype=torch.int64)

    # Develop the pattern buffer. This is a contiguous block of tensor
    # memory from which the correct tensor can, in theory, be drawn.
    #
    # It needs to be big enough to handle the starting and ending padding.

    start_repetitions = torch.ceil(start_padding/dim_lengths).to(dtype=torch.int64)*dim_lengths
    end_repetitions = torch.ceil(end_padding/dim_lengths).to(dtype=torch.int64)*dim_lengths
    total_repetitions: List[int] = (start_repetitions + end_repetitions + 1).tolist()

    repetition_instruction = [1]*(tensor.dim() - interesting_length)
    repetition_instruction += total_repetitions

    buffer = tensor.repeat(repetition_instruction)

    # Develop the new dynamic_shape. The dynamic_shape should be a little longer to
    # account for the additional padding introduced
    #
    # Develop the strides. They are the same as the old strides.

    static_shape = torch.tensor(tensor.shape[:-interesting_length], dtype=torch.int64)
    updated_shape = torch.tensor(tensor.shape[-interesting_length:], dtype=torch.int64)
    updated_shape = updated_shape + start_padding + end_padding
    new_shapes: List[int] = torch.concat([static_shape, updated_shape]).tolist()

    current_strides = torch.tensor([buffer.stride(dim) for dim in range(buffer.dim())],
                                   dtype=torch.int64)  # Workaround for bad typing on Tensor.stride.
                                                       # Torchscript wont
                                                       # do tensor.stride()
    new_strides: List[int] = current_strides.tolist()

    # Develop the offset. Figure out the difference between the initial start
    # and required start. Then multiply by the offset

    start_offsets = torch.remainder(-start_padding, dim_lengths)
    start_offsets = start_offsets*current_strides[-interesting_length:]
    new_offset = int(start_offsets.sum())

    # Select the appropriate element out of the buffer

    return buffer.as_strided(new_shapes, new_strides, new_offset)


def pad_circular(tensor: torch.Tensor, paddings: List[int], task: Optional[str] = None):
    """
    Performs circular padding to any dimension and to any
    degree while satisfying the torch padding paradynm.

    Circlular padding creates a new tensor which wraps the
    old tensor around in order to introduce extra dimensions.

    Highly efficient, performing only one new assign.

    :param tensor: The tensor to pad
    :param paddings: The paddings, starting from
        the last dimension, in terms of [start, end, ...]
    :return: A circularly padded tensor.
    """

    if len(paddings) == 0:
        # If the padding length is zero, nothing will happen.
        return tensor

    if len(paddings) % 2 != 0:
        reason = f"""\
        Padding must be implimented such that
        the start and end padding is defined for each 
        relevant dimension. This means the length of 
        'padding' should be even. However, found
        length of {len(paddings)}
        """
        reason = string_util.dedent(reason)
        raise PaddingException(reason, task, paddings, tensor)

    if len(paddings) // 2 > tensor.dim():
        reason = f"""\
        The rank of parameter 'tensor' is {tensor.dim()}. However, it is
        the case that the provided 'paddings' corrosponds to a rank
        of {len(paddings)//2}. The padding rank cannot be greater
        than the tensor rank.
        """
        reason = string_util.dedent(reason)
        raise PaddingException(reason, task, paddings, tensor)

    if torch.tensor(tensor.shape).prod() == 0:
        reason = f"""\
        One of the dimensions of parameter 'tensor' has a 
        length of zero. This will mean there is nothing to 
        pad with. 
        
        This is not allowed when using circular padding. 
        """
        reason = string_util.dedent(reason)
        raise PaddingException(reason, task, paddings, tensor)

    if torch.any(torch.tensor(paddings) < 0):
        reason = f"""\
        One or more of the provided paddings was a negative
        value. Negative padding would remove entries, which
        is not supported. Please ensure all padding
        directives are instead positive or zero.
        
        Provided paddings were {paddings}
        """
        reason = string_util.dedent(reason)
        raise PaddingException(reason, task, paddings, tensor)

    return _pad_circular(tensor, paddings)
