import torch
from typing import List


def pad_circular(tensor: torch.Tensor, paddings: List[int]):
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
    start_padding, end_padding = ordered_pairings

    interesting_length = ordered_pairings.shape[0]
    dim_lengths = torch.tensor(tensor.shape[-interesting_length:], dtype=torch.int64)

    # Develop the pattern buffer. This is a contiguous block of tensor
    # memory from which the correct tensor can, in theory, be drawn.
    #
    # It needs to be big enough to handle the starting and ending padding.

    start_repetitions = torch.ceil(start_padding/dim_lengths).to(dtype=torch.int64)*dim_lengths
    end_repetitions = torch.ceil(end_padding/dim_lengths).to(dtype=torch.int64)*dim_lengths
    total_repetitions = start_repetitions + end_repetitions + 1

    repetition_instruction = [1]*tensor.dim()
    repetition_instruction[-dim_lengths:] = total_repetitions
    buffer = torch.repeat(tensor, repetition_instruction)

    # Develop the new shape. The shape should be a little longer to
    # account for the additional padding introduced
    #
    # Develop the strides. They are the same as the old strides.

    static_shape = torch.tensor(tensor.shape[:-interesting_length], dtype=torch.int64)
    updated_shape = torch.tensor(tensor.shape[-interesting_length:], dtype=torch.int64)
    updated_shape = updated_shape + start_padding + end_padding
    new_shapes: List[int] = torch.concat([static_shape, updated_shape]).tolist()

    current_strides = torch.tensor([buffer.stride(dim) for dim in range(buffer.dim())],
                                   dtype=torch.int64)  # Workaround for bad typing on Tensor.stride. Torchscript wont
                                                       # do tensor.stride()
    new_strides: List[int] = current_strides.tolist()

    # Develop the offset. Figure out the difference between the initial start
    # and required start. Then multiply by the offset

    start_offsets = torch.remainder(-start_padding, dim_lengths)
    start_offsets = start_offsets*current_strides
    new_offset = int(start_offsets.sum())

    # Select the appropriate element out of the buffer

    return buffer.as_strided(new_shapes, new_strides, new_offset)
