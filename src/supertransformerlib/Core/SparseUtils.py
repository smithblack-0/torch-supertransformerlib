from typing import List

import torch


def calculate_shape_strides(shape: List[int])->torch.Tensor:
    """
    Calculate and return the strides associated
    with a particular tensor dynamic_shape assuming
    the strides are defined with the last dim
    having the smallest jump

    :param tensor: The tensor to calculate the strides of
    :return: The calculated strides
    """

    shape = list(shape)
    shape.reverse()

    cumulative_stride = 1
    strides: List[int] = []
    for dim_length in shape:
        strides.insert(0, cumulative_stride)
        cumulative_stride = cumulative_stride * dim_length

    return torch.tensor(strides)


def gen_indices_from_mask(mask: torch.Tensor)->torch.Tensor:
    """
    Generates sparse tensor indices from a mask of a particular dynamic_shape.

    This can be combined with a masked select to quickly create a hybrid
    or sparse tensor

    :param mask: The boolean mask to generate from
    :return: int64 index consisting of [dims, items]
    """

    assert mask.dtype == torch.bool
    indices = torch.meshgrid([torch.arange(i) for i in mask.shape], indexing="ij")
    indices = torch.stack(indices)
    indices = torch.reshape(indices, [mask.dim()] + list(mask.shape))
    indices = indices[..., mask]
    return indices