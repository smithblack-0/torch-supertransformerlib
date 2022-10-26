from typing import List, Optional
import torch
import src.supertransformerlib.Core.Errors as Errors




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


torch.jit.script(calculate_shape_strides)


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
    indices = torch.stack(indices, dim=-1)
    indices = indices[mask]
    indices = indices.transpose(0, 1)
    return indices

torch.jit.script(gen_indices_from_mask)


def convert_dense_to_hybrid(tensor: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
    """
    Converts a tensor into a hybrid sparse tensor by using a mask
    which indicates what to include. It is assumed we are
    working from the first dimension forward.

    For a tensor with shape [5, 3, 7] and a mask of
    shape [5,3], for example, you will end up with a
    sparse tensor with two sparse dimensions and one
    last dense dimension.

    :param tensor: The tensor to convert. Must be a dense tensor
    :param mask: The mask. Must be a bool tensor
    :return: The hybrid tensor to return.
    """

    index = gen_indices_from_mask(mask)
    values = tensor[mask]
    return torch.sparse_coo_tensor(index, values)

torch.jit.script(convert_dense_to_hybrid)

