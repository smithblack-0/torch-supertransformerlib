"""

A place for a few useful functions to end up

"""

import torch

def expand_along_dim(tensor: torch.Tensor, dim: int, length: int) -> torch.Tensor:
    """
    Unsqueezes, then expands a tensor along the indicated dimension
    the indicated number of times. Does this using a memory efficient
    view.

    :param tensor: the tensor to expand
    :param dim: The dimension to unsqueeze
    :param length: How much to repeat.
    :return: A tensor
    """

    shape = [-1] * tensor.dim()
    tensor = tensor.unsqueeze(dim)
    shape.insert(dim, length)
    tensor = tensor.expand(shape)
    return tensor

def permute_elements(tensor: torch.Tensor, permuter: torch.Tensor, dimension: int)->torch.Tensor:
    """
    Permutes elements within some sort of tensor, based on
    a given permuter. The permuter might be the result of running
    some sort of sort, or something else interesting, but
    should consist of integers representing where along

    :param tensor: The tensor to permute elements on.
    :param permuter: The tensor representing a permutation. Should consist of indices, each different
    :param dimension: The dimension to apply the permutation to.
    :return: The permuted tensor.
    """





