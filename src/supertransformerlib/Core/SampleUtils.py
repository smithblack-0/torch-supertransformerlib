"""

A module dedicated to handling things
like top-p and top-k in dense and sparse
formats.

Sampling will generally return a mask which would
only select the included elements. This can be
applied to mask out elements, or used to generate
a sparse index.
"""

import torch
from typing import Optional


def random_sample_mask(tensor: torch.Tensor, probability: float, task: Optional[str] = None)->torch.Tensor:
    """
    Accept a tensor. Return a boolean mask
    which samples each element with a random
    random probability.
    """

    shape = torch.Size(tensor.shape)
    rolls = torch.rand(shape)
    mask = rolls > probability
    return mask



def top_k_mask(tensor: torch.Tensor, num: int, task: Optional[str] = None)->torch.Tensor:
    """
    Accepts a tensor. Returns a mask which samples the
    top-k elements along the last dimension. The mask
    is the same shape as the tensor
    """

    # We develop the top-k mask by sorting the indices, giving us a list of numbers that are
    # valid across multiple dimensions. Then, we broadcast this in an equal statement with a
    # list of all index i's, and keep the entries where something in sorted index matches something
    # in index i's

    sorted_index = torch.argsort(tensor, dim=-1, descending=True)
    sorted_index = sorted_index[..., :num]
    numerical_index_reference = torch.arange(tensor.shape[-1], device=tensor.device, dtype= torch.int64)
    mask = sorted_index.unsqueeze(-1) == numerical_index_reference
    mask = torch.any(mask, dim=-2)
    return mask


def top_p_mask(tensor: torch.Tensor, probability_threshold: float)->torch.Tensor:
    """
    Accepts a tensor. Returns a mask which samples the
    top-p elements along the last dimension. Assumes the
    last dimension is a probability which adds up to one.
    """
    # We perform top p. We do this by first sorting the values and their indices in descending order.
    # We also create an output buffer. Then, we perform insertion into an output buffer by
    # the index until we run out of valid entries to insert.
    with torch.no_grad():
        values, sorted_index = torch.sort(tensor, dim=-1, descending=True)
        values = values.movedim(-1, 0)
        sorted_index = sorted_index.movedim(-1, 0)

        mask = torch.full_like(tensor, False, dtype=torch.bool)
        total_p = torch.zeros(tensor.shape[:-1], dtype = tensor.dtype, device = tensor.device)
        for level_values, level_index in zip(values, sorted_index):

            thresholdmask = total_p <= probability_threshold
            update = mask.scatter(-1, level_index.unsqueeze(-1), True)
            mask = torch.where(thresholdmask.unsqueeze(-1), update, mask)

            total_p += level_values
            if torch.all(torch.logical_not(thresholdmask)):
                break
        return mask


