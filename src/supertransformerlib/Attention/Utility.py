from typing import Optional, List, Union

import torch
from torch import nn
from src.supertransformerlib import Basics, Core

def score_based_on_query_key(query: torch.Tensor, key: torch.Tensor)->torch.Tensor:
    score = torch.matmul(query, key.transpose(-1, -2))
    return score

def perform_attention(scores: torch.Tensor, values: torch.Tensor, d_key: int)->torch.Tensor:
    """ Perfors the attention using the scores."""
    attn = torch.matmul(scores, values)
    attn = attn / torch.sqrt(torch.tensor([d_key], device=scores.device))
    return attn
def dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        )->torch.Tensor:
    """
    Performs dot product attention, as
    shown in "attention is all you need"

    :param query: The query.
    :param key: The key.
    :param value: The value
    :param mask: Any mask to include.
    :return:
    """
    d_key = key.shape[-1]
    score = score_based_on_query_key(query, key)
    if mask is not None:
        score = score.masked_fill(mask, -1e+8)
    score = torch.softmax(score, dim=-1)
    attn = perform_attention(score, value, d_key)
    return attn

def normalize_positive_definite(tensor: torch.Tensor)-> torch.Tensor:
    """
    Normalizes positive definite problems to add up to one.
    """
    numeric_safety_constant = 1e-8 # To prevent divide by zero
    return tensor  / (torch.linalg.vector_norm(tensor, ord=1) + numeric_safety_constant)

def double_action_relu_dot_product_attn(query: torch.Tensor,
                                        key: torch.Tensor,
                                        positive_values: torch.Tensor,
                                        negative_values: torch.Tensor)->torch.Tensor:
    """

    :param query: The query to work with
    :param key: The key to associate with the values
    :param positive_values: The values to pull from when keys are positive
    :param negative_values: The values to pull from when keys are negative.
    :return:
    """
    d_key = key.shape[-1]
    score = torch.matmul(query, key.transpose(-1, -2))

    positive_score = normalize_positive_definite(torch.relu(score))
    positive_attn = perform_attention(positive_score, positive_values, d_key)

    negative_score = -normalize_positive_definite(torch.relu(-score))
    negative_attn = perform_attention(negative_score, negative_values, d_key)

    output = positive_attn + negative_attn
    return output



def commitment_attn_scoring(query: torch.Tensor,
                           key: torch.Tensor,
                           mode: str = "none")->torch.Tensor:
    """
    Performs a special flavor of attn to get commitment scores for each
    output attn channel. A commitment score is a scalar logit which is
    roughly positive, though could be negative if leaky relu mode is on.

    modes are "sigmoid", "relu", and "none". These go off before summing:
    """
    score = torch.matmul(query, key.transpose(-1, -2))
    if mode == "sigmoid":
        score = torch.sigmoid(score)
    elif mode == "relu":
        score = torch.relu(score)
    elif mode == "none":
        pass
    else:
        raise ValueError("Invalid mode")
    score = score.sum(dim=-1)
    return score

class MakeHead(nn.Module):
    """
    A helper layer. This will go
    and make a head usable for an
    attention mechanism.
    """
    def __init__(self,
                 d_model: int,
                 heads: int,
                 d_head: int,
                 parallel: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        super().__init__()

        self.project_tensor_into_headspace = Basics.Linear(d_model,
                                                       [heads, d_head],
                                                       parallel,
                                                       dtype, device)
    def forward(self, tensor: torch.Tensor)->torch.Tensor:
        """ Move items out of the way, perform projection, then restore"""
        tensor = tensor.movedim(-2, 0)
        tensor = self.project_tensor_into_headspace(tensor)
        tensor = tensor.movedim(0, -2)
        return tensor

class MergeHeads(nn.Module):
    """
    A helper layer. This layer will
    take in a tensor with attention heads
    on it and merge the heads back together.
    """
    def __init__(self,
                 d_model: int,
                 heads: int,
                 d_head: int,
                 parallel: Optional[torch.Tensor],
                 dtype: Optional[torch.dtype],
                 device: Optional[torch.device],
                 ):

        super().__init__()

        self.merge_heads = Basics.Linear([heads, d_head], d_model,
                                         parallel, dtype, device)
    def forward(self, tensor: torch.Tensor)->torch.Tensor:
        """Move the head dimension next to the embedding dimension, merge, return"""
        tensor = tensor.movedim(-2, 0)
        tensor = self.merge_heads(tensor)
        tensor = tensor.movedim(0, -2)
        return tensor
