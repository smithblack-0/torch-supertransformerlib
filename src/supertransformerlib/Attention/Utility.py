from typing import Optional, List, Union

import torch
from torch import nn
from src.supertransformerlib import Basics, Core

def dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None, )->torch.Tensor:
    """
    Performs dot product attention, as
    shown in "attention is all you need"

    :param query: The query.
    :param key: The key.
    :param value: The value
    :param mask: Any mask to include.
    :return:
    """

    logits = torch.matmul(query, key.transpose(-1, -2))
    if mask is not None:
        logits = logits.masked_fill(mask, -1e+8)
    score = torch.softmax(logits, dim=-1)
    attn = torch.matmul(score, value)
    attn = attn / torch.sqrt(torch.tensor([query.shape[-1]], device=logits.device))
    return attn

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
