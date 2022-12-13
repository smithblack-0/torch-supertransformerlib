from typing import Optional, List, Union

import torch
from torch import nn
from src.supertransformerlib import Basics, Core

@torch.jit.script
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

@torch.jit.script
class _MakeHead:
    """
    A helper virtual layer.

    It makes the attention heads and
    ensures the resulting tensors
    are in (..., head, item, embedding)
    format.
    """
    def __init__(self,
                 Projector: Basics.LinearFactory.Type
                 ):
        self.Projector = Projector

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        headed = self.Projector(tensor)
        correctly_ordered_headed= headed.movedim(-2, -3)
        return correctly_ordered_headed


class MakeHeadFactory(nn.Module):
    """
    A factory layer designed for
    making the make head virtual layers.
    """
    Type = _MakeHead
    def __init__(self,
                 d_model: int,
                 heads: int,
                 d_head: int,
                 parallel: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        super().__init__()
        self.linear = Basics.LinearFactory(d_model, [heads, d_head], parallel,
                                           dtype, device)
    def forward(self)->_MakeHead:
        output = _MakeHead(self.linear())
        return output

@torch.jit.script
class RemoveHeads:
    """
    A helper virtual layer.

    This is responsible for taking the results
    of attention and eliminating the heads from
    it, restoring the original model shape.
    """
    def __init__(self,
                 flattenProjector: Basics.LinearFactory.Type,
                 ):
        self.flattenProjector = flattenProjector

    def __call__(self, attn_result: torch.Tensor)->torch.Tensor:
        reordered_results = attn_result.movedim(-3, -2)
        deheaded_results = self.flattenProjector(reordered_results)
        return deheaded_results


class RemoveHeadsFactory(nn.Module):

    def __init__(self,
                 d_model: int,
                 heads: int,
                 d_head: int,
                 parallel: Optional[torch.Tensor],
                 dtype: Optional[torch.dtype],
                 device: Optional[torch.device],
                 ):

        super().__init__()

        self.linear = Basics.LinearFactory([heads, d_head], d_model,
                                           parallel, dtype, device)
    def __call__(self)->RemoveHeads:
        return RemoveHeads(self.linear())
