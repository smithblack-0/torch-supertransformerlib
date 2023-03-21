"""
A module for handling multiheaded attention and related
utility functions
"""
from typing import Optional

import torch
from torch import nn
from supertransformerlib.Basics.head_generation import MakeHead, MergeHeads
from supertransformerlib import Core


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


class MultiHeadedAttention(nn.Module):
    """
    A layer for performing multiheaded attention.
    Accepts the standard query, key, value, mask
    combo and returns the result.
    """
    def __init__(self,
                 d_model: int,
                 heads: int,
                 d_key: Optional[int] = None,
                 d_value: Optional[int] = None,
                 d_output: Optional[int] = None,
                 parallel: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        Makes a multiheaded attention layer all at once.
        :param d_model: The dimensions of the query's embeddings
        :param d_key: The dimensions of the key embeddings. Defaults to d_query if not given
        :param d_value: The dimension of the value embedding. Defaults to d_query
        :param d_output: The output embedding. Defaults to d_query if not given.
        :param heads: The number of heads to initialize the MHA with.
        :param parallel: How much and in what shape to parallelize execution.
        """
        super().__init__()

        if d_key is None:
            d_key = d_model
        if d_value is None:
            d_value = d_model
        if d_output is None:
            d_output = d_model

        d_head = d_model // heads
        if d_head <= 1:
            reason = """\
            The number of heads is so large that the embedding width has 
            been compressed to one or less. This means the attention mechanism
            does not have enough dimensions to work. 
            Increase d_model or decrease the number of heads
            """
            reason = Core.dedent(reason)
            raise Core.Errors.ValidationError("FactorySetupException", reason)

        query_head_gen = MakeHead(d_model, heads, d_head,
                                  parallel, dtype, device)
        key_head_gen = MakeHead(d_key, heads, d_head,
                                parallel, dtype, device)
        value_head_gen = MakeHead(d_value, heads, d_head,
                                  parallel, dtype, device)
        head_reducer = MergeHeads(d_output, heads, d_head,
                                  parallel, dtype, device)


        self.make_query_head = query_head_gen
        self.make_key_head = key_head_gen
        self.make_value_head = value_head_gen
        self.merge_heads = head_reducer


    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param query: The query. Of shape( (...parallel), items, embedding)
        :param key: The key, Of shape ( (...parallel), content_items, embedding)
        :param value: The value. Of shape, ( (...parallel), content_items, embedding)
        :param mask: A bool mask. Optional. Of shape (..., (...parallel), items, content_items).
                    True masks out.
        :return: tensor. Attention result
        """

        query = self.make_query_head(query)
        key = self.make_key_head(key)
        value = self.make_value_head(value)
        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(-3)

        attn = dot_product_attention(query, key, value, mask)

        output = self.merge_heads(attn)
        return output