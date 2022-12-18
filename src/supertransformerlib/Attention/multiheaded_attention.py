"""
A module for handling multiheaded attention and related
utility functions
"""
from typing import Optional

import torch
from torch import nn
from src.supertransformerlib import Core
from src.supertransformerlib.Attention import Utility


class MultiHeadedAttention(nn.Module):
    """
    A layer for performing multiheaded attention.

    Accepts the standard query, key, value, mask
    combo and returns the result.
    """
    @staticmethod
    def make(
                 d_model: int,
                 heads: int,
                 d_key: Optional[int] = None,
                 d_value: Optional[int] = None,
                 d_output: Optional[int] = None,
                 parallel: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 )->"MultiHeadedAttention":
        """
        Makes a multiheaded attention layer all at once.

        :param d_model: The dimensions of the query's embeddings
        :param d_key: The dimensions of the key embeddings. Defaults to d_query if not given
        :param d_value: The dimension of the value embedding. Defaults to d_query
        :param d_output: The output embedding. Defaults to d_query if not given.
        :param heads: The number of heads to initialize the MHA with.
        :param parallel: How much and in what shape to parallelize execution.
        """

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

        query_head_gen = Utility.MakeHead(d_model, heads, d_head,
                                          parallel, dtype, device)
        key_head_gen = Utility.MakeHead(d_key, heads, d_head,
                                        parallel, dtype, device)
        value_head_gen = Utility.MakeHead(d_value, heads, d_head,
                                          parallel, dtype, device)
        head_reducer = Utility.MergeHeads(d_output, heads, d_head,
                                          parallel, dtype, device)

        return MultiHeadedAttention(query_head_gen,
                                    key_head_gen,
                                    value_head_gen,
                                    head_reducer)

    def __init__(self,
                 query_head_gen: Utility.MakeHead,
                 key_head_gen: Utility.MakeHead,
                 value_head_gen: Utility.MakeHead,
                 head_merger: Utility.MergeHeads
                 ):
        """
        Puts together the various layers
        together into a single cohesive whole.

        To define the layer from scratch, consider
        using the .make static method

        Custom head behavior can be gained by swapping out layers.

        :param query_head_gen: The layers responsible for forming the query head
        :param key_head_gen: The layers responsible for forming the key head
        :param value_head_gen: The layers responsible for forming the value head
        :param head_merger: The layers responsible for eliminating the heads
        """

        super().__init__()

        self.make_query_head = query_head_gen
        self.make_key_head = key_head_gen
        self.make_value_head = value_head_gen
        self.merge_heads = head_merger

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

        attn = Utility.dot_product_attention(query, key, value, mask)

        output = self.merge_heads(attn)
        return output
