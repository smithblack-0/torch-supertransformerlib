"""
A module for handling multiheaded attention and related
utility functions
"""
from typing import Optional, Union, List

import torch
from torch import nn

from src.supertransformerlib import Basics
from src.supertransformerlib.Attention.Utility import dot_product_attention


@torch.jit.script
class _MultiHeadedAttention_Forward:
    """
    A torchscript class implimenting multiheaded attention's
    forward mechanism. It is designed to be produced
    by a factory class.
    """
    def __init__(self,
                 query_projector: Basics.LinearFactory.Type,
                 key_projector: Basics.LinearFactory.Type,
                 value_projector: Basics.LinearFactory.Type,
                 collapse_projector: Basics.LinearFactory.Type
                 ):
        self.query_projector = query_projector
        self.key_projector = key_projector
        self.value_projector = value_projector
        self.collapse_projector = collapse_projector

    def __call__(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """


        :param query: The query. Of dynamic_shape (...,(dynamic), (...parallel), items, embedding)
        :param key: The key, Of dynamic_shape (..., (dynamic), (...parallel), content_items, embedding)
        :param value: The value. Of dynamic_shape, (..., (dynamic), (...parallel), content_items, embedding)
        :param mask: A bool mask. True masks. Optional. Of dynamic_shape (..., (ensemble), items, content_items)
        :return: tensor. Attention result
        """

        # Perform head generation

        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)
        value = value.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)

        headed_query = self.query_projector(query)  # (item ..., (dynamics), (..parallel), head, head_dim)
        headed_key = self.key_projector(key)  # (item ..., (dynamics), (..parallel), head, head_dim)
        headed_value = self.value_projector(value)  # (item ..., (dynamics), (..parallel), head, head_dim)

        headed_query = headed_query.unsqueeze(-2).transpose(0, -2).squeeze(
            0)  # ..., (dynamics), (..parallel), head, item, head_dim)
        headed_key = headed_key.unsqueeze(-2).transpose(0, -2).squeeze(
            0)  # ..., (dynamics), (..parallel), head, item, head_dim)
        headed_value = headed_value.unsqueeze(-2).transpose(0, -2).squeeze(
            0)  # ..., (dynamics), (..parallel), head, item, head_dim)

        # Do dot product attention
        attn = dot_product_attention(headed_query, headed_key, headed_value,
                                     mask)  # (...,(dynamics),(..parallel), head, item, head_dim)

        # Reduce heads. Return
        attn = attn.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item,...,(dynamics),(..parallel), head, head_dim)
        output = self.collapse_projector(attn)  # (item,...,(dynamics),(..parallel), embedding)
        output = output.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # (...,(dynamics),(..parallel), item, embedding)

        return output


class MultiHeadedAttentionFactory(nn.Module):
    """
    A MultiHeadedAttention factory. It can generate
    MultiHeadedAttention layers which can then
    execute Multiheaded Attention. Notably,
    using factory techniques, it can be passed
    around under torchscript.

    """
    Type = _MultiHeadedAttention_Forward

    def __init__(self,
                 d_query: int,
                 d_content: int,
                 d_output: int,
                 heads: int,
                 parallel: Optional[Union[torch.Tensor, List[int], int]] = None,
                 ):
        """

        :param d_query: The dimensions of the query's embeddings
        :param d_content: The dimensions of the contents embedding
        :param d_output: The output embedding
        :param heads: The number of heads to initialize the MHA with.
        :param parallel: How much and in what shape to parallelize execution.
        """

        super().__init__()

        assert d_query % heads == 0
        head_width = d_query // heads

        self.queryFactory = Basics.LinearFactory(d_query, [heads, head_width], parallel=parallel)
        self.keyFactory = Basics.LinearFactory(d_content, [heads, head_width], parallel=parallel)
        self.valueFactory = Basics.LinearFactory(d_content, [heads, head_width], parallel=parallel)
        self.collapseFactory = Basics.LinearFactory([heads, head_width], d_output, parallel=parallel)


    def forward(self) -> _MultiHeadedAttention_Forward:

        return _MultiHeadedAttention_Forward(
            self.queryFactory(),
            self.keyFactory(),
            self.valueFactory(),
            self.collapseFactory(),
        )
