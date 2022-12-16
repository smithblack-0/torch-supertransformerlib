"""
A module for handling multiheaded attention and related
utility functions
"""
from typing import Optional

import torch
from torch import nn
from src.supertransformerlib import Core
from src.supertransformerlib.Attention import Utility


class _MultiHeadedAttentionForward:
    """
    A torchscript class implimenting multiheaded attention's
    forward mechanism. It is designed to be produced
    by a factory class.
    """

    def __init__(self,
                 query_head_gen: Utility.MakeHeadFactory.Type,
                 key_head_gen: Utility.MakeHeadFactory.Type,
                 value_head_gen: Utility.MakeHeadFactory.Type,
                 deheader: Utility.RemoveHeadsFactory.Type,
                 ):

        self.make_query_head = query_head_gen
        self.make_value_head = value_head_gen
        self.make_key_head = key_head_gen
        self.dehead_attention = deheader

    def __call__(self,
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

        attn = Utility.dot_product_attention(query, key, value,
                                     mask)

        output = self.dehead_attention(attn)
        return output

torch.jit.script(_MultiHeadedAttentionForward)

class MultiHeadedAttentionFactory(nn.Module):
    """
    A MultiHeadedAttention factory. It can generate
    MultiHeadedAttention layers which can then
    execute Multiheaded Attention. Notably,
    using factory techniques, it can be passed
    around under torchscript.

    """
    Type = _MultiHeadedAttentionForward

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


        self.queryHeadFactory = Utility.MakeHeadFactory(d_model, heads, d_head,
                                                        parallel, dtype, device)
        self.keyHeadFactory = Utility.MakeHeadFactory(d_key, heads, d_head,
                                                      parallel, dtype, device)
        self.valueHeadFactory = Utility.MakeHeadFactory(d_value, heads, d_head,
                                                    parallel, dtype, device)
        self.deheaderFactory = Utility.RemoveHeadsFactory(d_output, heads, d_head,
                                                          parallel, dtype, device)

    def forward(self) -> _MultiHeadedAttentionForward:
        return _MultiHeadedAttentionForward(
            self.queryHeadFactory(),
            self.keyHeadFactory(),
            self.valueHeadFactory(),
            self.deheaderFactory(),
        )
