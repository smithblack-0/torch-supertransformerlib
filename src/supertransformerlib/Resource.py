"""
Various resource paradynms that can be fed
into a generative decoder.
"""
from typing import Optional

import torch
from torch import nn
from . import Attention
from . import Adaptive


@torch.jit.script
class AttentionResource:
    """
    A raw attention resource,
    for looking up information.
    """

    def __init__(self,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 mask: Optional[torch.Tensor],
                 attn: Attention.MultiHeadedAttention.ForwardType,
                 ):
        self.key = key
        self.value = value
        self.mask = mask
        self.attn = attn

    def __call__(self, query: torch.Tensor, map: Optional[Adaptive.AdaptiveMap] = None):

        if map is not None:
            key = map.transform(self.key)
            value = map.transform(self.value)
        else:
            key = self.key
            value = self.value

        output = self.attn(query, key, value, self.mask)
        return output


class AttentionResourceFactory(nn.Module):
    """
    A location for attention resource
    kernels to lay, and for resource
    creation.

    This can be fed three of the four parts of attention,
    that is the key, value, and mask, and will then create
    an "AttentionResource" which is a partial call which
    will allow the completion of the call to come later.

    When the returned item is called with a query, attention
    is executed between it and the constructed items.
    """

    def __init__(self,
                 attention: Attention.MultiHeadedAttention,
                 ):
        """
        :param attention: The customized attention layer.
        """
        super().__init__()
        self.mha = attention

    def forward(self,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None)->AttentionResource:

        return AttentionResource(key,
                                 value,
                                 mask,
                                 self.mha.setup_forward()
                                 )


