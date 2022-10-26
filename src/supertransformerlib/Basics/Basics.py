import math
from collections import namedtuple
from typing import Tuple, Optional, Union, List

import torch
import torch.jit
from torch import nn

from src.supertransformerlib import Glimpses
from src.supertransformerlib.Core.Functions import standardize_shape


@torch.jit.script
class ViewPoint:
    """
    A callable function for performing a viewpoint operation.

    Is given a number of views, a view width,
    a weight tensor, and an index tensor. May then
    be called to create views of segments of a text.

    Should be created by the ViewPointFactory.
    """
    def __init__(self,
                 views: int,
                 view_width: int,
                 weights: torch.Tensor,
                 index: torch.Tensor,
                 ):
        """

        :param views: The number of views
        :param view_width: The view width
        :param weights: The weights tensor. In dynamic_shape [..., views, query, top_k]
        :param index: The index tensor. In dynamic_shape [..., views, query, top_k]
        """
        self.view_width = view_width
        self.views = views
        self.weights = weights
        self.index = index

    def __call__(self, tensor: torch.Tensor)->torch.Tensor:


        #Generate the draw source. This will be a memory efficient strided
        #view of the input tensor

        strided_source = tensor.unsqueeze(0).transpose(0, -1).squeeze(-1)
        strided_source = Glimpses.dilocal(strided_source,
                                          self.view_width,
                                          1,
                                          [1]) # (..parallel), viewpoint, item, local, viewpoint_dim)
        strided_source = strided_source.squeeze(-3)
        strided_source = strided_source.unsqueeze(-1).transpose(-1, 0).squeeze(0)

        #The following code sets up a gather
        #
        #This consists of getting the index and strided source the same dynamic_shape,
        #then using expand to ensure gather will select all required index.
        #
        #Gather basically expects, along each non-gathered dimension, a
        #list indicating what elements to grab.

        index = self.index.unsqueeze(-1).unsqueeze(-1)
        strided_source = strided_source.unsqueeze(-4).unsqueeze(-4)

        index_expansion = [-1]*index.dim()
        source_expansion = [-1]*strided_source.dim()

        source_expansion[-4] = index.shape[-4]
        source_expansion[-5] = index.shape[-5]
        index_expansion[-1] = strided_source.shape[-1]
        index_expansion[-2] = strided_source.shape[-2]

        index = index.expand(index_expansion)
        strided_source = strided_source.expand(source_expansion)
        gathered_viewpoint = torch.gather(strided_source, dim=-3, index=  index)

        #Weight the viewpoints, combine them together, then return the result

        gathered_viewpoint = gathered_viewpoint*self.weights.unsqueeze(-1).unsqueeze(-1)
        output = gathered_viewpoint.sum(-3)
        return output


class ViewPointFactory(nn.Module):
    """
    A factory for making a ViewPoint, which is
    a map that returns a sequence of tensors with each
    tensor being corrolated to a distinct section of
    text.
    """

    def __init__(self,
                 d_query,
                 d_key,
                 viewpoints: int,
                 viewpoint_width: int,
                 top_k: int,
                 parallelization: Optional[Union[torch.Tensor, List[int], int]] = None,
                 ):
        super().__init__()
        d_viewpoint = d_query // viewpoints
        self.viewpoints = viewpoints
        self.query_projector = Linear(d_query, [viewpoints, d_viewpoint], parallelization)
        self.key_projector = Linear(d_key, [viewpoints, d_viewpoint], parallelization)
        self.width = viewpoint_width
        self.top_k = top_k

    def forward(self, query, key) -> ViewPoint:
        # Generate the viewpoint dims
        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, , (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, , (..parallel), embedding)

        viewpoint_query = self.query_projector(query)  # (item ..., , (..parallel), viewpoint, viewpoint_dim)
        viewpoint_key = self.key_projector(key)  # (item ..., , (..parallel), viewpoint, viewpoint_dim)

        viewpoint_query = viewpoint_query.unsqueeze(-2).transpose(0, -2).squeeze(
            0)  # ...,, (..parallel), viewpoint, item, viewpoint_dim)
        viewpoint_key = viewpoint_key.unsqueeze(-2).transpose(0, -2).squeeze(
            0)  # ..., , (..parallel), viewpoint, item, viewpoint_dim)

        # Sort out the candidates. Generate the index.

        score = torch.matmul(viewpoint_query, viewpoint_key.transpose(-1, -2))
        score, index = torch.sort(score, dim=-1, descending=True)
        index = index[..., :self.top_k]
        score = score[..., :self.top_k]
        score = torch.softmax(score, dim=-1)

        #Return the viewpoint.

        return ViewPoint(self.viewpoints,
                         self.width,
                         score,
                         index)