from typing import Optional, List, Union

import torch
from torch import nn

import src.supertransformerlib.Basics
from . import Glimpses
from . import Basics


def _dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None, ):
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
class _FeedForward_Forward:
    """
    A feedforward mechanism capable of executing
    a feedforward operation after the kernels have been
    loaded. Works with main layer FeedForward
    """
    def __init__(self,
                 ff1: Basics.Linear.ForwardType,
                 ff2: Basics.Linear.ForwardType,
                 ):

        self.ff1 = ff1
        self.ff2 = ff2

    def __call__(self, tensor: torch.Tensor):
        """
        :param tensor: A tensor to perform feedforward on
        :return tensor: The result of feedforward processing.
        """
        # Basically, we move the item dimension to the front
        # of the tensor, apply the linear layers, and
        # return the results

        tensor = tensor.unsqueeze(0).transpose(0, -2).squeeze(-2)  # Transfer the item channel out of the wayh
        tensor = self.ff1(tensor)  # (item, ..., (config), (ensemble), internal)
        tensor = torch.relu(tensor)
        tensor = self.ff2(tensor)  # (item,..., (config) , (ensemble), embedding)
        tensor = tensor.unsqueeze(-2).transpose(0, -2).squeeze(0)  # Transfer item back into position
        return tensor


class FeedForward(nn.Module):
    """
    A feedforward layer for attention purposes.
    As a subclass of KernelSpace, and being built using
    core linear layers, it supports the operations of
    parallel execution along with ensemblelike configuration.

    --- parallelization ---

    This is a feature better defined over on the Core.Linear
    section. It may be defined just as in a linear layer,
    and allows execution of a kernel across an entire tensor
    independently and in parallel.

    Long story short, the shape you place here will be
    added onto the deployed kernels, such that a problem
    with a tensor of shape (..., 20, 128) with
    parallelization (10, 15) will require a tensor
    of shape (..., 10, 15, 20, 128) and process each
    defined dimensions with completely independent parameters

    --- config ---

    set_config can be used to set all the configurations properly and without issue.
    """
    ForwardType = _FeedForward_Forward
    def __init__(self,
                 d_model: int,
                 d_internal: Optional[int] = None,
                 parallelization: Optional[Union[torch.Tensor, List[int], int]] = None,
                 ):
        """
        :param d_model: The model width
        :param d_internal: The internel width. By default 2048
        :param parallelization: The parallization layers
        :param dynamics: The dynamic width. None means off.
        """
        if d_internal is None:
            d_internal = 2048

        super().__init__()

        # Setup a few flags

        if parallelization is None:
            self.ff1 = src.supertransformerlib.Basics.Basics.Linear(d_model, d_internal, parallel=parallelization)
            self.activation = nn.ReLU()
            self.ff2 = src.supertransformerlib.Basics.Basics.Linear(d_internal, d_model, parallel=parallelization)
        else:
            self.ff1 = src.supertransformerlib.Basics.Basics.Linear(d_model, d_internal,
                                                                    parallel=parallelization,
                                                                    )
            self.activation = nn.ReLU()
            self.ff2 = src.supertransformerlib.Basics.Basics.Linear(d_internal, d_model,
                                                                    parallel=parallelization,
                                                                    )
    def setup_forward(self)->_FeedForward_Forward:
        """
        Sets up a callable function which will execute feedforward using
        the current kernel. This is torchscript passable.
        """
        #Sets up the discrete forward class, which is
        #passible through torchscript
        return _FeedForward_Forward(self.ff1.setup_forward(),
                                    self.ff2.setup_forward())

    def forward(self, tensor: torch.Tensor):
        """
        :param tensor: A tensor to perform feedforward on
        :return tensor: The result of feedforward processing.
        """
        forward_call = self.setup_forward()
        return forward_call(tensor)

@torch.jit.script
class _MultiHeadedAttention_Forward:
    """
    A small torchscript class to do multiheaded
    attention when given the current kernels.

    Works with the master class.
    """
    def __init__(self,
                 query_projector: Basics.Linear.ForwardType,
                 key_projector: Basics.Linear.ForwardType,
                 value_projector: Basics.Linear.ForwardType,
                 collapse_projector: Basics.Linear.ForwardType
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


        :param query: The query. Of shape (...,(dynamic), (...parallel), items, embedding)
        :param key: The key, Of shape (..., (dynamic), (...parallel), content_items, embedding)
        :param value: The value. Of shape, (..., (dynamic), (...parallel), content_items, embedding)
        :param mask: A bool mask. True masks. Optional. Of shape (..., (ensemble), items, content_items)
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
        attn = _dot_product_attention(headed_query, headed_key, headed_value,
                                      mask)  # (...,(dynamics),(..parallel), head, item, head_dim)

        # Reduce heads. Return
        attn = attn.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item,...,(dynamics),(..parallel), head, head_dim)
        output = self.collapse_projector(attn)  # (item,...,(dynamics),(..parallel), embedding)
        output = output.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # (...,(dynamics),(..parallel), item, embedding)

        return output


class MultiHeadedAttention(nn.Module):
    """
    A Multiheaded Attention layer capable of
    executing parallization alongside ensemble configured
    assembly

    --- parallelization ---

    This is a feature better defined over on the Core.Linear
    section. It may be defined just as in a linear layer,
    and allows execution of a kernel across an entire tensor
    independently and in parallel.

    Long story short, the shape you place here will be
    added onto the deployed kernels, such that a problem
    with a tensor of shape (..., 20, 128) with
    parallelization (10, 15) will require a tensor
    of shape (..., 10, 15, 20, 128) and process each
    defined dimensions with completely independent parameters


    """
    ForwardType = _MultiHeadedAttention_Forward

    def __init__(self,
                 d_query: int,
                 d_content: int,
                 d_output: int,
                 heads: int,
                 parallelization: Optional[Union[torch.Tensor, List[int], int]] = None,
                 ):
        """

        :param d_query: The dimensions of the query's embeddings
        :param d_content: The dimensions of the contents embedding
        :param d_output: The output embedding
        :param heads: The number of heads to initialize the MHA with.
        :param parallelization: How much, and in what shape, to parallelize. Static.
        :param dynamics: Whether or not to setup extra kernelspace for dynamic configuration.
        """

        super().__init__()

        assert d_query % heads == 0
        head_width = d_query // heads

        self.query_projector = src.supertransformerlib.Basics.Basics.Linear(d_query, [heads, head_width], parallel=parallelization)
        self.key_projector = src.supertransformerlib.Basics.Basics.Linear(d_content, [heads, head_width], parallel=parallelization)
        self.value_projector = src.supertransformerlib.Basics.Basics.Linear(d_content, [heads, head_width], parallel=parallelization)
        self.collapse_projector = src.supertransformerlib.Basics.Basics.Linear([heads, head_width], d_output, parallel=parallelization)

    def setup_forward(self)->_MultiHeadedAttention_Forward:
        """
        Sets up a torchscript passable forward
        function for multihead attention.

        Be warned: returned function call does not update
        upon gradient descent.
        """
        return _MultiHeadedAttention_Forward(
            self.query_projector.setup_forward(),
            self.key_projector.setup_forward(),
            self.value_projector.setup_forward(),
            self.collapse_projector.setup_forward(),
        )

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """


        :param query: The query. Of shape (...,(dynamic), (...parallel), items, embedding)
        :param key: The key, Of shape (..., (dynamic), (...parallel), content_items, embedding)
        :param value: The value. Of shape, (..., (dynamic), (...parallel), content_items, embedding)
        :param mask: A bool mask. True masks. Optional. Of shape (..., (ensemble), items, content_items)
        :return: tensor. Attention result
        """
        forward_call = self.setup_forward()
        return forward_call(query, key, value, mask)

@torch.jit.script
class _PIMU_Forward:
    """
    The forward call function for the pimu process
    """
    def __init__(self,
                 query_projector: Basics.Linear.ForwardType,
                 key: nn.Parameter,
                 value: nn.Parameter,
                 dehead_projector: Basics.Linear.ForwardType,
                 ):

        self.QueryProj = query_projector
        self.Key = key
        self.Value = value
        self.DeheadProj = dehead_projector

    def __call__(self, query: torch.Tensor) -> torch.Tensor:
        """
        :param query: A tensor to gain insight on
        :return: The calibrated result of the query.
        """

        # Perform head generation

        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)
        query = self.QueryProj(query)
        query = query.unsqueeze(-2).transpose(-2, 0).squeeze(0)

        key = self.Key
        value = self.Value

        # Perform dot product attention=

        attn = _dot_product_attention(query, key, value)  #

        # Collapse heads, then return
        attn = attn.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item,...,(dynamics),(..parallel), head, head_dim)
        output = self.DeheadProj(attn)
        output = output.unsqueeze(-2).transpose(-2, 0).squeeze(0)
        return output


class PIMU(nn.Module):
    """
    Parameter Injection Memory Unit. (PIMU)

    Parameter Memory are large blocks of parameters
    which act as if they are a key or value embedding,
    and are run by a given query. This means that all
    sorts of memorized context can be embedded within
    such a block.

    The location of best effect for parameter injection
    is within a model of some sort that is mapping many
    inputs onto only a few results, at some point in
    the logic of the process. This may exist in a
    transformer unit, a imagenet flow, or even just
    a standard dense network.

    For high granulaty, a high head count and softmax mode
    are desirable. In this case many options are considered
    avaibable. For a case in which only a few options should
    be allowed at each step, a low head count is recommmended.
    Generally, it is recommended to start with a high head
    count where possible; more heads does NOT slow the
    model down.

    """
    ForwardType = _PIMU_Forward

    def __init__(self,
                 d_model: int,
                 mem_width: int,
                 heads: int,
                 parallelization: Optional[Union[torch.Tensor, List[int], int]] = None,
                 ):
        """

        :param d_model: The embedding width. An int
        :param mem_width: The memory width. An int. Increases parameters
        :param heads: The number of heads to use
        :param parallelization: What tensor dimensions are static, and should be handled separately for each layer.
        :param dynamics: If defined, how many units wide the dynamics kernel should be.
        """
        super().__init__()

        assert d_model % heads == 0

        head_channel_width = d_model // heads
        # Construct kernel shape

        kernel_shape: List[int] = []
        kernel_shape = [heads, mem_width, head_channel_width] + kernel_shape

        if parallelization is not None:
            parallelization = self.standardize_input(parallelization)
            kernel_shape = parallelization.tolist() + kernel_shape

        key = torch.zeros(kernel_shape)
        value = torch.zeros(kernel_shape)

        nn.init.kaiming_uniform_(key)
        nn.init.kaiming_uniform_(value)

        self.QueryProj = src.supertransformerlib.Basics.Basics.Linear(d_model, [heads, head_channel_width], parallelization)
        self.Key = nn.Parameter(key)
        self.Value = nn.Parameter(value)
        self.DeheadProj = src.supertransformerlib.Basics.Basics.Linear([heads, head_channel_width], d_model, parallelization)
    def setup_forward(self)->_PIMU_Forward:
        """
        Set up a passable, callable, forward function that
        torchscript is happy to work with.
        :return: A forward callable
        """
        return _PIMU_Forward(
            self.QueryProj.setup_forward(),
            self.Key,
            self.Value,
            self.DeheadProj.setup_forward()
        )
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        :param query: A tensor to gain insight on
        :return: The calibrated result of the query.
        """

        forward_call = self.setup_forward()
        return forward_call(query)

@torch.jit.script
class _PISU_Forward:
    """
    The forward function for the PISU
    unit.
    """
    def __init__(self,
                 query: nn.Parameter,
                 key_projector: Basics.Linear.ForwardType,
                 value_projector: Basics.Linear.ForwardType,
                 dehead_projector: Basics.Linear.ForwardType
                 ):

        self.query = query
        self.key_projector = key_projector
        self.value_projector = value_projector
        self.dehead_projector = dehead_projector

    def __call__(self, content: torch.Tensor) -> torch.Tensor:
        """
        :param content: (..., (ensembles), items, embeddings)
        :return:
        """
        # content: (..., (parallel...), items, embeddings)

        # Prepare heads.

        content = content.unsqueeze(0).transpose(0, -2).squeeze(-2)  # (item, ...,  (parallel...), embedding)

        query = self.query  # ((parallel...),head, output_item, head_embed)
        key = self.key_projector(content)  # (item, ..., (parallel), head, head_embed)
        value = self.value_projector(content)  # (item, ..., (parallel...), head, head_embed)

        key = key.unsqueeze(-2).transpose(0, -2).squeeze(0)  # (..., (parallel..), head, item, head_embed)
        value = value.unsqueeze(-2).transpose(0, -2).squeeze(0)  # (..., (parallel..), head, item, head_embed)

        # Perform dot product attention

        attn = _dot_product_attention(query, key, value)  # (..., (parallel...), head, output_items, head_embed)

        # Collapse heads and return

        attn = attn.unsqueeze(0).transpose(0, -2).squeeze(-2)
        output = self.dehead_projector(attn)
        output = output.unsqueeze(-2).transpose(-2, 0).squeeze(0)
        return output


class PISU(nn.Module):
    """
    Parameter Injected Summary Unit (PISU)

    An attention layer designed to enable the
    collapse of a large number of items into
    something of fixed width. The sibling of PISU

    A fixed width, parameter based query is presented
    as attention to heads generate from the incoming content. The
    result, a embedding of the same width as indicated,
    is then returned.

    Note that, as with PISU, an aggressive number of heads will
    allow more degrees of freedom, while fewer will allow less.
    """


    def __init__(self,
                 d_model: int,
                 d_output: int,
                 output_items: int,
                 heads: int,
                 parallelization: Optional[Union[torch.Tensor, List[int], int]] = None,
                 ):
        """
        :param d_model: The embeddings width. Dim -1
        :param output_items: How many distinct items the output should have. Dim -2.
        :param heads: How many heads should exist
        :param parallelization: The number of parallel kernels to manufacture
        :param dynamics: If used, how big to make the dynamics portion of the kernel.
        """

        super().__init__()
        assert d_model % heads == 0
        head_width = d_model // heads

        query_shape = torch.tensor([heads, output_items, head_width])
        if parallelization is not None:
            parallelization = self.standardize_input(parallelization)
            query_shape = torch.concat([parallelization, query_shape], dim=0)

        query = torch.zeros(query_shape.tolist())
        nn.init.kaiming_uniform_(query)
        query = nn.Parameter(query)

        self.query = query
        self.key_projector = src.supertransformerlib.Basics.Basics.Linear(d_model, [heads, head_width], parallelization)
        self.value_projector = src.supertransformerlib.Basics.Basics.Linear(d_model, [heads, head_width], parallelization)
        self.dehead = src.supertransformerlib.Basics.Basics.Linear([heads, head_width], d_output, parallelization)

    def setup_forward(self) -> _PISU_Forward:
        """Setup a callable forward function"""
        return _PISU_Forward(
            self.query,
            self.key_projector.setup_forward(),
            self.value_projector.setup_forward(),
            self.dehead.setup_forward(),
        )

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        """
        :param content: (..., (ensembles), items, embeddings)
        :return:
        """
        forward_call = self.setup_forward()
        return forward_call(content)



class LCSA(nn.Module):
    """
    Local Context Self Attention (LCSA)

    A banded self attention class with positional
    intelligence. Once it constructs a convolutional
    kernel, each dimension is projected using an
    independent linear action. The net effect is the
    layer can learn to consider words at different
    positions in different manners.

        Multiple padding options exist allowing conditioning
        to be done based on only words that came before, only
        words that come after, or a centered view with both.

    One thing of note: The number of words passed into
    the layer MUST be equal to or greater than the kernel width.
    Remember to pad to above this length.

    Combined with add+norm, this nicely handles local context.

    """

    def __init__(self,
                 d_model: int,
                 kernel_width: int,
                 dilations: List[int],
                 parallelization: Optional[Union[torch.Tensor, List[int], int]] = None,
                 mode: str = "center",
                 ):
        """

        :param d_model: The width of the embedding dimension
        :param kernel_width: How wide the local kernel shall be
        :param dilations: A specification for how to dilate each head. The length of this
            determines the number of heads.
        :param parallelization: How, and in what shape, to perform parallel kernel operations
        :param dynamics: Whether or not, and how large, to make the dynamic kernel ability.
        :param  mode: controls how the padding is constructed. "forward", "center", "backward" are the
            options. forward only knows about prior words, center about words to the front and back,
            and backwards only future words.
        """
        super().__init__()
        heads = len(dilations)
        assert d_model % heads == 0, "Dilations not compatible with d_model"
        head_width = d_model // heads

        self.kernel_width = kernel_width
        self.dilations = dilations
        self.mode = mode

        parallel_shape = torch.tensor([heads])
        if parallelization is not None:
            parallelization = self.standardize_input(parallelization)
            parallel_shape = torch.concat([parallelization, parallel_shape], dim=0)

        self.query_projector = src.supertransformerlib.Basics.Basics.Linear(d_model, head_width, parallel_shape)
        self.key_projector = src.supertransformerlib.Basics.Basics.Linear(d_model, head_width, parallel_shape)
        self.value_projector = src.supertransformerlib.Basics.Basics.Linear(d_model, head_width, parallel_shape)
        self.dehead = src.supertransformerlib.Basics.Basics.Linear([heads, head_width], d_model, parallelization)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """


        :param tensor: The tensor to perform self attention with. Shape (..., (ensemble), item, embedding).
        :return: A tensor. The result of self attention. Shape (..., (ensemble), item, embedding)
        :raise RuntimeError: If the number of items is too small for the kernel.
        """
        # Tensor: (..., (parallel...), items, embedding)

        # Create local kernel out of input providing interaction between nearby elements
        # Rearrange kernel for independent parallel projection into head space.

        tensor = tensor.unsqueeze(0).transpose(-1, 0).squeeze(-1)

        query = Glimpses.dilocal(tensor, 1, 1,
                                 self.dilations)  # (embedding, ..., (parallel..), , head, items, local items [1])
        content = Glimpses.dilocal(tensor, self.kernel_width, 1, self.dilations,
                                   pad_justification=self.mode)

        # Move items to front dims. Move embedding to back dim
        # perform head projection. Restore

        query = query.transpose(-1, 0)
        content = content.transpose(-1, 0)
        query = query.unsqueeze(0).transpose(0, -2).squeeze(-2)
        content = content.unsqueeze(0).transpose(0, -2).squeeze(-2)  # (items,.=local_items, .., (parallel...),  head, embedding)

        query = self.query_projector(query)
        key = self.key_projector(content)
        value = self.value_projector(content)

        query = query.unsqueeze(-2).transpose(-2, 0).squeeze(0)
        key = key.unsqueeze(-2).transpose(-2, 0).squeeze(0)
        value = value.unsqueeze(-2).transpose(-2, 0).squeeze(0)

        query = query.unsqueeze(-2).transpose(-2, 0).squeeze(0)
        key = key.unsqueeze(-2).transpose(-2, 0).squeeze(0)
        value = value.unsqueeze(-2).transpose(-2, 0).squeeze(0)

        # Perform self attention

        attn = _dot_product_attention(query, key, value)
        attn = attn.squeeze(-2)

        # Remove the head then return.

        attn = attn.unsqueeze(0).transpose(-2, 0).squeeze(-2)
        outcome = self.dehead(attn)  # #(items, ..., (parallel...), d_model)
        outcome = outcome.unsqueeze(-2).transpose(-2, 0).squeeze(0)
        return outcome

