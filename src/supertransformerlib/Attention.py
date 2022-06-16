from typing import Optional, List

import torch
from torch import nn

from . import Glimpses
from .Linear import Linear


def _dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None):
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
    attn = attn / torch.sqrt(torch.tensor([query.shape[-1]]))
    return attn


class FeedForward(nn.Module):
    """
    A feedforward layer for attention purposes.
    Permits ensembling, but nothing clever
    beyond that.

    Expects inputs to be tensors in (..., (ensemble), item, embedding)
    where ensemble is optional, and only used if the ensemble
    channel was defined on initialization. Returns something
    of the same shape

    See 'Attention is all you need' for details.
    """

    def __init__(self,
                 d_model: int,
                 d_internal: int = 2048,
                 ensembles: Optional[int] = None,
                 ):
        """

        :param d_model: How wide the model embedding is
        :param d_internal: How wide the internal channel is
        :param ensembles: How many ensembles to construct
        """
        super().__init__()
        if ensembles is None:
            ensembles = 1
            self.ensembles = False
        else:
            self.ensembles = True

        self.ff1 = Linear(d_model, d_internal, ensembles)
        self.activation = nn.ReLU()
        self.ff2 = Linear(d_internal, d_model, ensembles)

    def forward(self, tensor: torch.Tensor):
        """
        :param tensor: A tensor to perform feedforward on
        :return tensor: The result of feedforward processing.

        """
        if self.ensembles is False:
            tensor = tensor.unsqueeze(-3)  # (..., ensemble, items, embedding)

        tensor = tensor.transpose(-2, -3)  # (..., items, ensemble, embedding)
        tensor = self.ff1(tensor)  # (..., item, ensemble, internal)
        tensor = self.activation(tensor)
        tensor = self.ff2(tensor)  # (..., item, ensemble, embedding)
        tensor = tensor.transpose(-2, -3)  # (..., ensemble, item, embedding)

        if self.ensembles is False:
            tensor = tensor.squeeze(-3)
        return tensor


class MultiHeadedAttention(nn.Module):
    """
    An ensemble-enabled implimentation of multiheaded attention.

    This is implimented as seen in "attention is all you need", with
    an additional option. The incoming content is projected to build
    heads, dot product attention is performed, multiheaded combine
    occurs, and the result is returned.

    A novel feature is the ensembles option. This may be left blank, but
    if defined will revise the expected input shape to be (..., ensemble, items, embedding).
    Each entry in ensembles will be processed in parallel, and with completely unique parameters.
    """

    def __init__(self,
                 d_query: int,
                 d_content: int,
                 d_output: int,
                 heads: int,
                 ensembles: Optional[int] = None,
                 ):
        """

        :param d_query: The dimensions of the query's embeddings
        :param d_content: The dimensions of the contents embedding
        :param d_output: The output embedding
        :param heads: The number of heads
        :param ensembles: Optional. The number of ensembles to construct.

        """
        super().__init__()
        assert d_query % heads == 0
        head_width = d_query // heads
        if ensembles is None:
            ensembles = 1
            self.ensembles = False
        else:
            self.ensembles = True

        self.query_projector = Linear(d_query, [heads, head_width], ensembles)
        self.key_projector = Linear(d_content, [heads, head_width], ensembles)
        self.value_projector = Linear(d_content, [heads, head_width], ensembles)
        self.collapse_projector = Linear([heads, head_width], d_output, ensembles)

    def forward(self,
                query: torch.Tensor,  # (..., (ensemble), item, embedding)
                key: torch.Tensor,  # (...., (ensemble), item, embedding)
                value: torch.Tensor,  # (..., (ensemble), item, embedding)
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """


            :param query: The query. Of shape (..., (ensemble), items, embedding)
            :param key: The key, Of shape (..., (ensemble), content_items, embedding)
            :param value: The value. Of shape, (..., (ensemble), content_items, embedding)
            :param mask: A bool mask. True masks. Optional. Of shape (..., (ensemble), items, content_items)
            :return: tensor. Attention result
            """

        # Prep query, key and value

        if self.ensembles is False:
            query = query.unsqueeze(-3)  # (..., ensemble, items, embeddings)
            key = key.unsqueeze(-3)
            value = value.unsqueeze(-3)

        if mask is not None:
            mask = mask.unsqueeze(-3)  # (..., ensemble, items, content_items)

        query = query.transpose(-2, -3)  # (..., item, ensemble, content_items)
        key = key.transpose(-2, -3)
        value = value.transpose(-2, -3)

        headed_query = self.query_projector(query)  # (..., item, ensemble, head, head_dim)
        headed_key = self.key_projector(key)  # (..., item_b, ensemble, head, head_dim)
        headed_value = self.value_projector(value)  # (..., item_b, ensemble, head, head_dim)

        headed_query = headed_query.transpose(-2, -4)  # (..., head, ensemble, item, head_dim)
        headed_key = headed_key.transpose(-2, -4)  # (..., head, ensemble, item2, head_dim)
        headed_value = headed_value.transpose(-2, -4)  # (..., head, ensemble, item2, head_dim)

        if mask is not None:
            mask = mask.unsqueeze(-4)

        attn = _dot_product_attention(headed_query, headed_key, headed_value,
                                      mask)  # (...,head, ensemble, item, head_dim)
        attn = attn.transpose(-2, -4)  # (..., item, ensemble, head, head_widht)
        output = self.collapse_projector(attn)  # (..., item, ensemble, embedding)
        if self.ensembles is False:
            output = output.squeeze(-2)
        else:
            output = output.transpose(-2, -3)

        return output


class PIMU(nn.Module):
    """
    Parameter Injection Memory Unit. (PIMU)

    Parameter Memory are large blocks of parameters
    which are compatible with an embedded stream
    as though they are embeddings themselves.

    The process of Parameter Injection is a process
    of conditionally injecting whole blocks of parameters,
    into a running embedded stream as though it were an
    embedding itself. Two tasks exist. First, the module
    must figure out what parameter block to inject, and
    when. Second, the module must train the parameter
    blocks to provide useful context.

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

    def __init__(self,
                 d_model: int,
                 mem_width: int,
                 heads: int,
                 ensembles: Optional[int] = None,
                 ):
        """

        :param d_model: The embedding width. An int
        :param mem_width: The memory width. An int. Increases parameters
        :param heads: The number of heads to use
        :param ensembles: How many ensembles to set up. If none, dimension is disabled.
        """
        super().__init__()

        assert d_model % heads == 0

        head_channel_width = d_model // heads
        if ensembles is None:
            key = torch.zeros([heads, mem_width, head_channel_width])
            value = torch.zeros([heads, mem_width, head_channel_width])
            self.ensembles = False
        else:
            key = torch.zeros([ensembles, heads, mem_width, head_channel_width])
            value = torch.zeros([ensembles, heads, mem_width, head_channel_width])
            self.ensembles = True

        nn.init.kaiming_uniform_(key)
        nn.init.kaiming_uniform_(value)

        self.QueryProj = Linear(d_model, [heads, head_channel_width], ensembles)
        self.Key = nn.Parameter(key)
        self.Value = nn.Parameter(value)
        self.DeheadProj = Linear([heads, head_channel_width], d_model, ensembles)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        :param query: A tensor to gain insight on
        :return: The calibrated result of the query.
        """

        # query : (..., (ensemble), items, embedding_width)
        if self.ensembles:
            query = query.transpose(-2, -3)  # (..., items, ensemble, embedding_width

        # Get key, value, and query prepped

        query = self.QueryProj(query)  # (..., items,  (ensemble), head,  head_embedding)
        key = self.Key  # ((ensemble),  head, mem, head_embedding)
        value = self.Value  # (ensemble), head, mem, head_embedding)

        if self.ensembles:
            query = query.transpose(-4, -2).transpose(-4, -3)  # (..., ensemble, head, items, head_embed)
        else:
            query = query.transpose(-3, -2)  # (..., head, items, head_embed)

        # Perform scoring and attention

        attn = _dot_product_attention(query, key, value)  # (..., (ensemble), head, items, head_embedding)
        if self.ensembles:
            attn = attn.transpose(-2, -3).transpose(-3, -4)  # (...,  items, ensemble, head, head_dim)
        else:
            attn = attn.transpose(-2, -3)  # (..., items, head, head_dim)
        output = self.DeheadProj(attn)  # (... items, (ensemble), output_dim)
        if self.ensembles:
            output = output.transpose(-2, -3)  # (..., ensemble, items, embed_dim)
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
                 ensembles: Optional[int] = None
                 ):
        """


        :param d_model: The embeddings width
        :param output_items: How many distinct the output should ha
        :param heads: How many heads should exist
        :param ensembles: The number of distinct ensembles.
        """

        super().__init__()
        assert d_model % heads == 0
        head_width = d_model // heads

        if ensembles is None:
            ensembles = 1
            self.ensembles = False
        else:
            self.ensembles = True

        query = torch.zeros([ensembles, heads, output_items, head_width])
        nn.init.kaiming_uniform_(query)

        self.query = nn.Parameter(query)
        self.key_projector = Linear(d_model, [heads, head_width], ensembles)
        self.value_projector = Linear(d_model, [heads, head_width], ensembles)
        self.dehead = Linear([heads, head_width], d_output, ensembles)

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        """


        :param content: (..., (ensembles), items, embeddings)
        :return:
        """
        # content: (..., (ensembles), items, embeddings)
        # Project

        if self.ensembles is False:
            content = content.unsqueeze(-3)  # (..., ensembles, items, embedding)

        content = content.transpose(-3, -2)  # (..., items, ensembles, embedding)

        query = self.query  # (ensemble, head, output_items, head_embed)
        key = self.key_projector(content).transpose(-4, -2).transpose(-4,
                                                                      -3)  # (...., ensemble, head, items, head_embedding)
        value = self.value_projector(content).transpose(-4, -2).transpose(-4,
                                                                          -3)  # (...., ensemble, head, items, head_embedding)

        attn = _dot_product_attention(query, key, value)  # (..., ensemble, head, output_items, head_embed)
        attn = attn.transpose(-4, -2).transpose(-3, -2)  # (..., output_items, ensemble, head, head_embed)
        output = self.dehead(attn)  # (..., output_items, ensemble, embedding)
        output = output.transpose(-3, -2)  # (..., ensemble, output_items, embedding)

        if self.ensembles is False:
            output = output.squeeze(-3)

        return output


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
                 mode: str = "center",
                 ensemble: Optional[int] = None):
        """

        :param d_model: The width of the embedding dimension
        :param kernel_width: How wide the local kernel shall be
        :param dilations: A specification for how to dilate each head. The length of this
            determines the number of heads
        :param  mode: controls how the padding is constructed. "forward", "center", "backward" are the
            options. forward only knows about prior words, center about words to the front and back,
            and backwards only future words.
        :param ensemble: The number of ensembles to construct.
        """
        super().__init__()
        heads = len(dilations)

        assert d_model % heads == 0, "Dilations not compatible with d_model"
        head_width = d_model // heads

        if ensemble is None:
            ensemble = 1
            self.ensemble = False
        else:
            self.ensemble = True

        self.kernel_width = kernel_width
        self.dilations = dilations
        self.mode = mode

        self.query_projector = Linear(d_model, head_width, [ensemble, heads, 1])
        self.key_projector = Linear(d_model, head_width, [ensemble, heads, kernel_width])
        self.value_projector = Linear(d_model, head_width, [ensemble, heads, kernel_width])
        self.dehead = Linear([heads, head_width], d_model, ensemble)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """


        :param tensor: The tensor to perform self attention with. Shape (..., (ensemble), item, embedding).
        :return: A tensor. The result of self attention. Shape (..., (ensemble), item, embedding)
        :raise RuntimeError: If the number of items is too small for the kernel.
        """
        # Tensor: (..., (ensemble), items, embedding)

        # Ensure ensemble
        if self.ensemble is False:
            tensor = tensor.unsqueeze(-3)  # (..., ensemble, items, embedding
        if self.kernel_width > tensor.shape[-2]:
            raise RuntimeError("Kernel requires items dim to be length %s, but received tensor of length %s" %
                               (self.kernel_width, tensor.shape[-2]))

        tensor = tensor.transpose(-2, -1)  # (..., ensemble, embedding, items)

        # Localize
        query = Glimpses.dilocal(tensor, 1, 1, self.dilations)  # (..., ensemble, embedding, head, items, 1 (local))
        content = Glimpses.dilocal(tensor, self.kernel_width, 1, self.dilations,
                                   pad_justification=self.mode)  # (... ensemble, embedding, head, items, `local`)

        # Project and setup attention, taking care to process each
        # ensemble independently.

        # (..., ensemble, embedding, head, items, local)
        # (...ensemble, local,  head, items, embedding)
        # (...ensemble, items, head, local, embedding)
        # (...items, ensemble, head, local, embedding)
        query = query.transpose(-4, -1).transpose(-4, -2).transpose(-5,
                                                                    -4)  # (..., items. ensemble, head, 1, embedding)
        content = content.transpose(-4, -1).transpose(-4, -2).transpose(-5,
                                                                        -4)  # (..., items. ensemble, head, local, embedding)

        query = self.query_projector(query)  # (..., items. ensemble, head, 1, head_embed)
        key = self.key_projector(content)  # (.., items. ensemble, head, local, head_embed)
        value = self.value_projector(content)  # (..., items. ensemble, head, local, head_embed)

        # Perform attention

        attn = _dot_product_attention(query, key, value)  # (..., items, ensemble, head, 1, head_embed)
        attn = attn.squeeze(-2)  # (..., items, ensemble, head, head_embedding)

        # Remove head and return
        outcome = self.dehead(attn)  # (..., items, ensemble, embedding
        outcome = outcome.transpose(-3, -2)  # (..., ensemble, items, embedding)
        if self.ensemble is False:
            outcome = outcome.squeeze(-3)
        return outcome


class EESA(nn.Module):
    """

    Ensemble Exchange Self Attention (EESA)

    Allows different ensembles to exchange information,
    while constraining available parameters among lower level units.
    Ensembles are only allowed to perform attention with units
    whose index are equal to or lower then themselves. This is performed
    by the attention mechanism.

    This, it is hoped, will help provide good test
    behavior by ensuring that even if one section is overfit, others
    are not. It should have the effect of increasing fine tuning
    speed as well.

    """

    def __init__(self,
                 d_model: int,
                 heads: int,
                 ensembles: int,
                 ):
        """

        :param d_model: The size of the model embeddings
        :param heads: The number of heads to use
        :param ensembles: The number of ensembles to use.
        """
        super().__init__()
        items = torch.arange(ensembles)
        raw_mask = items.unsqueeze(-1) + items.unsqueeze(-2)
        mask = raw_mask >= ensembles
        mask = mask.flip(-2)

        self.mask = mask
        self.Attn = MultiHeadedAttention(d_model, d_model, d_model, heads)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """

        :param tensor: A tensor. Of shape (..., ensemble, items, embedding)
        :return: Another tensor. Shape (..., ensemble, items, embedding)
        """
        # tensor: (..., ensemble, items, embedding)
        tensor = tensor.transpose(-2, -3)  # (..., items, ensembe, embedding)
        tensor = self.Attn(tensor, tensor, tensor, self.mask)  # (..., items, ensemble, embedding)
        tensor = tensor.transpose(-2, -3)  # (..., ensemble, items, embedding)
        return tensor


class GSPU(nn.Module):
    """
    Global Strategic Processing Unit.

    This essentially performs a PISU summary
    processses the summary by whatever arbritary logic
    desired, then uses this to generate a conditioning
    tensor for each individual word. Combined with
    add+norm, it takes care of global context.
    """

    def _straightthrough(self, tensor: torch.tensor):
        return tensor

    def __init__(self,
                 d_model: int,
                 d_summary: int,
                 summary_width: int,
                 pisu_heads: int,
                 mha_heads: int,
                 layers: Optional[List[nn.Module]] = None,
                 dropout: Optional[float] = 0.2,
                 ensembles: Optional[int] = None,
                 ):
        """

        :param d_model: The input model embedding width
        :param d_summary: How wide to make the summary embedding. Internal, and fed to the layers
        :param summary_width: How long the PISU summary will be in the item dimension.
        :param pisu_heads: How many heads the PISU process should use. Must be compatible with d_model
        :param mha_heads: How many heads the final MHA process will use.
        :param layers: Optional. layers which may lie in between the global generation and
            feedback. Without this, it is equivalent to
        :param dropout: Optional. The strengh of the dropout process during the add+layernorm
        :param ensembles: Optional. How many ensembles to make.
        """
        super().__init__()

        assert d_model % pisu_heads == 0
        assert d_model % mha_heads == 0

        if ensembles is None:
            ensembles = 1
            self.ensembles = False
        else:
            self.ensembles = True

        if layers is None:
            layers = []

        self.Dropout = nn.Dropout(dropout)
        self.PISU = PISU(d_model, d_summary, summary_width, pisu_heads, ensembles)
        self.PISU_Norm = nn.LayerNorm(d_summary)
        self.Layers = nn.ModuleList(layers)
        self.Norms = nn.ModuleList([nn.LayerNorm(d_summary) for _ in layers])
        self.Localize = MultiHeadedAttention(d_model, d_summary, d_model, mha_heads, ensembles)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """

        :param tensor: A tensor, of shape (..., (ensemble), items, embedding)
        :return:  A tensor, of shape (..., (ensemble), items, embedding)
        """

        # tensor: (..., (ensemble), items, normal_embedding)
        if self.ensembles is False:
            tensor = tensor.unsqueeze(-3)  # (..., ensemble, items, normal_embedding)

        summary = self.PISU(tensor)
        summary = self.PISU_Norm(summary)

        for layer, norm in zip(self.Layers, self.Norms):
            summary = self.Dropout(layer(summary)) + summary
            summary = norm(summary)

        output = self.Localize(tensor, summary, summary)

        if self.ensembles is False:
            output = output.squeeze(-3)
        return output
