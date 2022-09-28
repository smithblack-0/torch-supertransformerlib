"""

A collection of functions designed to
aid in the design of more intelligent
algorithms.

"""
import math
from typing import Optional, Union, List

import torch
from torch import nn
from . import Core


@torch.jit.script
class AdaptiveMap:
    """'
    A small class capable of mapping from and to unhalted
    space efficiency. This improves calculation efficiency

    It bases all calculations on the halting probabilities
    tensor it is fed - probability 1.0 means halted.

    Two functions exist. A forward transformation restricting
    the tensor to the unhalted domain, and an update function.

    '"""

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        When transforming, it restricts the output to
        a flat batch of only unhalted channels.

        :param tensor: Tensor to map into restricted space. Should be shape (...batch, query, Qptional[...more])
        :return: The tensor, mapped into restricted space. Will look like (flatbatch, query, Optional[...more])
        """
        tensor = self._flatten_batch(tensor)
        return tensor[self.index]

    def update(self, tensor: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """
        The update map. Takes a result from performing an
        update and merges it into the accumulator.

        :param tensor: The original tensor
        :param update: The updated tensor
        :return: The updated tensor.
        """

        # Expand mask to match shape
        mask = self.mask
        tensor = tensor.clone()
        tensor = self._flatten_batch(tensor)
        if mask.dim() < len(self.shape):
            # Figure out what the expansion shape is,
            # make extra dimensions, then expand mask
            # using views.
            shape = [-1] * mask.dim()
            shape += list(tensor.shape[mask.dim():])
            while mask.dim() < len(shape):
                mask = mask.unsqueeze(-1)
            mask = mask.expand(shape)

        masked_update = torch.where(mask, update, tensor[self.index])
        tensor[self.index] = masked_update
        tensor = self._unflatten_batch(tensor)
        return tensor

    def _flatten_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Flattens all batch dimensions"""
        return tensor.flatten(0, self.batch_dim)

    def _unflatten_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Unflatten all batch dimensions. """
        shape = list(self.shape) + list(tensor.shape[2:])
        shape = torch.Size(shape)
        return tensor.view(shape)

    def __init__(self,
                 halting_probabilities: torch.Tensor,
                 ):
        """

        :param halting_probabilities: A tensor
        """

        # Figure out the number of batch dimensions.
        batch_dims = halting_probabilities.dim() - 2

        # Figure out which batches are unhalted. Do this by
        # collecting all the data dimensions together into one row,
        # then asking if they have all halted.

        unhalted_batches = halting_probabilities < 1 - 0.001  # (... [data...])
        unhalted_batches = unhalted_batches.flatten(-1, -1)
        unhalted_batches = torch.any(unhalted_batches, dim=-1)  # ([batch...])
        unhalted_batches = unhalted_batches.flatten()
        index = torch.arange(unhalted_batches.shape[0], device=unhalted_batches.device).masked_select(unhalted_batches)

        # Generate the mask.

        mask = halting_probabilities.flatten(0, batch_dims)
        mask = mask[index]
        mask = mask < 1 - 0.001

        # Store away

        self.batch_dim = batch_dims
        self.shape = halting_probabilities.shape
        self.index = index
        self.mask = mask


@torch.jit.script
class Subaccumulator:
    """
    A subset of the entire batch, containing
    elements which are not yet halted.
    """

    def update(self,
               Halting_Probabilities: Optional[torch.Tensor] = None,
               Residuals: Optional[torch.Tensor] = None,
               Output: Optional[torch.Tensor] = None,
               ):
        """
        Updates the given entries with the new values. Anything not given stays the same.
        """
        if Halting_Probabilities is None:
            Halting_Probabilities = self.Halting_Probabilities
        if Residuals is None:
            Residuals = self.Residuals
        if Output is None:
            Output = self.Output
        return Subaccumulator(Halting_Probabilities, Residuals, Output)

    def __init__(self,
                 Halting_Probabilities: torch.Tensor,
                 Residuals: torch.Tensor,
                 Output: torch.Tensor,
                 ):
        self.Halting_Probabilities = Halting_Probabilities
        self.Residuals = Residuals
        self.Output = Output


@torch.jit.script
class AdaptiveTranslator():
    """
    A buffer and a map, the class keepts track
    of what batches and queries are and are not
    halted and performs translation between
    batch and unhalted domain.

    The batch domaim consists of everything being provided, while
    the unhalted domain consists of only those entries which are
    not batched.

    A buffer is created on start, holding the requirements
    for performing adaptive halting, and from which information
    can be drawn for a round. Alternatively, a tensor can be the
    draw source for mapping so long as it is the case the tensor
    matches the buffer shape.

    After processing, the update functions will then update
    the original tensor or buffer, while carrying over halted
    channels instead.
    """

    def is_done(self):
        """Check if everything is fully halted."""
        return torch.all(self.Halting_Probabilities >= 1 - 0.001)

    def get_map(self) -> AdaptiveMap:
        """Gets the tensor mapping associated with the current halting probabilities"""
        return self._Map

    @staticmethod
    def start_buffer(word_embeddings: torch.Tensor, embedding_length: Optional[int] = None) -> "AdaptiveTranslator":
        """Starts a buffer. If the output dim is none, it is expected the
        embed dim and output dim are the same
        :param word_embeddings: A word embedding tensor. Used to find out shapes
        :param embedding_length: Optional. An embedding length. If the output embedding is different
        than the input embedding this may be set. Else, leave alone.
        """

        halting_probabilities = torch.zeros(word_embeddings.shape[:-1], device=word_embeddings.device)
        residuals = torch.zeros(word_embeddings.shape[:-1], device=word_embeddings.device)
        if embedding_length is not None:
            shape = list(word_embeddings.shape[:-1]) + [embedding_length]
        else:
            shape = list(word_embeddings.shape)
        output = torch.zeros(shape, device=word_embeddings.device)
        return AdaptiveTranslator(halting_probabilities, residuals, output)

    def get_subbatch(self) -> Subaccumulator:
        """Gets a subaccumulator which contains the unhalted entries."""
        halting_probs = self._Map.transform(self.Halting_Probabilities)
        residuals = self._Map.transform(self.Residuals)
        output = self._Map.transform(self.Output)
        return Subaccumulator(halting_probs, residuals, output)

    def update_buffer(self, update: Subaccumulator):
        """Updates the accumulator with the results from a particular subbatch"""
        self.Halting_Probabilities = self._Map.update(self.Halting_Probabilities, update.Halting_Probabilities)
        self.Residuals = self._Map.update(self.Residuals, update.Residuals)
        self.Output = self._Map.update(self.Output, update.Output)
        self._Map = AdaptiveMap(self.Halting_Probabilities)

    def _manual_update(self,
                       Halting_Probabilities: Optional[torch.Tensor] = None,
                       Residuals: Optional[torch.Tensor] = None,
                       Output: Optional[torch.Tensor] = None,
                       ):
        """
        Updates the given entries with the new values. Anything not given stays the same.
        """
        if Halting_Probabilities is None:
            Halting_Probabilities = self.Halting_Probabilities
        if Residuals is None:
            Residuals = self.Residuals
        if Output is None:
            Output = self.Output
        return AdaptiveTranslator(Halting_Probabilities, Residuals, Output)

    def __init__(self,
                 Halting_Probabilities: torch.Tensor,
                 Residuals: torch.Tensor,
                 Output: torch.Tensor,
                 ):
        self._Map = AdaptiveMap(Halting_Probabilities)
        self.Halting_Probabilities = Halting_Probabilities
        self.Residuals = Residuals
        self.Output = Output


class AdaptiveAttention(nn.Module):
    """
    A very special variety of attention mechanism, this
    works together with halting probabilities to
    ensure that a particular loop only runs a certain number
    of times. The pattern used for this is similar to the one
    in ACT.

    The halting transformer effectively performs transformer with the query
    and content as in a standard transformer. However, it is the case that the
    score percentages, used to assemble the output, are adjusted by a sigmoid
    unit, and the probabilities are kept around between iterations. When the
    total probability exceeds one, the probability is clamped back down to one,
    and certain extra functions can detect that the particular channel is "halted"

    The returns provided are something somewhat special - they are updates.
    The output, the residuals, and the halting probabilities are designed to
    be added to an existing accumulation tensor and the added entity may be
    provided next round.
    """

    def __init__(self,
                 d_query: int,
                 d_key: int,
                 d_value: int,
                 d_confidence: int,
                 d_assembly: int,
                 heads: int = 4,
                 parallelization: Optional[Union[torch.Tensor, List[int], int]] = None,
                 dynamics: Optional[int] = None):
        """
        :param d_query: The query embedding width
        :param d_key: The key embedding width
        :param d_value: The value embedding width
        :param d_confidence: Internal. How much embedding to dedicate to confidence
        :param d_assembly: Internal. How much embedding to dedicate to assembly
        :param heads: The number of heads.
        :param subheads: The number of subheads.
        :param parallelization:
        :param dynamics:
        """

        super().__init__()

        d_head = d_query // heads

        # Attention projectors
        self.attn_query_projector = Core.Linear(d_query, [heads, d_head], parallelization)
        self.attn_key_projector = Core.Linear(d_key, [heads, d_head], parallelization)

        # Confidence projectors

        self.confidence_query_projector = Core.Linear(d_query, [heads, d_confidence], parallelization)
        self.confidence_key_projector = Core.Linear(d_key, [heads, d_confidence], parallelization)

        # Assembly projectors

        self.assembly_query_projector = Core.Linear(d_query, [heads, d_assembly], parallelization)
        self.assembly_key_projector = Core.Linear(d_key, [heads, d_assembly], parallelization)

        # Value and deheading projectors.

        self.value_projection = Core.Linear(d_value, [heads, d_head], parallelization)
        self.dehead = Core.Linear([heads, d_head], d_value, parallelization)

    def make_attn_heads(self, query, key, value):
        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item,, (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # #(item, , (..parallel), embedding)
        value = value.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item,..., (..parallel), embedding)

        query = self.attn_query_projector(query)
        key = self.attn_key_projector(key)
        value = self.value_projection(value)

        query = query.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # (item,... (..parallel), embedding)
        key = key.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # #(item, ..., (..parallel), embedding)
        value = value.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # (item, ...., (..parallel), embedding)

        return query, key, value

    def make_confidence_heads(self, query, key):
        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, ..., (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # #(item, ..., (..parallel), embedding)

        query = self.confidence_query_projector(query)
        key = self.confidence_key_projector(key)

        query = query.unsqueeze(-2).transpose(-2, 0).squeeze(0)
        key = key.unsqueeze(-2).transpose(-2, 0).squeeze(0)

        return query, key

    def make_assembly_heads(self, query, key):
        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, ...), (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # #(item, ..., (..parallel), embedding)

        query = self.assembly_query_projector(query)
        key = self.assembly_key_projector(key)

        query = query.unsqueeze(-2).transpose(-2, 0).squeeze(0)
        key = key.unsqueeze(-2).transpose(-2, 0).squeeze(0)

        return query, key

    def forward(self,
                accumulator: Subaccumulator,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                ) -> Subaccumulator:
        # Generate the heads

        attn_query, attn_key, attn_value = self.make_attn_heads(query, key, value)
        confidence_query, confidence_key = self.make_confidence_heads(query, key)
        assembly_query, assembly_key = self.make_assembly_heads(query, key)

        # Begin working towards attention. Generate the required logits,
        # and mask. Take care to note that the assembly and confidence logit,
        # will be undergoing a later sum into a composite logit
        # and not fed directly through an activation function - as
        # such, the mask value should be 0, not negative infinity.

        attn_logits = torch.matmul(attn_query, attn_key.transpose(-1, -2))
        confidence_logits = torch.matmul(confidence_query, confidence_key.transpose(-1, -2))
        assembly_logits = torch.matmul(assembly_query, assembly_key.transpose(-1, -2))
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask, -1e+8)
            confidence_logits = confidence_logits.masked_fill(mask, 0)
            assembly_logits = assembly_logits.masked_fill(mask, 0)

        # (..., dynamic, (par..), query, content) for all

        # Develop primary calculation features. These consist of activating the previously defined
        # logits - the attention logits, the confidence logits, and the assembly logits.
        #
        # Of note, we calculate two things from the assembly logits. These are the
        # assembly weights, then the assembly probabilities. The assembly probabilities
        # exist to provide a frame of reference in which each assembly weights contributes
        # towards a probability that adds up to one. This could be done using a softmax.
        # However, it is easier for a model to turn heads on and off using sigmoids plus
        # rescaling

        score = torch.softmax(attn_logits, dim=-1)  # (..., head, query, content)
        confidence = torch.sigmoid(confidence_logits.sum(dim=-1)).unsqueeze(-1)  # (... head, query, 1)
        assembly_weights = torch.sigmoid(assembly_logits.sum(dim=-1))  # (...head, query)
        assembly_probabilities = assembly_weights / (
                    assembly_weights.sum(dim=-2).unsqueeze(-2) + 0.001)  # (..., head, query)

        # The pieces developed above will be utilized to weight the attention which will occur
        # further down the line. This develops probabilities from these weights. First, we combine
        # all the probabilities we have to make a raw probability update, and get the update in query
        # format. Once this is done, we figure out if the score needs adjustment to keep the total
        # probability equal to one, calculate a scalar change which will make it happen, and rescale
        # the current score. The residuals and halting probility update is also calculated here.
        #
        # One thing worth mentioning is the clamp adjustment. It is the solution to the problem
        # k*sum(update)+sum(original) = 1, where k is solved for. It ensures adding a probability
        # update will not exceed 1.

        raw_halting_probability_update = score * confidence * assembly_probabilities.unsqueeze(
            -1)  # (...head, query, content)
        raw_halting_probability_update = raw_halting_probability_update.sum(-3).sum(-1)  # (..., query)

        raw_new_halting_probability = raw_halting_probability_update + accumulator.Halting_Probabilities
        requires_adjustment = raw_new_halting_probability > 1 - 0.001  # (..., query)
        requires_adjustment = torch.logical_and(requires_adjustment,
                                                raw_new_halting_probability != 1.0)  # Shuts off finished.
        clamp_adjustment = (
                    (1 - accumulator.Halting_Probabilities) / (raw_halting_probability_update + 1e-12))  # (..., query)

        score = torch.where(
            requires_adjustment.unsqueeze(-1).unsqueeze(-3),
            clamp_adjustment.unsqueeze(-1).unsqueeze(-3) * score,
            score)  # (..., head, query, content)
        residuals_update = torch.where(
            requires_adjustment,
            clamp_adjustment * raw_halting_probability_update,
            torch.tensor(0.0, device=raw_halting_probability_update.device))  # (..., query)
        halting_probability_update = torch.where(
            requires_adjustment,
            clamp_adjustment * raw_halting_probability_update,
            raw_halting_probability_update,
        )  # (..., query)

        # The weird probability work is now done. Score will not overflow
        # probability. As a result, we now proceed as normal and perform
        # dot product attention. Then we weight the heads by the assembly
        # weights and get the outputs.

        attn = torch.matmul(score * confidence, attn_value)  # (..., head, query, d_value)
        attn = attn / math.sqrt(query.shape[-1])

        output_update = attn.unsqueeze(0).transpose(0, -1).squeeze(-1)  # (d_value, ..., head, query)
        output_update = output_update * assembly_weights  # (d_value, ..., head, query)
        output_update = output_update.unsqueeze(-1).transpose(0, -1).squeeze(0)  # (..., head, query, d_head)
        output_update = output_update.unsqueeze(0).transpose(0, -2).squeeze(-2)
        output_update = self.dehead(output_update)
        output_update = output_update.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # (..., query, d_value)

        # Run updates. Return new accumulator

        halting_probabilities = accumulator.Halting_Probabilities + halting_probability_update
        residuals = accumulator.Residuals + residuals_update
        output = accumulator.Output + output_update
        return accumulator.update(halting_probabilities, residuals, output)
