"""

A collection of functions designed to
aid in the design of more intelligent
algorithms.

"""
import math

"""
Design:

* Two domains: The state and the instance. Have different dimensions. 
* State domain: 
*       Generated as we travel along. 
*       Collective across word embedding and across generation instances.
*       Combined with instance domain for compound domain
* Instance Domain:
*       The actual output we are looking for
*       Consists of a superposition of things from the state domain
*       Outputs and residuals generally end up here
*       Combined with state domain for compound domain.
* Compound Domain:
*       Probabilities indicating how to put state domain together to make something in instance domain
*       Also used to perform state updates sometimes.



"""
import dataclasses
from typing import Optional, Union, List, Tuple

import torch
from torch import nn
from collections import namedtuple

from . import Core
from . import Attention

@torch.jit.script
class Adaptive_Map():
    """'
    A small class capable of mapping from and to unhalted
    space efficiency. This improves calculation efficiency

    It bases all calculations on the halting probabilities
    tensor it is fed - probability 1.0 means halted.

    Two functions exist. These are restrict, and update.
    Restrict ensures that as efficient a batch as possible
    is fed to

    '"""
    def restrict(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        When restricting, it restricts the output to
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

        #Expand mask to match shape
        mask = self.mask
        tensor = tensor.clone()
        tensor = self._flatten_batch(tensor)
        if mask.dim() < len(self.shape):
            #Figure out what the expansion shape is,
            #make extra dimensions, then expand mask
            #using views.
            shape = [-1]*mask.dim()
            shape += list(tensor.shape[mask.dim():])
            while mask.dim() < len(shape):
                mask = mask.unsqueeze(-1)
            mask = mask.expand(shape)

        masked_update = torch.where(mask, update, tensor[self.index])
        tensor[self.index] = masked_update
        tensor = self._unflatten_batch(tensor)
        return tensor

    def _flatten_batch(self, tensor: torch.Tensor)->torch.Tensor:
        """Flattens all batch dimensions"""
        return tensor.flatten(0, self.batch_dim)

    def _unflatten_batch(self, tensor: torch.Tensor)-> torch.Tensor:
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

        #Figure out the number of batch dimensions.
        batch_dims = halting_probabilities.dim() - 2

        # Figure out which batches are unhalted. Do this by
        # collecting all the data dimensions together into one row,
        # then asking if they have all halted.

        unhalted_batches = halting_probabilities < 1 - 0.001  # (... [data...])
        unhalted_batches = unhalted_batches.flatten(-1, -1)
        unhalted_batches = torch.any(unhalted_batches, dim=-1)  # ([batch...])
        unhalted_batches = unhalted_batches.flatten()
        index = torch.arange(unhalted_batches.shape[0], device=unhalted_batches.device).masked_select(unhalted_batches)

        #Generate the mask.

        mask = halting_probabilities.flatten(0, batch_dims)
        mask = mask[index]
        mask = mask < 1 - 0.001

        #Store away

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
class Adaptive_Translator():
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
        return torch.all(self.Halting_Probabilities >= 1 - 0.001)

    @staticmethod
    def start_buffer(word_embeddings: torch.Tensor, embedding_length: Optional[int] = None)->"Adaptive_Translator":
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
        return Adaptive_Translator(halting_probabilities, residuals, output)

    def get_from_tensor(self, tensor: torch.Tensor)->torch.Tensor:
        """
        Restricts a tensor of batch-query shape to be of exactly
        the same shape as the unhalted batches. Unused elements are
        just excluded.

        :param tensor: A tensor in [...batch, query, ..other] shape
        :return: A reshaped tensor, in [flatbatch, query, ...other] shape
        """
        return self.Map.restrict(tensor)
    def set_to_tensor(self, tensor: torch.Tensor, update: torch.Tensor)->torch.Tensor:
        """
        Accepts the original tensor and a unhalted-restricted tensor,
        then updates the original with the update respecting masks.

        :param tensor: An original tensor
        :param update: An update developed from the original tensor
        :return: An updated tensor
        """
        return self.Map.update(tensor, update)

    def get_subaccumulator(self)->Subaccumulator:
        """Gets a subaccumulator which contains the unhalted entries."""
        halting_probs = self.Map.restrict(self.Halting_Probabilities)
        residuals = self.Map.restrict(self.Residuals)
        output = self.Map.restrict(self.Output)
        return Subaccumulator(halting_probs, residuals, output)

    def set_from_subaccumulator(self, update: Subaccumulator):
        """Updates the accumulator with the results from a particular subbatch"""
        self.Halting_Probabilities = self.Map.update(self.Halting_Probabilities, update.Halting_Probabilities)
        self.Residuals = self.Map.update(self.Residuals, update.Residuals)
        self.Output = self.Map.update(self.Output, update.Output)
        self.Map = Adaptive_Map(self.Halting_Probabilities)

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
        return Adaptive_Translator(Halting_Probabilities, Residuals, Output)

    def __init__(self,
                Halting_Probabilities: torch.Tensor,
                Residuals: torch.Tensor,
                Output: torch.Tensor,
                 ):
        self.Map = Adaptive_Map(Halting_Probabilities)
        self.Halting_Probabilities = Halting_Probabilities
        self.Residuals = Residuals
        self.Output = Output



class Adaptive_Attention(Core.KernelSpace):
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

        #Attention projectors
        self.attn_query_projector = Core.Linear(d_query, [heads, d_head], parallelization, dynamics)
        self.attn_key_projector = Core.Linear(d_key, [heads, d_head], parallelization, dynamics)

        #Confidence projectors

        self.confidence_query_projector = Core.Linear(d_query, [heads, d_confidence], parallelization, dynamics)
        self.confidence_key_projector = Core.Linear(d_key, [heads, d_confidence], parallelization, dynamics)

        #Assembly projectors

        self.assembly_query_projector = Core.Linear(d_query, [heads, d_assembly], parallelization, dynamics)
        self.assembly_key_projector = Core.Linear(d_key, [heads, d_assembly], parallelization, dynamics)

        #Value and deheading projectors.

        self.value_projection = Core.Linear(d_value, [heads, d_head], parallelization, dynamics)
        self.dehead = Core.Linear([heads, d_head], d_value, parallelization, dynamics)

    def make_attn_heads(self, query, key, value):
        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # #(item, (dynamics), (..parallel), embedding)
        value = value.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)

        query = self.attn_query_projector(query)
        key = self.attn_key_projector(key)
        value = self.value_projection(value)

        query = query.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # #(item, (dynamics), (..parallel), embedding)
        value = value.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # (item, (dynamics), (..parallel), embedding)

        return query, key, value

    def make_confidence_heads(self, query, key):
        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # #(item, (dynamics), (..parallel), embedding)

        query = self.confidence_query_projector(query)
        key = self.confidence_key_projector(key)

        query = query.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # #(item, (dynamics), (..parallel), embedding)

        return query, key

    def make_assembly_heads(self, query, key):
        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # #(item, (dynamics), (..parallel), embedding)

        query = self.assembly_query_projector(query)
        key = self.assembly_key_projector(key)

        query = query.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(-2).transpose(-2, 0).squeeze(0)  # #(item, (dynamics), (..parallel), embedding)

        return query, key

    def forward(self,
                accumulator: Subaccumulator,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                )->Subaccumulator:

        #Generate the heads

        attn_query, attn_key, attn_value = self.make_attn_heads(query, key, value)
        confidence_query, confidence_key = self.make_confidence_heads(query, key)
        assembly_query, assembly_key = self.make_assembly_heads(query, key)

        #Begin working towards attention. Generate the required logits,
        #and mask. Take care to note that the assembly and confidence logit,
        #will be undergoing a later sum into a composite logit
        # and not fed directly through an activation function - as
        # such, the mask value should be 0, not negative infinity.


        attn_logits = torch.matmul(attn_query, attn_key.transpose(-1, -2))
        confidence_logits = torch.matmul(confidence_query, confidence_key.transpose(-1, -2))
        assembly_logits = torch.matmul(assembly_query, assembly_key.transpose(-1, -2))
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask, -1e+8)
            confidence_logits = confidence_logits.masked_fill(mask, 0)
            assembly_logits = assembly_logits.masked_fill(mask, 0)

        #(..., dynamic, (par..), query, content) for all



        #Develop primary calculation features. These consist of activating the previously defined
        # logits - the attention logits, the confidence logits, and the assembly logits.
        #
        # Of note, we calculate two things from the assembly logits. These are the
        # assembly weights, then the assembly probabilities. The assembly probabilities
        # exist to provide a frame of reference in which each assembly weights contributes
        # towards a probability that adds up to one. This could be done using a softmax.
        # However, it is easier for a model to turn heads on and off using sigmoids plus
        # rescaling

        score = torch.softmax(attn_logits, dim=-1) #(..., head, query, content)
        confidence = torch.sigmoid(confidence_logits.sum(dim=-1)).unsqueeze(-1) #(... head, query, 1)
        assembly_weights = torch.sigmoid(assembly_logits.sum(dim=-1))  #(...head, query)
        assembly_probabilities = assembly_weights/(assembly_weights.sum(dim=-2).unsqueeze(-2)+0.001) #(..., head, query)

        #The pieces developed above will be utilized to weight the attention which will occur
        #further down the line. This develops probabilities from these weights. First, we combine
        #all the probabilities we have to make a raw probability update, and get the update in query
        #format. Once this is done, we figure out if the score needs adjustment to keep the total
        #probability equal to one, calculate a scalar change which will make it happen, and rescale
        #the current score. The residuals and halting probility update is also calculated here.
        #
        # One thing worth mentioning is the clamp adjustment. It is the solution to the problem
        # k*sum(update)+sum(original) = 1, where k is solved for. It ensures adding a probability
        # update will not exceed 1.

        raw_halting_probability_update = score*confidence*assembly_probabilities.unsqueeze(-1) #(...head, query, content)
        raw_halting_probability_update = raw_halting_probability_update.sum(-3).sum(-1) #(..., query)

        raw_new_halting_probability = raw_halting_probability_update + accumulator.Halting_Probabilities
        requires_adjustment = raw_new_halting_probability > 1 - 0.001 #(..., query)
        requires_adjustment = torch.logical_and(requires_adjustment,  raw_new_halting_probability != 1.0) #Shuts off finished.
        clamp_adjustment = ((1-accumulator.Halting_Probabilities)/(raw_halting_probability_update + 1e-12)) #(..., query)

        score = torch.where(
            requires_adjustment.unsqueeze(-1).unsqueeze(-3),
            clamp_adjustment.unsqueeze(-1).unsqueeze(-3)*score,
            score) #(..., head, query, content)
        residuals_update = torch.where(
            requires_adjustment,
            clamp_adjustment*raw_halting_probability_update,
            torch.tensor(0.0, device=raw_halting_probability_update.device)) #(..., query)
        halting_probability_update = torch.where(
            requires_adjustment,
            clamp_adjustment * raw_halting_probability_update,
            raw_halting_probability_update,
        )   #(..., query)

        #The weird probability work is now done. Score will not overflow
        #probability. As a result, we now proceed as normal and perform
        #dot product attention. Then we weight the heads by the assembly
        #weights and get the outputs.

        attn = torch.matmul(score*confidence, attn_value) #(..., head, query, d_value)
        attn = attn/math.sqrt(query.shape[-1])

        output_update = attn.unsqueeze(0).transpose(0, -1).squeeze(-1) #(d_value, ..., head, query)
        output_update = output_update*assembly_weights #(d_value, ..., head, query)
        output_update = output_update.unsqueeze(-1).transpose(0, -1).squeeze(0) #(..., head, query, d_head)
        output_update = output_update.unsqueeze(0).transpose(0, -2).squeeze(-2)
        output_update = self.dehead(output_update)
        output_update = output_update.unsqueeze(-2).transpose(-2, 0).squeeze(0) #(..., query, d_value)

        #Run updates. Return new accumulator

        halting_probabilities = accumulator.Halting_Probabilities + halting_probability_update
        residuals = accumulator.Residuals + residuals_update
        output = accumulator.Output + output_update
        return accumulator.update(halting_probabilities, residuals, output)

# Focus utilities. These generally help minimize computational overhead
# by avoiding performing calculations where they are unneeded.


class Calculate_Trash_Update(Core.KernelSpace):
    """

    Calculates an update for the so called "trash" parameter.

    The trash consists of two things. First is the trash
    probabilities. These are probabilities associated with each
    and every state that masks their value. Second is the trash
    can. This is where embeddings end up when trashed, and
    can be used to encourage training of items back out of
    the trash.

    This class calculates an update to the trash probabilities
    based on a given state update using attention. This can
    be added to the current trash probabilities without issue.
    Appropriate care is taken to ensure trash probability cannot
    rise above 100 %.
    """

    def __init__(self,
                 d_model: int,
                 d_internal: Optional[int] = None,
                 parallelization: Optional[Union[torch.Tensor, List[int], int]] = None,
                 dynamics: Optional[int] = None):
        super().__init__()
        self.feedforward = Attention.FeedForward(d_model + d_model,
                                                 d_internal=d_internal,
                                                 parallelization=parallelization,
                                                 dynamics=dynamics)

        self.trash_logits = Core.Linear(d_model + d_model,
                                          1,
                                          parallel=parallelization,
                                          dynamics=dynamics
                                          )
        self.sigmoid = nn.Sigmoid()

    def forward(self, trash_probs: torch.Tensor, compound_state: torch.Tensor, compound_update: torch.Tensor)->\
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the trash update

        :param trash_probs: The current trash probabilities, in the compound domain
        :param compound_state: The current compound state.
        :param state_update: The current compound state update.
        :return:
        """

        #Generate the trash prediction

        size = [-1] * state.dim()
        size[-2] = state.shape[-2]
        state_update = state_update.expand(size)


        raw_trash_update = torch.concat([state, state_update], dim=-1)
        raw_trash_update = self.feedforward(raw_trash_update)
        raw_trash_update = self.trash_logits(raw_trash_update)
        raw_trash_update = self.sigmoid(raw_trash_update).squeeze(-1)

        # Note: Magic constant 1e-8 is present to prevent division by zero.
        # Note: Magic constant 0.001 exists to allow halting after one round.
        clamp_adjustment = ((1 - trash_probs.sum(dim=-2)) / (raw_trash_update.sum(dim=-2) + 1e-8)).unsqueeze(-2)
        requires_adjustment = (raw_trash_update.sum(dim=-2) + trash_probs.sum(dim=-2) > 1 - 0.001).unsqueeze(-2)

        trash_update = torch.where(
            requires_adjustment,
            clamp_adjustment * raw_trash_update,
            raw_trash_update
        )

        # Calculate the residual_update

        residual_update = torch.where(
            requires_adjustment,
            clamp_adjustment * raw_trash_update,
            0).sum(dim=-2)

        return trash_update, residual_update

#Compound state manipulation.


### Dataflow. ####

class Output_Accumulator:
    """
    A small class for accumulating output and nonpersistant
    information.
    """
    def restrict(self, map: torch.tensor):
        """
        Uses a unhalted map to generate a view of the unhalted parameters.
        Returns the new accumulator.
        """
        map = map.unbind(-1)
        output = self.output[map]
        output_probabilities = self.output_probabilities[map]
        halting_residuals = self.halting_residuals[map]
        trash_residuals = self.trash_residuals[map]
        return Output_Accumulator(output,
                                  output_probabilities,
                                  halting_residuals,
                                  trash_residuals)
    def __init__(self,
                 output: torch.Tensor,
                 output_probabilities: torch.Tensor,
                 halting_residuals: torch.Tensor,
                 trash_residuals: torch.Tensor,
                 ):

        self.output = output
        self.output_probabilities = output_probabilities
        self.halting_residuals = halting_residuals
        self.trash_residuals = trash_residuals

class State_Accumulator:
    """
    A small location for persistant details,
    such as state, to gather.
    """
    def restrict(self, map):
        """
        Uses a unhalted map to generate a view of the unhalted parameters.
        Returns the new accumulator.
        """
        map = map.unbind(-1)
        state = self.state[map]
        trash_probabilities = self.trash_probabilities[map]
        trashcan = self.trashcan[map]
        return State_Accumulator(state, trash_probabilities, trashcan)
    def __init__(self,
                 state: torch.Tensor,
                 trash_probabilities: torch.Tensor,
                 trashcan: torch.Tensor
                 ):
        self.state = state
        self.trash_probabilities = trash_probabilities
        self.trashcan = trashcan

class Communication_Accumulator:
    """
    A small class for holding intercommunication
    information
    """



class Adaptive_Stateful_Decoder(Core.KernelSpace):


    def generate_unhalted_mesh(self, halting_probs):
        """
        Develops a meshgrid index set capable of mapping the unhalted
        elements.
        """
        cumulative_probs = halting_probs.sum(dim=-1)
        unhalted = cumulative_probs < 1.0
        unhalted = unhalted.flatten()
        index = torch.arange(unhalted.shape[-1]).masked_select(unhalted)


        mesh = torch.meshgrid(*[torch.arange(elem) for elem in cumulative_probs.shape])
        mesh = torch.stack(mesh, dim=-1)
        mesh = mesh.flatten(0, -2)

        output = mesh[index, :]
        return output






