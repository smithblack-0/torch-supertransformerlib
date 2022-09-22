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
from dataclasses import dataclass

from . import Core
from . import Attention

def is_fully_halted(halting_probability: torch.Tensor, halting_epsilon = 0.001)->bool:
    """ A small function which simply evaluates if all halting probabilities have reached a halted state"""
    return torch.all(halting_probability > 1 -halting_epsilon)

def get_unhalted_batchmesh(halting_probability: torch.Tensor):
    pass


Adaptive_Accumulator = namedtuple(
    "Adaptive_Accumulator",
    [
        "Halting_Probabilities",
        "Residuals",
        "Output"
    ]

)


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

    @torch.jit.export
    def start_accumulator(self, query, key, value)->Adaptive_Accumulator:
        """
        Given a query, key, and value starts a compatible accumulator for catching
        outputs and residuals.
        """

        halting_probabilities = torch.zeros(query.shape[:-1])
        residuals = torch.zeros(query.shape[:-1])
        output = torch.zeros(list(query.shape[:-1]) + [value.shape[-1]])

        return Adaptive_Accumulator(halting_probabilities, residuals, output)

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
        self.confidence_key_projector = Core.Linear(d_key, [heads, d_assembly], parallelization, dynamics)

        #Assembly projectors

        self.assembly_query_projector = Core.Linear(d_query, [heads, d_assembly], parallelization, dynamics)
        self.assembly_key_projector = Core.Linear(d_key, [heads, d_assembly], parallelization, dynamics)

        #Value and deheading projectors.

        self.value_projection = Core.Linear(d_value, [heads, d_head], parallelization, dynamics)
        self.deheader = Core.Linear([heads, d_head], d_query, parallelization, dynamics)

    def make_attn_heads(self, query, key, value):
        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # #(item, (dynamics), (..parallel), embedding)
        value = value.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)

        query = self.attn_query_projector(query)
        key = self.attn_key_projector(key)
        value = self.value_projection(value)

        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # #(item, (dynamics), (..parallel), embedding)
        value = value.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)

        return query, key, value

    def make_confidence_heads(self, query, key):
        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # #(item, (dynamics), (..parallel), embedding)

        query = self.confidence_query_projector(query)
        key = self.confidence_key_projector(key)

        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # #(item, (dynamics), (..parallel), embedding)

        return query, key

    def make_assembly_heads(self, query, key):
        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # #(item, (dynamics), (..parallel), embedding)

        query = self.assembly_query_projector(query)
        key = self.assembly_key_projector(key)

        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, (dynamics), (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # #(item, (dynamics), (..parallel), embedding)

        return query, key

    def forward(self,
                accumulator: Adaptive_Accumulator,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                )->Adaptive_Accumulator:

        #Generate the heads

        attn_query, attn_key, attn_value = self.make_attn_heads(query, key, value)
        confidence_query, confidence_key = self.make_confidence_heads(query, key)
        assembly_query, assembly_key = self.make_assembly_heads(query, key)

        #Begin working towards attention

        attn_logits = torch.matmul(attn_query, attn_key.transpose(-1, -2))
        confidence_logits = torch.matmul(confidence_query, confidence_key.transpose(-1, -2))
        assembly_logits = torch.matmul(assembly_query, assembly_key.transpose(-1, -2))
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask, -1e+8)
            confidence_logits = confidence_logits.masked_fill(mask, 0)
            assembly_logits = assembly_logits.masked_fill(mask, 0)

        #(..., dynamic, (par..), query, content) for all

        #Calculate the aggression and the assembly weights.
        #
        #Then use these to format the scores that will be
        #needed, clamp total probability to 1, and produce
        #residuals if clamping action occurred.

        score = torch.softmax(attn_logits, dim=-1) #(..., head, query, content)
        confidence = torch.sigmoid(confidence_logits.sum(dim=-1)).unsqueeze(-1) #(... head, query, 1)
        assembly_weights = torch.sigmoid(assembly_logits.sum(dim=-1))  #(...head, query)
        raw_halting_probability_update = score*confidence*assembly_weights.unsqueeze(-1) #(...head, query, content)
        raw_halting_probability_update = raw_halting_probability_update.sum(-3).sum(-1) #(..., query)

        raw_new_halting_probability = raw_halting_probability_update + accumulator.Halting_Probabilities
        requires_adjustment = raw_new_halting_probability > 1 - 0.001 #(..., query)
        requires_adjustment = torch.logical_and(requires_adjustment,  raw_new_halting_probability != 1.0)
        clamp_adjustment = ((1-accumulator.Halting_Probabilities)/(raw_halting_probability_update + 1e-12)) #(..., query)

        score = torch.where(
            requires_adjustment.unsqueeze(-2).unsqueeze(-1),
            clamp_adjustment.unsqueeze(-2).unsqueeze(-1)*score,
            score) #(..., head, query, content)
        residuals_update = torch.where(
            requires_adjustment,
            clamp_adjustment*raw_halting_probability_update,
            torch.tensor(0.0)) #(..., query)
        halting_probability_update = torch.where(
            requires_adjustment,
            clamp_adjustment * raw_halting_probability_update,
            raw_halting_probability_update,
        )   #(..., query)

        #Run attention. Then assemble headed return.

        attn = torch.matmul(score*confidence, value) #(..., head, query, d_head)
        attn = attn.unsqueeze(0).transpose(0, -2).squeeze(-2)
        attn = attn*assembly_weights
        output_update = self.deheader(attn)
        output_update = output_update.unsqueeze(-2).transpose(0, -2).squeeze(0)

        #Run updates. Return new accumulator

        halting_probabilities = accumulator.Halting_Probabilities + halting_probability_update
        residuals = accumulator.Residuals + residuals_update
        output = accumulator.Output + output_update

        return Adaptive_Accumulator(halting_probabilities, residuals, output)


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






