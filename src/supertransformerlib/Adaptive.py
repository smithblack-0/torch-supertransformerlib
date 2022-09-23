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

Adaptive_Accumulator = namedtuple(
    "Adaptive_Accumulator",
    [
        "Halting_Probabilities",
        "Residuals",
        "Output",
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
            torch.tensor(0.0)) #(..., query)
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
        return Adaptive_Accumulator(halting_probabilities, residuals, output)

# Focus utilities. These generally help minimize computational overhead
# by avoiding performing calculations where they are unneeded.

MeshMap = namedtuple(
    "Meshmap",
    ("Mesh",
     "Mask")
)

def make_meshmap(halting_probabilities, data_width: int = 1)->MeshMap:
    """
    Makes a mapping mesh capable of selecting
    the entire halting probability tensor.

    This will be restricted downstream.



    :param halting_probabilities: The halting probabilities
    :param data_width: How wide the data dimension needs to be.
    :return: A meshmap. Indexed on last dim
    """
    mesh = torch.meshgrid(*[torch.arange(elem) for elem in halting_probabilities.shape])
    mask = torch.full_like(halting_probabilities, True, dtype=torch.bool)
    return MeshMap(mesh, mask)

def mesh_batchprune(halting_probabilities: torch.Tensor,
                    meshmap: MeshMap)->MeshMap:
    """
    A meshmap focus function. Performs the function of
    discarding any text embeddings from the map for
    which it is found the batch is fully done. Flattens
    the batch space as well.

    :param halting_probabilities: The current halting probabilities for each mesh item
    :param mesh: The current mesh.
    :return: A new mesh.
    """

    #Turn the mesh back into a tensor
    mesh = torch.stack(meshmap.Mesh, dim=-1) #(..., [data...], index)

    #Figure out which batches are unhalted. Do this by
    #collecting all the data dimensions together into one row,
    #then asking if they have all halted.

    unhalted_batches = halting_probabilities > 1 - 0.001 #(... [data...])
    unhalted_batches = unhalted_batches.flatten(-1-data_width, -1) #(..., flatdata)
    unhalted_batches = torch.any(unhalted_batches, dim=-1) #([batch...])

    #Flatten the mesh and the batch dimensions, then select from the
    #mesh only the unhalted batches.

    unhalted_batches = unhalted_batches.flatten()
    flatmesh = mesh.flatten(0,-2-data_width)
    index = torch.arange(unhalted_batches.shape[0]).masked_select(unhalted_batches)
    mesh = flatmesh.index_select(dim=0, index=index)
    mesh = mesh.unbind(-1)

    return MeshMap(mesh, meshmap.Mask[mesh], meshmap.d_dims)

def mesh_dataprune(halting_probabilities: torch.Tensor,
                    meshmap: MeshMap,
                    )->MeshMap:
    """
    Uses the halting probabilities which are known to compact
    the data dimensions as much as possible, ignoring the
    unutilized features.

    :param halting_probabilities: The current halting probabilities, as far as known
    :param meshmap: The current meshmap
    :return: A new meshmap. This one will have an active mask.
    """

    data_dimensions = meshmap.d_dims
    mesh = meshmap.Mesh
    unhalted_data = halting_probabilities < 1.0 - 0.001

    for i in range(data_dimensions):
        i=i+1

        # Figure out what the current mesh mapping should be
        max_unhalted_queries = torch.max(unhalted_data.sum(dim=-i))
        sort_order = torch.argsort(unhalted_data, dim=-i, descending=True)
        sort_order = sort_order[:, :max_unhalted_queries]

        dim_mesh = torch.meshgrid(*[torch.arange(elem) for elem in sort_order.shape])
        dim_mesh = mesh[..., -i] = sort_order

        #Update the outputs




    # Create the rest of the mesh. Reassign the mesh based on the sort order.
    # Permute the mask.

    mesh = torch.stack(mesh, dim=-1)
    mesh[..., -1] = sort_order
    mask = unhalted_mask[mesh.unbind(-1)]


def get_batch_mesh(
        halting_probabilities: torch.Tensor
        )->torch.Tensor:
        """
        Gets a mesh which can be used to
        sample a currently existing tensor of
        initial shape halting probability, and which will
        draw from it only the nonhalted batch.

        :param halting_probabilities:
        :return:
        """
        #Go create the

        #Create a mesh which matches the current situation



        #Look at the halting probabilities, and keep anything with an active query
        #in it. Once this is done, go ahead and flatten the probabilities, and
        #find indexes we can associated with each unhalted segment.

        unhalted = halting_probabilities < 1.0 -0.000001
        unhalted = torch.any(unhalted, dim = - 1)
        unhalted = unhalted.flatten()
        index = torch.arange(unhalted.shape[-1]).masked_select(unhalted)

        #Flatten the mesh down to only batch, and query. Select
        #the unhalted entries.

        mesh = mesh.flatten(0, -3)
        output = mesh[index, :, :]

        #Return the output
        return output

def get_query_meshmask(
        halting_probabilities
    )->Tuple[torch.Tensor, torch.Tensor]:

    """

    Gets using the halting probabilities in as
    compact a representation as possible the unhalted
    queries.

    This consists of identifying what is unhalted, promoting it

    :param halting_probabilities:
    :return:
        index_mesh: A mesh which will map the problem such that unhalted queries
        come first.
        mask: A mask which can multiply the mapped tensor to mask out whatever remains
        unmasked.
    """

    # Create a mask indicating what queries have already halted. Sort the mask in
    # descending order. Clip off unneeded portions.

    unhalted_mask = halting_probabilities < 1.0 - 0.001
    max_unhalted_queries = torch.max(unhalted_mask.sum(dim=-1))
    sort_order = torch.argsort(unhalted_mask, dim=-1, descending=True)
    sort_order = sort_order[:, :max_unhalted_queries]

    #Create the rest of the mesh. Reassign the mesh based on the sort order.
    #Permute the mask.

    mesh = torch.meshgrid(*[torch.arange(elem) for elem in sort_order.shape])
    mesh = torch.stack(mesh, dim=-1)
    mesh[...,-1] = sort_order
    mask = unhalted_mask[mesh.unbind(-1)]

    return mesh, mask



class Accumulator_Focus():
    """
    A small transformation class which is capable
    of transforming an accumulator to focus only on
    the unhalted batches and queries.
    """
    def restrict(self, accumulator: Adaptive_Accumulator)->Adaptive_Accumulator:
        """
        Generates a new adapative accumulator with halted batches and queries trimmed out,
        dependent on configuration

        :param accumulator: The base accumulator to work with
        :return: A revised accumulator, with as high a calculation efficiency as possible
        """
        halting_probabilities = accumulator.Halting_Probabilities
        residuals = accumulator.Residuals
        output = accumulator.Output

        #Handle batch level refinement.
        if self.restrict_batch:
            mesh = get_batch_mesh(halting_probabilities)
            halting_probabilities = halting_probabilities[mesh.unbind(-1)]
            residuals = residuals[mesh.unbind(-1)]
            output = output[mesh.unbind(-1)]
        if self.restrict_query:
            mesh, mask = get_query_meshmask(halting_probabilities)
            halting_probabilities = halting_probabilities[mesh.unbind(-1)]*mask
            residuals = residuals[mask.unbind(-1)]*mask
            output = output[mask.unbind(-1)]*mask

        return Adaptive_Accumulator(halting_probabilities, residuals, output)


    def update(self,
               accumulator: Adaptive_Accumulator,
               new_accumulator: Adaptive_Accumulator)->Adaptive_Accumulator:
        """

        Updates the original accumulator with the results of
        adaptive attention

        :param accumulator: The entire accumulator
        :param new_accumulator: The results of attention.
        :return: The entire accumulator, with updates applied.
        """
        halting_probabilities = accumulator.Halting_Probabilities.clone()
        residuals = accumulator.Residuals.clone()
        output = accumulator.Output.clone()

        mesh = torch.meshgrid(*[torch.arange(elem) for elem in halting_probabilities.shape])
        mesh = torch.stack(mesh, dim=-1)
        if self.restrict_batch:
            meshmap = get_batch_mesh(halting_probabilities)
            halting_probabilities = halting_probabilities[meshmap.unbind(-1)]
            mesh = mesh[meshmap.unbind(-1)]
        if self.restrict_query:
            meshmap, mask = get_batch_mesh(halting_probabilities)
            mesh = mesh[meshmap.unbind(-1)]
        else:
            mask = None

        if mask is not None:
            halting_update = new_accumulator.Halting_Probabilities*mask
            residual_update = new_accumulator.Residuals*mask
            output_update = new_accumulator.Output*mask
        else:
            halting_update = new_accumulator.Halting_Probabilities
            residual_update = new_accumulator.Residuals
            output_update = new_accumulator.Output

        halting_probabilities[mesh.unbind(-1)] = halting_update
        residuals[mesh.unbind(-1)] = residual_update
        output[mesh.unbind(-1)] = output_update

        return Adaptive_Accumulator(halting_probabilities, residuals, output)


    def __init__(self,
                 restrict_halted_batches: bool = True,
                 restrict_halted_queries: bool = True,
                 ):
        self.restrict_batch = restrict_halted_batches
        self.restrict_query = restrict_halted_queries

class Query_Focus():
    """
    A small transformation package which will focus
    a given query tensor backed by an accumulator
    onto as small a batch as I can.
   """

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






