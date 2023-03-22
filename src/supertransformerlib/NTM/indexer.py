from typing import Dict, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from .. import Basics
from .. import Core

class Indexer(nn.Module):
    """
    A small class designed to make memory
    weights by means of NTM indexing using
    the appropriate parameters.

     It accepts a  control state, the memory
     tensor for context addressing, and a prior weights
     tensor for recursion. It then generates the appropriate control
     tensors from the control state and uses them to make a tensor
     of weights indicating the importance for each given memory element.

     It performs in sequence

    * Memory resetting. It will reset memory to a stored parameter default if
                        the model so desires. The default state is in turn learnable
    * Context addressing. It will see to what degree the weights should just jump to
                       a particular context piece
    * Interpolation. IT will interpolate between the prior weights and the new context
                    weights
    * Shifting. It will roll the weights kernel to the left and right according to the produced
                probabilities, then perform a weighted sum of the results
    * Sharpening. It will sharpen the result to produce a weaker or stronger softmax output.

    The return will then be a tensor of memory weights.
    """

    def create_control_tensors(self, control_state: torch.Tensor)->Dict[str, torch.Tensor]:
        """
        Creates a dictionary of the control tensors which
        we will be using within this subsection.

        """
        # perform projection then split up tensor
        control_tensors = self.control_projector(control_state)
        control_tensors = torch.split(control_tensors, self.split_lengths, dim=-1)

        # Collect results into dictionarly
        output_dictionary = {}
        for name, activation, tensor in zip(self.split_names, self.split_activations,control_tensors):
            if activation == "None":
                pass
            elif activation == "softplus":
                tensor = F.softplus(tensor)
            elif activation == "sigmoid":
                tensor = torch.sigmoid(tensor)
            elif activation == "softmax":
                tensor = F.softmax(tensor, dim=-1)
            else:
                raise ValueError("Illegal state: Not among None, softplus, sigmoid, or softmax")

            output_dictionary[name] = tensor

        # Handle special case. Sharpening logit is a exponent and should be one or
        # greater.
        output_dictionary["sharpening_logit"] = 1 + output_dictionary["sharpening_logit"]
        return output_dictionary



    def content_addressing(self,
                           memory: torch.Tensor,
                           keys: torch.Tensor,
                           strength: torch.Tensor) -> torch.Tensor:
        similarity = F.cosine_similarity(memory, keys.unsqueeze(-2), dim=-1)
        content_weights = F.softmax(similarity * strength, dim=-2)
        return content_weights

    def interpolate(self, interp_prob: torch.Tensor, content_weights: torch.Tensor,
                    prior_weights: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between the content and prior weights using the interpolation probabilities.
        :param interp_prob: A tensor of shape `(..., 1)` representing the interpolation probabilities.
        :param content_weights: A tensor of shape `(..., memory_size)` representing the content weights.
        :param prior_weights: A tensor of shape `(..., memory_size)` representing the prior weights.
        :return: A tensor of shape `(..., memory_size)` representing the new memory weights after interpolation.
        """
        # Compute new memory weights as an interpolation between the content weights and prior weights
        new_weights = interp_prob * content_weights + (1 - interp_prob) * prior_weights

        # Return the new memory weights
        return new_weights

    def shift(self, shift_prob: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Shift the memory weights vector by a given number of steps.
        :param shift_prob: A tensor of shape `(..., shift_kernel_size)` containing the shift probabilities.
        :param weights: A tensor of shape `(..., memory_size)` representing the memory weights.
        :return: A tensor of shape `(..., memory_size)` representing the shifted memory weights.
        """
        # Compute roll numbers
        shift_kernel_size = shift_prob.shape[-1]
        roll_values = torch.arange(-(shift_kernel_size // 2), shift_kernel_size // 2 + 1, device=shift_prob.device)

        # Create roll results and multiply by shift probabilities

        shift_accumulator = []
        for roll_value, shift_prob_case in zip(roll_values, shift_prob.unbind(-1)):
            rolled_case = torch.roll(weights, int(roll_value), dims=-1)
            weighted_case = rolled_case * shift_prob_case.unsqueeze(-1)
            shift_accumulator.append(weighted_case)
        return torch.stack(shift_accumulator, dim=-1).sum(dim=-1)

    def sharpen(self, sharpening: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

        sharpening = sharpening + 1
        sharp_weights = weights ** sharpening
        sharp_weights = sharp_weights / torch.sum(sharp_weights, dim=-1, keepdim=True)
        return sharp_weights

    def __init__(self,
                 memory_size: int,
                 memory_width: int,
                 control_width: int,
                 shift_kernel_width: int,
                 ensemble_shape: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None
                 ):
        """

        :param memory_size: The number of memory elements to setup
        :param memory_width: The width of the memory embeddings
        :param control_width: The width of the control tensor embeddings
        :param shift_kernel_width: The width of the shift kernel options
        :param ensemble_shape: The shape of the kernel ensemble
        :param dtype: The dtype expected
        :param device: The device expected
        """
        super().__init__()

        # Store initialization parameters

        self.memory_size = memory_size
        self.memory_width = memory_width
        self.control_width = control_width
        self.shift_kernel_width = shift_kernel_width
        self.ensemble_shape = ensemble_shape


        # Set up the layer to create the embeddings, and configure the embeddings as well

        # Define the configuration. This is a list consisting of tuples of
        # name, the activation to take, and how big this tensor
        # should be

        # Config is in terms of (name, activation, length)
        configuration = []
        configuration.append(("content_key", "None", memory_width))
        configuration.append(("content_strength", "softplus", 1))
        configuration.append(("interpolation_probability", "sigmoid", 1))
        configuration.append(("shift_probabilities", "softmax", shift_kernel_width))
        configuration.append(("sharpening_logit", "softplus", 1))

        self.split_configuration = configuration
        self.split_names = [item[0] for item in configuration]
        self.split_activations = [item[1] for item in configuration]
        self.split_lengths = [item[2] for item in configuration]

        self.control_projector = Basics.Linear(control_width,
                                               sum(self.split_lengths),
                                               ensemble_shape,
                                               dtype=dtype,
                                               device=device)

    def forward(self,
                control_state: torch.Tensor,
                memory: torch.Tensor,
                prior_weights: torch.Tensor,
                ):
        """
        :param control_state: A ... x control_width shaped tensor designed to be used
                             to create control parameters
        :param memory: A ... mem_num x memory_size external memory tensor
        :param prior_weights: A ... x mem_num tensor of prior float wieghts
        """

        assert torch.all(prior_weights >= 0)
        control_tensors = self.create_control_tensors(control_state)
        content_weights = self.content_addressing(memory,
                                                  control_tensors["content_key"],
                                                  control_tensors["content_strength"])
        weights = self.interpolate(control_tensors["interpolation_probability"],
                                   content_weights,
                                   prior_weights)
        weights = self.shift(control_tensors["shift_probabilities"],
                             weights)

        if torch.any(torch.isnan(weights)):
            raise Exception("Not a numer")
        weights = self.sharpen(control_tensors["sharpening_logit"],
                               weights)

        return weights
