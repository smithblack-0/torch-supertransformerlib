"""

A mechanism for performing adaptive computation time.

Generally, the point behind these entities is that
they will keep haltable elements syncronized.

"""

import torch
import src.supertransformerlib.Core as Core
from torch import nn
from typing import Optional, List, Tuple


class Accumulator_Exception(Core.ValidationError):
    """
    An error for problems encountered when doing halting
    initialization.
    """

    def __init__(self,
                 reason: str,
                 ):
        type = "ACT_Exception"
        super().__init__(type, reason)

class latticeSpec:
    """
    A place to put specifications regarding
    what the expectations for the shape and size
    of this batch are.
    """
    def __init__(self,
                 batch_shape: Core.StandardShapeType,
                 halting_shape: Core.StandardShapeType,
                 device: torch.device = None,
                 dtype: torch.dtype = None):

        self.halting_shape = Core.standardize_shape(halting_shape, "halting_shape")
        self.batch_shape = Core.standardize_shape(batch_shape, "batch_shape")
        self.total_shape = torch.concat([self.batch_shape, self.halting_shape])
        self.device = device
        self.dtype = dtype
class ponderAccumulator:
    """
    Accumulates information on the remainders and ponder
    cost utilizable for optimization.
    """
    @property
    def ponder_cost(self)->torch.Tensor:
        """
        Returns the current ponder cost for each batch
        """
        batch_dims = self.spec.batch_shape.shape[0]
        element_ponder_cost = self.count + self.remainder
        batch_ponder_cost = element_ponder_cost.flatten(batch_dims).sum(dim=-1)
        return batch_ponder_cost

    def update(self, remainder: torch.Tensor):
        self.remainder += remainder
        self.count += (self.remainder == 0)
    def __init__(self, spec: latticeSpec):
        shape_as_list: List[int] = spec.total_shape.tolist()
        self.spec = spec
        self.remainder = torch.zeros(shape_as_list, device=spec.device, dtype=spec.dtype)
        self.count = torch.zeros(shape_as_list, device=spec.device, dtype=torch.int64)

class haltingLatticeAccumulator:
    """
    ----- Introduction ----
    The class conceptually syncs itself to the defined lattice
    and sets up a probability container for each lattice point.
    Updates can then increase or, depending on the mode, decrease
    the amount of probability in each container. Once a container
    is at a probability of 100% it is said to be "halted" which
    affects the mapping behavior. Once all containers are fully halted,
    the "is_halted" field returns true.

    Probability updates are provided as a delta probability, and should be
    in [0,1] in ACT mode or [-1, 1] in filtration mode. To ensure total
    probability updates do not go above 100 % or below 0 %, it is smart
    to use the clamp_probabilities method. It also optionally returns a remainder,
    for usage with ACT's ponder penalty.


    ------ modes ----

    Two modes exist. These are

    * ACT mode
    * Filtration mode

    In ACT mode a model is expected to contribute positive delta probability towards halting.
    Thus, in this mode a model thus interacts with this class by deciding primarily whether to
    further increment the halting probability of an element. The clamping method restricts outputs
    to be in the range [0, 1] or smaller.

    In Filtration mode, a model is expected to inform the class with a delta probability
    how done it is with an element. Ideal updates range in a value from [-1, 1]. It is the
    case that negative probabilities will decrease the degree of halting, allowing the model
    to go back on a decision. Additionally, the method "get_filtration_penalty" becomes useful.

    "get_filtration_penalty" will, when provided with the rank to multiply with, return a tensor
    which can be broadcast with a lattice data tensor to partially mask out the data. The more
    filtered the element, the more heavy the mask. This allows for a trainable mechanism for
    throwing out junk or useless elements.

    """
    @property
    def unhalted_elements(self)->torch.Tensor:
        return self.lattice_halting_probability < 1 - self.epsilon
    @property
    def unhalted_batches(self)->torch.Tensor:
        batch_length = self.spec.batch_shape.shape[0]
        flattened_elements = self.unhalted_elements.flatten(batch_length)
        return torch.any(flattened_elements, dim=-1)
    @property
    def is_halted(self)->bool:
        if self.unhalted_elements.sum() == 0:
            return True
        return False
    def get_penalty(self, rank: int)->torch.Tensor:
        """
        Gets a  penalty which can be multiplied
        by a lattice tensor of the given shape. In
        the case of "ACT" mode, this will return
        a mask that ignores the halted elements. This can
        easily be used to mask out finished channels if
        desired.

        In filtration mode, it returns a floating tensor
        which represents how filtered out each entry is

        :param rank: The rank of the tensor to penalize
        :return: A compatible lattice penalty tensor.
        """
        if self.mode == "Filtration":

            assert rank >= self.spec.total_shape.shape[0]
            filtration_penalty = 1 - self.lattice_halting_probability
            while filtration_penalty.dim() < rank:
                filtration_penalty = filtration_penalty.unsqueeze(-1)
            return filtration_penalty
        else:
            return self.unhalted_elements
    def map_tensor_into_sparseElements_space(self, tensor: torch.Tensor)->torch.Tensor:
        """
        This method maps the provided tensor, which should correspond with
        the defined lattice space, into a flat tensor consisting only of
        the active, unhalted elements.

        It should be used to transform entities well defined in lattice
        space into entities which interact across the now singular batchlike
        dimension.

        :param tensor: A tensor corrosponding to [...latticespec, whatever...]
        :return: A tensor of shape [U, whatever...], where U is the number of unhalted elements
        """

        shape_as_list: List[int] = self.spec.total_shape.tolist()
        shape = torch.Size(shape_as_list)
        if tensor.shape[:len(shape_as_list)] != shape:
            reason = f"""\
            It is the case that the provided parameter "tensor" was expected
            to have initial dimension specifications {shape}. However,
            received a tensor of shape {tensor.shape}
            """
            reason = Core.dedent(reason)
            raise Accumulator_Exception(reason)
        return tensor[self.unhalted_elements]
    def map_tensor_into_lattice_space(self, tensor: torch.Tensor)->torch.Tensor:
        """
        This method maps the provided tensor, which should corrospond
        to a tensor in efficency space, back into lattice space. This
        means treating it like it is sparse, and filling in zeros where
        currently halted/filtered.

        :param tensor: A tensor of shape [U, whatever...] where U is the number of unhalted elements
        :return: A tensor of shape [...laticespec, whatever...], where the elements of tensor
                 have been scattered reversably back into their original lattice points.
        """

        if tensor.shape[0] != self.unhalted_elements.sum():
            reason = f"""\
            It was expected that a number of unhalted elements equal to
            {self.unhalted_elements.sum()} would be provided on the first
            tensor dimension. However, the first dimension of parameter "tensor"
            had length {tensor.shape[0]} which does not match.
            """
            reason = Core.dedent(reason)
            raise Accumulator_Exception(reason)

        shape_as_list: List[int] = self.spec.total_shape.tolist() + list(tensor.shape[1:])
        output = torch.zeros(shape_as_list, device=tensor.device, dtype=tensor.dtype)
        output[self.unhalted_elements] = tensor
        return output


    def clamp_probabilities(self, delta_halting_probabilities: torch.Tensor)->\
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Clamps the incoming halting probabilities to be between [0.0 and 1.0]. Returns the updated probabilities, and the residuals
        generated when performing this clamping
        :param delta_halting_probabilities: The computated halting probabilities, with one
            probability for each element of the spec.
        :returns: **clamped probabilities**, **residuals**
        """
        if delta_halting_probabilities.shape != self.lattice_halting_probability.shape:
            reason = f"""\
            The latticespec was defined to have total shape 
            {self.lattice_halting_probability.shape}. 
            However, passed parameter halting_probabilities
            had shape {delta_halting_probabilities.shape}
            """
            reason = Core.dedent(reason)
            raise Accumulator_Exception(reason)

        probability_needs_top_clamp = delta_halting_probabilities + self.lattice_halting_probability >= 1 - self.epsilon
        remainder = 1 - self.lattice_halting_probability
        clamped_probabilities = torch.where(probability_needs_top_clamp,
                                            remainder,
                                            delta_halting_probabilities)

        remainder_output = torch.where(probability_needs_top_clamp, remainder, 0)
        if self.mode == "Filtration":
            probability_needs_bottom_clamp = delta_halting_probabilities + self.lattice_halting_probability < 0
            clamped_probabilities = torch.where(probability_needs_bottom_clamp,
                                                self.lattice_halting_probability,
                                                clamped_probabilities)
        else:
            clamped_probabilities = torch.relu(clamped_probabilities)


        return clamped_probabilities, remainder_output
    def update(self, halting_probabilities: torch.Tensor):
        """
        Updates the probabilities based on the given halting
        probability. Returns the probabilities, which may in turn
        be clamped.

        :param halting_probabilities: The halting probabilities
        :return: True if all elements are halted. False otherwise.
        """
        if halting_probabilities.shape != self.lattice_halting_probability.shape:
            reason = f"""\
            The latticespec was defined to have total shape 
            {self.lattice_halting_probability.shape}. 
            However, passed parameter halting_probabilities
            had shape {halting_probabilities.shape}
            """
            reason = Core.dedent(reason)
            raise Accumulator_Exception(reason)

        self.lattice_halting_probability += halting_probabilities

    def __init__(self,
                 spec: latticeSpec,
                 epsilon: float = 0.001,
                 mode: str = "ACT",):
        """
        :param spec: The lattice spec to build to
        :param epsilon: The halting epsilon. A small threshold seen in the ACT paper
        :param mode: Two options. "ACT" and "Filtration".
                     ACT mode expects the clamp_probabilities results to be between 0 and 1.
                     Filtration mode expects it to be in [-1, 1]

        """

        valid_modes = ["ACT", "Filtration"]
        Core.validate_string_in_options(mode, "mode", valid_modes, "configuration_options")
        assert epsilon >= 0

        shape_as_list: List[int] = spec.total_shape.tolist()
        self.lattice_halting_probability = torch.zeros(shape_as_list, device=spec.device, dtype=spec.dtype)
        self.epsilon = epsilon
        self.spec = spec
        self.mode = mode

    def __call__(self)->torch.Tensor:
        """Returns the current halting probabilities"""
        return self.lattice_halting_probability

