"""

A mechanism for performing adaptive computation time.

Generally, the point behind these entities is that
they will keep haltable elements syncronized.

"""

import torch
import src.supertransformerlib.Core as Core
from torch import nn
from typing import Optional, List, Tuple


class ACT_Exception(Core.ValidationError):
    """
    An error for problems encountered when doing halting
    initialization.
    """

    def __init__(self,
                 reason: str,
                 ):
        type = "ACT_Exception"
        super().__init__(type, reason)

class BatchSpec:
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
    def __init__(self, spec: BatchSpec):
        shape_as_list: List[int] = spec.total_shape.tolist()
        self.spec = spec
        self.remainder = torch.zeros(shape_as_list, device=spec.device, dtype=spec.dtype)
        self.count = torch.zeros(shape_as_list, device=spec.device, dtype=torch.int64)

class haltingLattice:
    """
    Concetually the haltingLattice tracks a lattice point
    for each element on each batch. These latticepoints
    represent the current halting probability. This probability
    can be updated, the unhalted elements may be checked, and other
    operations related to the halting state for the latticepoints
    exist.

    * It clamps the probabilities such that the total probability will not go above one
      when requested. This will also spit out remainders.
    * When it's update feature is called, it updates the internal probabilities
    """
    @property
    def unhalted_elements(self)->torch.Tensor:
        return self.halting_probability < 1 - self.epsilon
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

    def clamp_probabilities(self, halting_probabilities: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Clamps the incoming halting probabilities to not add up above
        1.0. Returns the updated probabilities, and the residuals
        generated when performing this clamping
        :param halting_probabilities: The computated halting probabilities, with one
            probability for each element of the spec.
        :returns: **clamped probabilities**, **residuals**
        """
        if halting_probabilities.shape != self.halting_probability.shape:
            reason = f"""\
            The batchspec was defined to have total shape 
            {self.halting_probability.shape}. 
            However, passed parameter halting_probabilities
            had shape {halting_probabilities.shape}
            """
            reason = Core.dedent(reason)
            raise ACT_Exception(reason)

        probability_needs_remainder = halting_probabilities + self.halting_probability >= 1 - self.epsilon
        remainder = 1 - self.halting_probability #TODO: Should we stop calculating after one remainder return?
        clamped_probabilities = torch.where(probability_needs_remainder,
                                            remainder,
                                            halting_probabilities)
        remainder_output = torch.where(probability_needs_remainder, remainder, 0)

        return clamped_probabilities, remainder_output

    def update(self, halting_probabilities: torch.Tensor):
        """
        Updates the probabilities based on the given halting
        probability. Returns the probabilities, which may in turn
        be clamped.

        :param halting_probabilities: The halting probabilities
        :return: True if all elements are halted. False otherwise.
        """
        if halting_probabilities.shape != self.halting_probability.shape:
            reason = f"""\
            The batchspec was defined to have total shape 
            {self.halting_probability.shape}. 
            However, passed parameter halting_probabilities
            had shape {halting_probabilities.shape}
            """
            reason = Core.dedent(reason)
            raise ACT_Exception(reason)

        self.halting_probability += halting_probabilities

    def __init__(self,
                 spec: BatchSpec,
                 epsilon: float = 0.001):

        assert epsilon >= 0
        shape_as_list: List[int] = spec.total_shape.tolist()
        self.halting_probability = torch.zeros(shape_as_list, device=spec.device, dtype=spec.dtype)
        self.epsilon = epsilon
        self.spec = spec
    def __call__(self)->torch.Tensor:
        return self.halting_probability

class stateAccumulator:
    """
    Accumulates state information matching a pattern
    defined by the spec. A shape of zero will exactly
    match the spec 1-1. A shape of, for example, 10
    will place an embedding dimension of length ten
    onto the accumulator.

    The accumulator starts with a tensor of zeros. Then,
    every time it is updated, it adds the update into the
    tensor.
    """
    def __init__(self,
                 spec: BatchSpec,
                 shape: Core.StandardShapeType):
        shape = Core.standardize_shape(shape, "shape", allow_zeros=True)
        shape_as_list: List[int] = spec.total_shape.tolist()
        if shape == 0:
            state = torch.zeros(shape_as_list, device=spec.device, dtype=spec.dtype)
        else:
            update_as_list: List[int] = shape.tolist()
            shape_as_list = shape_as_list + update_as_list
            state = torch.zeros(shape_as_list, device=spec.device, dtype=spec.dtype)
        self.state = state
    def update(self, tensor: torch.Tensor):
        self.state += tensor

