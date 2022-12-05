"""

A mechanism for performing adaptive computation time.

Adaptive computation time is quite useful to allowing
extra time when needed to think about a problem. It uses
halting probabilities to determine when a particular
element is "halted"

"""

import torch
import src.supertransformerlib.Core as Core
from torch import nn
from typing import Optional, List, Tuple


class HaltingInitException(Core.ValidationError):
    """
    An error for problems encountered when doing halting
    initialization.
    """

    def __init__(self,
                 reason: str,
                 ):
        type = "HaltingInitException"
        super().__init__(type, reason)

class BatchSpec:
    """
    A place to put specifications regarding
    what the expectations for the shape and size
    of this batch are.
    """
    def __init__(self,
                 halting_shape: Core.StandardShapeType,
                 batch_shape: Core.StandardShapeType,
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

class haltingAccumulator:
    """
    Accumulates information on the halting
    probability and can display when something is halted
    """
    @property
    def unhalted_elements(self):
        return self.halting_probability < 1 - self.epsilon
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
        self.halting_probability += halting_probabilities

    def __init__(self,
                 spec: BatchSpec,
                 epsilon: float = 0.001):
        shape_as_list: List[int] = spec.total_shape.tolist()
        self.halting_probability = torch.zeros(shape_as_list, device=spec.device, dtype=spec.dtype)
        self.epsilon = epsilon

class stateAccumulator:
    """
    Accumulates state information using the halting
    probabilities to encourage superposition.
    """
    def __init__(self, tensor: torch.Tensor):

class HaltState:
    """

    The purpose of this class is to track the cumulative
    halting probabilities, display information based on it,
    and provide modifications in the event of clamping being required.

    This code supports a flavor of adaptive computation time. Read
    that paper if supporting.

    The class is initialized by defining halting and batch shape
    when running the constructor. Optionally, a mask of halting shape
    may be provided as well.

    Once initialized, the class then has a data role, and a
    logical role to perform. The data role is to display the
    unhalted_mask, is_fully_halted, and unhalted_index fields.
    Meanwhile, the logical role is to accept halting updates,
    in one of two formats, and use that to update the data fields.
    External code is expected to read the fields to gain insight
    and perform operations.

    As it is immediately logically relevant, it is the case that
    the halting probabilities, which may have been "trimmed" producing
    residuals, are returned when an update is run. These returned
    probabilities should be utilized in further update actions.

    ---- fields ----

    * is_all_halted: True only if every coordinate tracked has reached a halted state.

    * halted_mask: A mask which is true if the corrolated coordinate has halted
    * halted_batches: A mask which addresses only the batch dimensions, and is true if
      all elements within the batch has halted

    * unhalted_mask: A mask which is true if the corrolated coordinate is not halted
    * unhalted_batches: A mask which addresses only the batch dimensions, and is true if
      all elements within the batch are not halted.

    * Residual: Collected automatically when reaching a halted state. Useful for training


    halting_shape
    batch_shape
    total_shape
    """

    @property
    def is_completely_halted(self):
        return torch.all(self.halted_mask)

    @property
    def halted_mask(self) -> torch.Tensor:
        return self.halting_probabilities >= 1 - self.epsilon

    @property
    def halted_batches(self) -> torch.Tensor:
        batch_length = self.batch_shape.shape[0]
        flat_mask = self.halted_mask.flatten(batch_length, -1)
        output = torch.all(flat_mask, dim=-1)
        return output

    @property
    def unhalted_mask(self) -> torch.Tensor:
        return torch.logical_not(self.halted_mask)

    @property
    def unhalted_batches(self) -> torch.Tensor:
        return torch.logical_not(self.halted_batches)

    def __init__(self,
                 halting_shape: Core.StandardShapeType,
                 batch_shape: Core.StandardShapeType,
                 halting_epsilon: float = 0.001,
                 halted_mask: Optional[torch.Tensor] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None):
        """

        :param halting_shape: The shape of the halting dimension.
        :param batch_shape: The shape of the batch dimension.
        :param halted_mask: Optional. Of shape [batch_shape, halting_shape].
            Manually defines each entry as either "halted" (true) or "unhalted"
            "false"
        :param dtype: The floating point dtype the halting probabilities are in
        :param device: The device to build the class on.
        """

        if dtype is None:
            dtype = torch.float32

        halting_shape = Core.standardize_shape(halting_shape, "halting_shape")
        batch_shape = Core.standardize_shape(batch_shape, "batch_shape")
        total_shape = torch.concat([batch_shape, halting_shape])

        # Create halting probabilities containers
        #
        # This means the halting probabilities and
        # batch residual containers

        if halted_mask is not None:
            # We convert to list, then to tuple, to make torchscript happy.
            shape_as_list: List[int] = total_shape.tolist()
            shape_as_tuple = torch.Size(shape_as_list)
            if shape_as_tuple != halted_mask.shape:
                reason = f"""\
                It is the case that the halted_mask and the
                total shape should be the same. However, it was found
                that the halted_ mask had shape {halted_mask.shape} while
                the total shape was {total_shape}
                """
                reason = Core.dedent(reason)
                raise HaltingInitException(reason)

            halting_probabilities = halted_mask.to(dtype=dtype, device=device)
            residuals = torch.zeros(batch_shape, dtype=dtype, device=device)
        else:
            halting_probabilities = torch.zeros(total_shape, dtype=dtype, device=device)
            residuals = torch.zeros(batch_shape, dtype=dtype, device=device)

        self.halting_shape = halting_shape
        self.epsilon = halting_epsilon
        self.batch_shape = batch_shape
        self.total_shape = total_shape
        self.halting_probabilities = halting_probabilities
        self.residuals = residuals

