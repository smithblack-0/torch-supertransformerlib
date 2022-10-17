"""

Code for the reshape mechanism. A version
of torch's reshape with a few extra tricks.

"""

from typing import Optional, List
import torch
from torch import nn
from src.supertransformerlib import Core

class ReshapeException(Core.ValidationError):
    """
    A error type for when reshape fails
    """

    def __init__(self, reason: str, tasks: Optional[List[str]] = None):
        error_message_type = "ReshapeException"
        super().__init__(error_message_type, reason, tasks)


def validate_sparse_reshape(tensor: torch.Tensor,
                            input_shape: torch.Tensor,
                            output_shape: torch.Tensor,
                            task: Optional[str],
                            ):
    sparse_dim = tensor.sparse_dim()
    reshape_dim = input_shape.shape[0]
    if sparse_dim < reshape_dim:
        reason = f"""\
        Sparse rank of tensor parameter insufficient for resize:
        Param 'tensor' has sparse rank {sparse_dim}. 
        However, Param 'input_shape' is {reshape_dim} units long.

        Are you trying to reshape the dense part of a hybrid tensor? That is
        not currently supported. 
        """
        reason = Core.dedent(reason)
        raise ReshapeException(reason, task)

    sparse_shape = tensor.shape[:sparse_dim]
    if sparse_shape[-reshape_dim:] != torch.Size(input_shape):
        temp_input_shape = torch.Size(input_shape)
        tensor_shape = sparse_shape[-reshape_dim:]
        reason = f"""\
        Param 'tensor' shape and param 'input_shape' shape do not match:
            The param 'tensor' has shape {tensor_shape}.
            This cannot be broadcast with {temp_input_shape}.
        """
        reason = Core.dedent(reason)
        raise ReshapeException(reason, task)

    # Now th
    if input_shape.prod() != output_shape.prod():
        temp_input_shape = torch.Size(input_shape)
        temp_output_shape = torch.Size(output_shape)
        input_shape_elements = int(input_shape.prod())
        output_shape_elements = int(output_shape.prod())
        reason = f"""\
        Reshape is impossible with unequal number of elements.
            The number of elements for 'input_shape' shaped as {temp_input_shape} is {input_shape_elements}
            However, the number of elements for 'output_shape' shaped as {temp_output_shape} is {output_shape_elements}
            These do not match.

        """
        reason = Core.dedent(reason)
        raise ReshapeException(reason, task)


def _sparse_reshape(tensor: torch.Tensor,
                    input_shape: torch.Tensor,
                    output_shape: torch.Tensor) -> torch.Tensor:
    """
    Performs a sparse reshape on a tensor to another shape.

    :param tensor: The sparse tensor to reshape
    :param shape: The shape the tensor should end up in
    :return: The reshaped sparse tensor
    """
    # Perform a sparse reshape in a torchscript compatible method
    #
    # This is done by first converting the index to be an equivalent tensor
    # in flattened strided notation, then rebuilding the index under the new
    # shape.
    #
    # This is only capable of reshaping the sparse dimensions. Dense dimensions
    # will cause errors

    if not tensor.is_coalesced():
        tensor = tensor.coalesce()


    #Develop strided representation

    sparse_dim = tensor.sparse_dim()
    sparse_shape = tensor.shape[:sparse_dim]
    dense_shape = tensor.shape[sparse_dim:]
    sparse_strides = Core.calculate_shape_strides(sparse_shape)

    indices = tensor.indices()
    values = tensor.values()

    flat_indices = indices*sparse_strides.unsqueeze(-1)
    flat_indices = flat_indices.sum(dim=0)

    #Develop proper final shape.

    broadcast_length = input_shape.shape[0]
    static_shape = torch.tensor(sparse_shape[:-broadcast_length], dtype=torch.int64)
    final_shape = torch.concat([static_shape, output_shape])
    final_strides = Core.calculate_shape_strides(final_shape)

    # Use strides to reassemble flat indices. This is a little
    # complex, so here is what is going on
    #
    # We rescale to within the proper domain using floor division, which discards any
    # unnecessary information pertaining to later index locations. Then, we go ahead
    # and use modulo to eliminate information on earlier index positions.

    indices = flat_indices.unsqueeze(0)
    indices = torch.div(indices, final_strides.unsqueeze(-1), rounding_mode="floor")
    indices = torch.remainder(indices, final_shape.unsqueeze(-1))

    entire_shape = torch.concat([final_shape, torch.tensor(dense_shape, dtype=torch.int64)])
    entire_shape = torch.Size(entire_shape)

    output = torch.sparse_coo_tensor(indices, values, size=entire_shape)
    if not output.is_coalesced():
        output = output.coalesce()

    return output

def validate_dense_reshape(tensor: torch.Tensor,
                     input_shape: torch.Tensor,
                     output_shape: torch.Tensor,
                     task: Optional[str] = None
                     ):
    # Verify the number of tensor dimensions is large enough to encapsulate
    # the input shape and output shape

    if tensor.dim() - input_shape.shape[0] < 0:
        tensor_dim = tensor.dim()
        input_dim = input_shape.shape[0]
        reason = f"""\
        Rank of tensor parameter insufficient for resize:
        Param 'tensor' has rank {tensor_dim}. 
        However, Param 'input_shape' is {input_dim} units long.
        """
        reason = Core.dedent(reason)
        raise ReshapeException(reason, task)

    # Verify that the input_shape matches the tensor shape on
    # the broadcast dimensions

    input_shape_length = input_shape.shape[0]
    shape_as_list: List[int] = input_shape.tolist() # Required for torchscript to be happy
    if tensor.shape[-input_shape_length:] != torch.Size(shape_as_list):
        temp_input_shape = torch.Size(input_shape)
        tensor_shape = tensor.shape
        reason = f"""\
        Param 'tensor' shape and param 'input_shape' shape do not match:
            The param 'tensor' has shape {tensor_shape}.
            This cannot be broadcast with {temp_input_shape}.
        """
        reason = Core.dedent(reason)
        raise ReshapeException(reason, task)

    if input_shape.prod() != output_shape.prod():
        temp_input_shape = torch.Size(input_shape)
        temp_output_shape = torch.Size(output_shape)
        input_shape_elements = int(input_shape.prod())
        output_shape_elements = int(output_shape.prod())
        reason = f"""\
        Reshape is impossible with unequal number of elements.
            The number of elements for 'input_shape' shaped as {temp_input_shape} is {input_shape_elements}
            However, the number of elements for 'output_shape' shaped as {temp_output_shape} is {output_shape_elements}
            These do not match.

        """
        reason = Core.dedent(reason)
        raise ReshapeException(reason, task)

def dense_reshape(tensor: torch.Tensor,
                  input_shape: torch.Tensor,
                  output_shape: torch.Tensor):
    """
    Performs a dense reshape with broadcast
    tricks enabled.
    """

    broadcast_length = input_shape.shape[0]
    static_shape = torch.tensor(tensor.shape[:-broadcast_length], dtype=torch.int64)
    final_shape = torch.concat([static_shape, output_shape])
    shape_as_list: List[int] = final_shape.tolist() #Required for torchscript to compile
    final_size = torch.Size(shape_as_list)
    return torch.reshape(tensor, final_size)


def reshape(tensor: torch.Tensor,
            input_shape: Core.StandardShapeType,
            output_shape: Core.StandardShapeType,
            validate: bool = True,
            task: Optional[str] = None,
            ) -> torch.Tensor:
    """
    Performs validation and then performs a reshape.
    The reshape mechanism is broadcastable. This means
    demanding a reshape from [4, 5] to [20] will work
    on a tensor of shape [10, 3, 4, 5] and yield
    [10, 3, 20]

    An entirely pure function. For clearer errors,
    one may provide the task being performed while reshaping occurs

    Both sparse and dense reshapes are acceptable. Sparse reshapes
    will broadcast across sparse dimensions if the tensor is hybrid.
    There is no hybrid reshaping.

    :param tensor: The tensor to reshape
    :param input_shape: The input shape
    :param output_shape: The output shape
    :param validate: Whether or not to validate the action.
    :param task: What is going on when the error happened. Used for nice error messages.
    :return: The reshaped tensor
    """

    input_shape = Core.standardize_shape(input_shape, "input_shape")
    output_shape = Core.standardize_shape(output_shape, "output_shape")

    if tensor.is_sparse:
        if validate:
            validate_sparse_reshape(tensor, input_shape, output_shape, task)
        return _sparse_reshape(tensor, input_shape, output_shape)
    else:
        if validate:
            validate_dense_reshape(tensor, input_shape, output_shape, task)
        return dense_reshape(tensor, input_shape, output_shape)




class ReshapeClosure:
    """
    Executes a reshape with the stored closure
    parameters. Emitted by certain factory methods.
    """

    def generate_missing_param_message(self, parameter: str):
        missing_parameter_message = f"""\
        Neither Reshape '__call__' nor Reshape '__init__' include a definition for 
        parameter '{parameter}'. This parameter is required. Either call with 
        a definition for '{parameter}' or run constructor with a default for
        '{parameter}'
        """
        missing_parameter_message = Core.dedent(missing_parameter_message)
        return missing_parameter_message

    def __init__(self,
                 input_shape: Optional[Core.StandardShapeType] = None,
                 output_shape: Optional[Core.StandardShapeType] = None,
                 validate: Optional[bool] = None,
                 task: Optional[str] = None,
                 ):
        """
        One can provide a variety of defaults here.
        Doing so will cause an unfilled call to automatically
        substitute these defaults.
        """
        if input_shape is not None:
            input_shape = Core.standardize_shape(input_shape, 'input_shape')
        if output_shape is not None:
            output_shape = Core.standardize_shape(output_shape, 'output_shape')

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.validate = validate
        self.tasks = task

    def __call__(self,
                 tensor: torch.Tensor,
                 input_shape: Optional[Core.StandardShapeType] = None,
                 output_shape: Optional[Core.StandardShapeType] = None,
                 validate: bool = True,
                 task: Optional[str] = None,
                 ) -> torch.Tensor:

        if input_shape is None:
            input_shape = self.input_shape
        if input_shape is None:
            reason = self.generate_missing_param_message("input_shape")
            raise ReshapeException(reason, task)
        input_shape = Core.standardize_shape(input_shape, "input_shape", task=task)

        if output_shape is None:
            output_shape = self.output_shape
        if output_shape is None:
            reason = self.generate_missing_param_message("output_shape")
            raise ReshapeException(reason, task)
        output_shape = Core.standardize_shape(output_shape, "output_shape", task=task)

        # handle validation
        if validate is None:
            if self.validate is None:
                reason = self.generate_missing_param_message("validate")
                raise ReshapeException(reason, task)
            else:
                validate = self.validate

        if task is None:
            task = self.tasks

        return reshape(
            tensor,
            input_shape,
            output_shape,
            validate,
            task
        )


torch.jit.script(ReshapeClosure)


class ReshapeFactory(nn.Module):
    """
    A layer intended to emit reshape closures on
    command. May have parts of the reshape call
    filled in, in which case it will emit a
    ReshapeClosure that simply expects the remaining
    parameters to be filled
    """

    def __init__(self,
                 input_shape: Optional[Core.StandardShapeType] = None,
                 output_shape: Optional[Core.StandardShapeType] = None,
                 validate: bool = True,
                 task: Optional[str] = None
                 ):
        """
        One can provide a variety of defaults here.
        Doing so will cause an unfilled call to automatically
        substitute these defaults.
        """
        super().__init__()
        if input_shape is not None:
            input_shape = Core.standardize_shape(input_shape, 'input_shape')
        if output_shape is not None:
            output_shape = Core.standardize_shape(output_shape, 'output_shape')

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.validate = validate
        self.task = task

    def __call__(self) -> ReshapeClosure:
        closure = ReshapeClosure(
            self.input_shape,
            self.output_shape,
            self.validate,
            self.task
        )
        return closure
