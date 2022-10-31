"""

Code for the reshape mechanism. A version
of torch's reshape with a few extra tricks.

"""

from typing import Optional, List
import torch
from torch import nn

import src.supertransformerlib.Core.Errors as Errors
import src.supertransformerlib.Core.Functions as Functions
import src.supertransformerlib.Core.StringUtil as StringUtil
import src.supertransformerlib.Core.SparseUtils as SparseUtil

class ReshapeException(Errors.ValidationError):
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
        reason = StringUtil.dedent(reason)
        raise ReshapeException(reason, task)

    sparse_shape = tensor.shape[:sparse_dim]
    input_shape_as_list: List[int] = input_shape.tolist() #Line required to prevent torchscript from throwing a fit
    if sparse_shape[-reshape_dim:] != torch.Size(input_shape_as_list):
        temp_input_shape = torch.Size(input_shape_as_list)
        reason = f"""\
        Param 'tensor' mutable shape and param 'input_shape' mutable shape do not match:
        
        The param 'tensor' has a shape of {tensor.shape}. Of these, the dimensions
        which will be targetted by the sparse reshape are {sparse_shape}. The reshape
        instruction we were provided with was to work with something which has last
        sparse dimensions of shape {temp_input_shape}. This was not compatible
        """
        reason = StringUtil.dedent(reason)
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
        reason = StringUtil.dedent(reason)
        raise ReshapeException(reason, task)


def _sparse_reshape(tensor: torch.Tensor,
                    input_shape: torch.Tensor,
                    output_shape: torch.Tensor) -> torch.Tensor:
    """
    Performs a sparse reshape on a tensor to another dynamic_shape.

    :param tensor: The sparse tensor to reshape
    :param dynamic_shape: The dynamic_shape the tensor should end up in
    :return: The reshaped sparse tensor
    """
    # Perform a sparse reshape in a torchscript compatible method
    #
    # This is done by first converting the index to be an equivalent tensor
    # in flattened strided notation, then rebuilding the index under the new
    # dynamic_shape.
    #
    # This is only capable of reshaping the sparse dimensions. Dense dimensions
    # will cause errors

    if not tensor.is_coalesced():
        tensor = tensor.coalesce()


    #Develop strided representation

    sparse_dim = tensor.sparse_dim()
    sparse_shape = tensor.shape[:sparse_dim]
    dense_shape = tensor.shape[sparse_dim:]
    sparse_strides = SparseUtil.calculate_shape_strides(sparse_shape)

    indices = tensor.indices()
    values = tensor.values()

    flat_indices = indices*sparse_strides.unsqueeze(-1)
    flat_indices = flat_indices.sum(dim=0)

    #Develop proper final dynamic_shape.

    broadcast_length = input_shape.shape[0]
    static_shape = torch.tensor(sparse_shape[:-broadcast_length], dtype=torch.int64)
    final_shape = torch.concat([static_shape, output_shape])
    final_shape_as_list: List[int] = final_shape.tolist() #Line required or torchscript throws a hissy fit
    final_strides = SparseUtil.calculate_shape_strides(final_shape_as_list)

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
    entire_shape_as_list: List[int] = entire_shape.tolist() # Line required or torchscript throws a fit
    entire_shape = torch.Size(entire_shape_as_list)

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
    # the input dynamic_shape and output dynamic_shape

    if tensor.dim() - input_shape.shape[0] < 0:
        tensor_dim = tensor.dim()
        input_dim = input_shape.shape[0]
        reason = f"""\
        Rank of tensor parameter insufficient for resize:
        Param 'tensor' has rank {tensor_dim}. 
        However, Param 'input_shape' is {input_dim} units long.
        """
        reason = StringUtil.dedent(reason)
        raise ReshapeException(reason, task)

    # Verify that the input_shape matches the tensor dynamic_shape on
    # the broadcast dimensions

    input_shape_length = input_shape.shape[0]
    shape_as_list: List[int] = input_shape.tolist() # Required for torchscript to be happy
    if tensor.shape[-input_shape_length:] != torch.Size(shape_as_list):
        temp_input_shape = torch.Size(input_shape)
        tensor_shape = tensor.shape
        reason = f"""\
        Param 'tensor' dynamic_shape and param 'input_shape' dynamic_shape do not match:
            The param 'tensor' has dynamic_shape {tensor_shape}.
            This cannot be broadcast with {temp_input_shape}.
        """
        reason = StringUtil.dedent(reason)
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
        reason = StringUtil.dedent(reason)
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
            input_shape: Functions.StandardShapeType,
            output_shape: Functions.StandardShapeType,
            validate: bool = True,
            task: Optional[str] = None,
            ) -> torch.Tensor:
    """
    Performs validation and then performs a reshape.
    The reshape mechanism is broadcastable. This means
    demanding a reshape from [4, 5] to [20] will work
    on a tensor of dynamic_shape [10, 3, 4, 5] and yield
    [10, 3, 20]

    An entirely pure function. For clearer errors,
    one may provide the task being performed while reshaping occurs

    Both sparse and dense reshapes are acceptable. Sparse reshapes
    will broadcast across sparse dimensions if the tensor is hybrid.
    There is no hybrid reshaping.

    :param tensor: The tensor to reshape
    :param input_shape: The input dynamic_shape
    :param output_shape: The output dynamic_shape
    :param validate: Whether or not to validate the action.
    :param task: What is going on when the error happened. Used for nice error messages.
    :return: The reshaped tensor
    """

    input_shape = Functions.standardize_shape(input_shape, "input_shape")
    output_shape = Functions.standardize_shape(output_shape, "output_shape")

    if tensor.is_sparse:
        if validate:
            validate_sparse_reshape(tensor, input_shape, output_shape, task)
        return _sparse_reshape(tensor, input_shape, output_shape)
    else:
        if validate:
            validate_dense_reshape(tensor, input_shape, output_shape, task)
        return dense_reshape(tensor, input_shape, output_shape)

class Reshape(nn.Module):
    """
    A layer to perform the same reshape
    operation over and over. It also save
    and scripts nicely.
    """
    def __init__(self,
                 initial_shape: Functions.StandardShapeType,
                 final_shape: Functions.StandardShapeType):
        super().__init__()

        initial_shape = Functions.standardize_shape(initial_shape, 'initial_shape', task="setting up reshape")
        final_shape = Functions.standardize_shape(final_shape, 'final_shape', task='setting up reshape')

        self.register_buffer('input_shape', initial_shape)
        self.register_buffer('output_shape', final_shape)
    def forward(self, tensor: torch.Tensor, task: Optional[str] = None)->torch.Tensor:
        return reshape(tensor, self.input_shape, self.output_shape, task=task)

