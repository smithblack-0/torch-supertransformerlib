"""

Definitions for the version of linear used
throughout the library.

In general, getting a layer running consists of
three distinct steps. These are

* Defining the factory layer
* Getting a Closure for the layer
* Executing the Closure with whatever is so desired.

"""
from typing import Optional, List
import torch

from torch import nn
from supertransformerlib import Core


class LinearForwardException(Core.ValidationError):
    """
    Called when catching an error during
    the forward phase
    """

    def __init__(self, reason: str, task: Optional[str] = None):
        typing = "LinearForwardException"
        self.reason = reason
        self.task = task
        super().__init__(typing, reason, task)


class LinearCreationException(Core.ValidationError):
    """
    Called when something goes wrong on creating
    a linear layer.
    """

    def __init__(self, reason: str, task: Optional[str] = None):
        typing = "LinearCreationException"
        self.reason = reason
        self.task = task
        super().__init__(typing, reason, task)


class LinearFactoryException(Core.ValidationError):
    """
    Called when something goes wrong when making
    the linear closure in the first place
    """

    def __init__(self, reason: str, task: Optional[str] = None):
        typing = "LinearFactory"
        super().__init__(typing, reason, task)


def _linear_forward(tensor: torch.Tensor,
                    kernel: torch.Tensor,
                    bias: Optional[torch.Tensor] = None,
                    ) -> torch.Tensor:
    """
    The direct linear forward function.
    Entirely pure.

    :param tensor: The tensor for the forward operation
    :param kernel: The kernel for the forward operation
    :param bias: Optionally, the bias for the forward operation
    :return:
    """

    tensor = tensor.unsqueeze(-1)
    tensor = torch.matmul(kernel.transpose(-1, -2), tensor)
    tensor = tensor.squeeze(-1)

    if bias is not None:
        tensor = tensor + bias
    return tensor


def linear_forward(tensor: torch.Tensor,
                   kernel: torch.Tensor,
                   bias: Optional[torch.Tensor] = None,
                   task: Optional[str] = None
                   ):
    """
    The linear forward method. With valdiation included.
    """

    if tensor.dtype != kernel.dtype:
        tensor_dtype = tensor.dtype
        kernel_dtype = kernel.dtype
        reason = f"""\
        The parameter 'tensor' was found to have the wrong
        dtype when executing the linear operation. The tensor
        had dtype {tensor_dtype}. However, the linear kernel
        has dtype {kernel_dtype}.
        
        Either move the layer or the tensor to a common dtype
        using .to(dtype)
        """
        reason = Core.StringUtil.dedent(reason)
        raise LinearForwardException(reason, task)
    if tensor.device != kernel.device:
        tensor_device = tensor.device
        kernel_device = kernel.device
        reason = f"""\
        The parameter 'tensor' was found to have the 
        wrong device when executing the linear operation. 
        The tensor has device {tensor_device}, but
        the layer is defined on device {kernel_device}
        
        Either move the layer or the tensor to a common
        device using .to(device)"""
        reason = Core.StringUtil.dedent(reason)
        raise LinearForwardException(reason, task)

    if tensor.shape[-1] != kernel.shape[-2]:
        tensor_dim_size = tensor.shape[-1]
        kernel_dim_size = kernel.shape[-2]
        reason = f"""\
        Cannot perform linear operation. 'tensor' Tensor's dim -1 has
        size {tensor_dim_size}. However, the layer was setup
        with an expected input width of {kernel_dim_size}.
        
        Tensor had shape of {tensor.shape}
        """
        reason = Core.StringUtil.dedent(reason)
        raise LinearForwardException(reason, task)

    if tensor.dim() < kernel.dim() - 1:
        tensor_actual_dim = tensor.dim()
        kernel_required_dim = kernel.dim() - 1
        reason = f"""\
         Tensor is insufficient rank for parallel linear execution.
         The tensor was expected to have rank {kernel_required_dim}
         due to the parallel kernels. However, it was only found to have
         rank {tensor_actual_dim}.

         Reduce the number of parallel kernel dimensions, or increase 
         the rank of the tensor.
         """
        reason = Core.StringUtil.dedent(reason)
        raise LinearForwardException(reason, task)

    kernel_parallel_shape = kernel.shape[:-2]
    parallel_length = len(kernel_parallel_shape)
    tensor_parallel_shape = tensor.shape[-(1 + parallel_length):-1]
    if kernel_parallel_shape != tensor_parallel_shape:
        reason = f"""\
        Tensor has incorrect dynamic_shape for parallel linear
        operation. The tensor was found to have a 
        parallel dynamic_shape of {tensor_parallel_shape}. 
        
        However, it is the case that the kernel was 
        defined with parallel dynamic_shape {kernel_parallel_shape}
        
        Modify the tensor to match, or the parallel dimensions
        to match.
        """
        reason = Core.StringUtil.dedent(reason)
        raise LinearForwardException(reason, task)

    return _linear_forward(tensor, kernel, bias)


torch.jit.script(linear_forward)  # noqa


class Linear(nn.Module):
    """
    The core linear operation. It is capable
    of doing autoreshaping and parallel ensemble
    operations.

    It is torchscript compatible.

    """

    @staticmethod
    def make_kernel(input_shape: torch.Tensor,
                    output_shape: torch.Tensor,
                    parallel: Optional[torch.Tensor],
                    device: Optional[torch.device],
                    dtype: Optional[torch.dtype]) -> nn.Parameter:
        """
        Makes the required kernel usable for performing
        linear operations in a particular situation.

        Dimensions are defined, from left to right, in terms of
        (dynamic_dims..., parallel_dims..., input_shape, output_shape)


        :param input_shape: The expected input dynamic_shape
        :param output_shape: The expected output dynamic_shape
        :param parallel: The number of parallel dimensions
        :param dynamic: The number of dynamic dimensions
        :param device: The torch device
        :param dtype: The torch dtype
        :return: The matmul kernel
        """

        matrix_rows = input_shape.prod().unsqueeze(-1)
        matrix_columns = output_shape.prod().unsqueeze(-1)
        shape = torch.concat([matrix_rows, matrix_columns])
        if parallel is not None:
            shape = torch.concat([parallel, shape])

        shape_as_list: List[int] = shape.tolist()  # Required by torchscript
        parameter = torch.empty(shape_as_list, dtype=dtype, device=device)
        nn.init.kaiming_uniform_(parameter)
        parameter = nn.Parameter(parameter)
        return parameter

    @staticmethod
    def make_bias(output_shape: torch.Tensor,
                  parallel: Optional[torch.Tensor],
                  device: Optional[torch.device],
                  dtype: Optional[torch.dtype]) -> nn.Parameter:
        """ Makes the bias. Only used on init"""

        matrix_columns = output_shape.prod().unsqueeze(-1)
        shape = matrix_columns
        if parallel is not None:
            shape = torch.concat([parallel, shape])

        shape_as_list: List[int] = shape.tolist()  # Required by torchscript
        parameter = torch.zeros(shape_as_list, dtype=dtype, device=device)
        parameter = nn.Parameter(parameter)
        return parameter

    def __init__(self,
                 input_shape: Core.StandardShapeType,
                 output_shape: Core.StandardShapeType,
                 parallel: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 use_bias: bool = True,
                 ):
        """
        :param input_shape: The expected shape of the input. May be list of ints
        :param output_shape: The expected shape of the output. May be list of ints
        :param parallel: The shape of the parallel ensembles
        :param dtype: The dtype of the kernsl
        :param device: The device to build this on
        :param use_bias: If you should use bias or not.
        """

        super().__init__()

        task = "Creating a linear layer"
        input_shape = Core.standardize_shape(input_shape, 'input_shape', task=task)
        output_shape = Core.standardize_shape(output_shape, 'output_shape', task=task)

        if parallel is not None:
            parallel = Core.standardize_shape(parallel, 'parallel', task=task)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.parallel = parallel

        self.input_reshaper = Core.Reshape(input_shape, input_shape.prod().unsqueeze(-1))
        self.output_reshaper = Core.Reshape(output_shape.prod().unsqueeze(-1), output_shape)
        self.expected_input_shape = input_shape

        self.kernel = self.make_kernel(input_shape, output_shape, parallel, device, dtype)
        if use_bias:
            self.bias = self.make_bias(output_shape, parallel, device, dtype)
        else:
            self.bias = None

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """ Runs reshape and linear operation."""
        tensor = self.input_reshaper(tensor)
        tensor = linear_forward(tensor, self.kernel, self.bias)
        tensor = self.output_reshaper(tensor)
        return tensor
