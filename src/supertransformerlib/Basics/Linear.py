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
import src.supertransformerlib.Core.Errors as Errors
import src.supertransformerlib.Core.Functions as Functions
from src.supertransformerlib import Core
from torch import nn


class LinearForwardException(Errors.ValidationError):
    """
    Called when catching an error during
    the forward phase
    """

    def __init__(self, reason: str, task: Optional[str] = None):
        typing = "LinearForwardException"
        self.reason = reason
        self.task = task
        super().__init__(typing, reason, task)


class LinearCreationException(Errors.ValidationError):
    """
    Called when something goes wrong on creating
    a linear layer.
    """

    def __init__(self, reason: str, task: Optional[str] = None):
        typing = "LinearCreationException"
        self.reason = reason
        self.task = task
        super().__init__(typing, reason, task)


class LinearFactoryException(Errors.ValidationError):
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


### Parameters Logic ###
# Here are stored various information about the nn.Module layers
# which may actually contain the parameters and generation
# mechanisms needed to get a closure mechanism setup.

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
    :return: The matmul kernel
    """

    matrix_rows = input_shape.prod().unsqueeze(-1)
    matrix_columns = output_shape.prod().unsqueeze(-1)
    shape = torch.concat([matrix_rows, matrix_columns])
    if parallel is not None:
        shape = torch.concat([parallel, shape])

    shape_as_list: List[int] = shape.tolist() # Required by torchscript
    parameter = torch.empty(shape_as_list, dtype = dtype, device =device)
    nn.init.kaiming_uniform_(parameter)
    parameter = nn.Parameter(parameter)
    return parameter


def make_bias(output_shape: torch.Tensor,
              parallel: Optional[torch.Tensor],
              device: Optional[torch.device],
              dtype: Optional[torch.dtype]) -> nn.Parameter:

    matrix_columns = output_shape.prod().unsqueeze(-1)
    shape = matrix_columns
    if parallel is not None:
        shape = torch.concat([parallel, shape])

    shape_as_list: List[int] = shape.tolist() # Required by torchscript
    parameter = torch.zeros(shape_as_list, dtype=dtype, device=device)
    parameter = nn.Parameter(parameter)
    return parameter

class _Linear:
    """
    The core linear forward operation

    Will apply a linear transform to an incoming
    tensor. Will also throw errors on violations.

    It is produced by linear factory, and should
    be configured in that class.
    """
    def __init__(self,
                 input_reshape: Core.ReshapeClosure,
                 kernel: torch.Tensor,
                 bias: Optional[torch.Tensor],
                 output_reshape: Core.ReshapeClosure
                 ):
        self.inputReshape = input_reshape
        self.kernel = kernel
        self.bias = bias
        self.outputReshape = output_reshape
    def __call__(self, tensor: torch.Tensor)->torch.Tensor:
        tensor = self.inputReshape(tensor)
        tensor = linear_forward(tensor, self.kernel, self.bias)
        tensor = self.outputReshape(tensor)
        return tensor

torch.jit.script(_Linear)
class LinearFactory(nn.Module):
    """
    The core linear factory. Used in many other products.
    Produces a linear closure class when called, capable of applying
    the linear operation. Displays typing of closure class as well


    ---- Design ----

    The basic idea is that you will go ahead and use the factory
    layer to make a linear operation for a particular batch.


    ---- Basics ----

    To use the class, one must first setup the closure and
    then the closure can be utilized. As an example, consider
    projecting tensor [10, 4] to [10, 7].

    ```
        test_tensor = torch.rand([10, 4])

        linearFactory = Basics.LinearFactory(4, 7)
        linear = linearFactory()

        output = linear(test_tensor)
        print(output.shape) # will be [10, 7]
    ```

    ---- Tricks ----

    There are a number of useful tricks to be aware of.

    * Autoreshaping
    * Parallel Kernels
    * Torchscript Compatibility
    * Functional Passthrough
    ---- Autoreshaping ----

    It is quite possible to demand the linear operation
    transform an entire block from one shape to another. Such
    an action is called autoreshaping. It consists of flattening the
    targetted dimensions, performing linear, then restoring.

    For example, consider transforming a tensor of shape [4, 5, 10] to [4, 3]:

    ```
        test_tensor = torch.rand([4, 5, 10])

        linearFactory = Basics.LinearFactory([5, 10], 3)
        linear = linearFactory()

        output = linear(test_tensor)
        print(output.shape) # Will be [4, 3]
    ```

    ---- Parallel execution ----

    It is also possible to dedicate unique blocks of parameters
    to different dimensions. This is called utilizing parallel
    kernels, and might be useful for making, for example, ensembles.

    Consider an example with a tensor mapping from [4, 2, 5, 10] to
    [4, 2, 5, 20]. 4 is the batch dimension. [2,5] is the ensemble dimension, and we
    want each one to be different.

    ```
        test_tensor = torch.rand([4, 5, 10])

        linearFactory = Basics.LinearFactory(10, 20, [2, 5])
        linear = linearFactory()

        output = linear(test_tensor) # each dimension among [2, 5] has independent parameters
    ```

    ---- torchscript compatibility ----

    The program is defined such that it is possible to use torchscript with
    it. Additionally, one can pass a factory entity into a class or function
    using the right typing hint. This exists in large part to allow some
    degree of object-oriented programming.

    ```
        test_tensor = torch.rand([10, 4])

        linearFactory = Basics.LinearFactory(4, 20)

        @torch.jit.script
        def apply_later(linear: linearFactory.Type, tensor: torch.Tensor):
            return linear(tensor)

        #... later

        linear = linearFactory()
        output = apply_later(linear, test_tensor)
    ```


    """
    Type = _Linear
    def __init__(self,
                 input_shape: Functions.StandardShapeType,
                 output_shape: Functions.StandardShapeType,
                 parallel: Optional[Functions.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 use_bias: bool = True,
                 ):

        super().__init__()

        task = "Creating a linear layer"
        input_shape = Functions.standardize_shape(input_shape, 'input_shape', task=task)
        output_shape = Functions.standardize_shape(output_shape, 'output_shape', task=task)

        if parallel is not None:
            parallel = Functions.standardize_shape(parallel, 'parallel', task=task)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.parallel = parallel

        self.inputMapFactory = Core.ReshapeFactory(input_shape, input_shape.prod().unsqueeze(-1))
        self.outputMapFactory = Core.ReshapeFactory(output_shape.prod().unsqueeze(-1), output_shape)
        self.expected_input_shape = input_shape

        self.kernel = make_kernel(input_shape, output_shape, parallel, device, dtype)
        if use_bias:
            self.bias = make_bias(output_shape, parallel, device, dtype)
        else:
            self.bias = None
    def forward(self)->_Linear:
        # Create the kernels accounting for any superposition
        task = "executing linear forward reshape"
        return _Linear(self.inputMapFactory(task),
                       self.kernel,
                       self.bias,
                       self.outputMapFactory(task), )

