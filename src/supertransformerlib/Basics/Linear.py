"""

Definitions for the version of linear used
throughout the library.

In general, getting a layer running consists of
three distinct steps. These are

* Defining the factory layer
* Getting a Closure for the layer
* Executing the Closure with whatever is so desired.

"""
from typing import Optional, Dict
from collections import namedtuple

import torch

import src.supertransformerlib.Core.Errors as Errors
import src.supertransformerlib.Core.Functions as Functions
import src.supertransformerlib.Core.StringUtil as StringUtil
import src.supertransformerlib.Core.Reshape as Reshape
from src.supertransformerlib.Core import Reshape
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
                dynamic: Optional[torch.Tensor],
                device: Optional[torch.device],
                dtype: Optional[torch.dtype]) -> Core.Parameter:
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

    init = torch.nn.init.kaiming_uniform_
    parameter = Core.Parameter(init, shape, dynamic, dtype=dtype, device=device)
    return parameter


def make_bias(output_shape: torch.Tensor,
              parallel: Optional[torch.Tensor],
              dynamic: Optional[torch.Tensor],
              device: Optional[torch.device],
              dtype: Optional[torch.dtype]) -> Core.Parameter:
    """
    Makes the required bias usable for performing
    linear operations in a particular situation.

    Dimensions are defined, from left to right, in terms of
    (dynamic_dims..., parallel_dims..., output_shape)


    :param output_shape: The expected input dynamic_shape
    :param parallel: The number of parallel dimensions
    :param dynamic: The number of dynamic dimensions
    :return: The matmul kernel
    """

    matrix_columns = output_shape.prod().unsqueeze(-1)
    shape = matrix_columns
    if parallel is not None:
        shape = torch.concat([parallel, shape])

    init = torch.nn.init.zeros_
    parameter = Core.Parameter(init, shape, dynamic, dtype=dtype, device=device)
    return parameter





class Linear(nn.Module):
    """
    The core linear layer. Used in almost all other reasonable work. Performs
    matrix multiplication plus add. But has a lot of additional tricks
    associated with it.

    --- Tricks ----

    A number of additional tricks are included in the linear layer
    operation. These include, but are not restricted only to,

    * Autoreshaping
    * Parallel Kernels
    * Dynamic Superposition

    ---- Basics ----

    It is assumed the programmer working with this class
    has some level of familiarity with torch's or another
    libraries linear layer.

    One may setup and use a Linear layer presented
    here just like any ol' layer in torch.

    ```
        tensor = torch.randn([10])

        layer = Basics.Linear(10, 5)
        output = layer(tensor)
    ```

    Batched tensors are also processed sanely

    ```
        tensor = torch.randn([5, 10])

        layer = Basics.Linear(10, 5)
        output = layer(tensor)
    ```

    Notably, there is no theoredical limit to how many batch dimensions
    can be processed at once

        ```
        tensor = torch.randn([20, 30, 7, 5, 10])

        layer = Basics.Linear(10, 5)
        output = layer(tensor)
    ```

    It is possible to do additional actions, such as ignoring bias,
    defining the device and dtype of the kernel, and more on initialization.

    ---- Torchscript passing ----

    A setup closure can be passed around through torchscript for
    later usage. This allows modularization of code. The type
    of the closure is located on the layer as parameter "ClosureType"

    A bare bones example looks something like as follows:

    ```
        tensor = torch.randn([10])

        @torch.jit.script
        def do_linear(tensor: torch.Tensor, closure: Linear.ClosureType):
            return closure(tensor)

        layer = Linear(10, 5)
        closure = layer()
        output = do_linear(tensor, closure)
    ```

    More sophisticated constructions are also possible

    ---- Autoreshaping ---

    It is the case one can define a projection to occur and the layer
    will automatically reshape to match. This consists of flattening along
    these dimensions, then performing a linear operation

    For example, say you want to project a tensor with dynamic_shape
    [3, 5, 6] into a tensor with dynamic_shape [3, 3, 2].  The first dimension
    is a batch dimension. This can be done as:

    ```
        tensor = torch.randn([3, 5, 6])
        input_shape = [5, 6]
        output_shape = [3, 2]

        layer = Linear(input_shape, output_shape)
        output = layer(tensor)
        assert output.shape == torch.Size([3, 3, 2])
    ```

    Conceptually, what is going on is the projected dimensions are
    flattened, run as a standard linear operation, and then
    reassembled.

    ----- parallel kernels -----

    It is the case ensembles can also be elegantly handled using this
    linear layer. That is the purpose of the parallel parameter.

    Lets say you have a matmul from 10 to 5  with batch size 20, which you want to apply
    in parallel with 10 independent ensembles. This can easily be done

    ```

        tensor = torch.randn([20, 10, 10])

        layer = Basics.Linear(10, 5, parallel=10)
        output = layer(tensor)
    ```

    This is not the limit, however. If multiple dimensions
    worth of ensembles are needed, for whatever reason, that is also
    supported. Suppose you want one ensemble for every location on a
    7 by 7 grid, and your data is coming in binned by x and y.

    This would be supported as

    ```

    data = torch.randn([20, 7, 7, 10])

    layer = Basics.Linear(10, 5, parallel=[7, 7])
    output = layer(data)
    ```

    ------ dynamic superposition ------

    By far the most difficult to understand feature is dynamic superposition.

    If you have exposure to quantum mechanics, this will be quick and easy. Dynamic Superposition
    sets up ensembles using the 'superposition_shape' parameter on initialization. Then, on call,
    a "superposition_weight" parameter is defined which acts as weights on each kernel, which
    are then superimposed together and executed in one go. Those who have taken quantum mechanics
    can now leave.

    If you have exposure to mathematics or advanced tensor mechanics, we can also get through
    this fairly quick. The superposition_weights will have their dot product taken with regards
    to the superposition shape in the kernel, reducing all associated dimensions. THen the
    resulting kernel is executed.



    The superposition_weights parameter must be filled when a superposition is defined.
    Each element in first N dimensions of the weight must corrospond to an element
    of the superposition_shape defined in the constructor. The weights can be dense
    or sparse. Additionally, if the superposition_weights parameter is found to be longer than
    N, the excess on the end are defined to be batch dimensions.
    """
    def __init__(self,
                 input_shape: Functions.StandardShapeType,
                 output_shape: Functions.StandardShapeType,
                 parallel: Optional[Functions.StandardShapeType] = None,
                 dynamic: Optional[Functions.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 use_bias: bool = True,
                 ):
        """

        :param input_shape: What dynamic_shape, in either list or int format, to accept as inputs for linear
        :param output_shape: What dynamic_shape, in either list or int format, to accept as inputs for linear
        :param parallel: What dynamic_shape, in either list or int format, to setup as ensemble dimensions
        :param dynamic: What dynamic_shape, in either list or int format, to setup as dynamic dimensions
        :param dtype: The dtype. Defaults to float64
        :param device: The device. Defaults to CPU
        :param use_bias: Whether to use bias
        """

        super().__init__()

        task = "Creating a linear layer"
        input_shape = Functions.standardize_shape(input_shape, 'input_shape', task=task)
        output_shape = Functions.standardize_shape(output_shape, 'output_shape', task=task)

        if parallel is not None:
            parallel = Functions.standardize_shape(parallel, 'parallel', task=task)
        if dynamic is not None:
            dynamic = Functions.standardize_shape(dynamic, 'dynamic', task=task)


        self.input_shape = input_shape
        self.output_shape = output_shape
        self.parallel = parallel
        self.dynamic = dynamic

        self.input_map = Core.Reshape(input_shape, input_shape.prod().unsqueeze(-1))
        self.output_map = Core.Reshape(output_shape.prod().unsqueeze(-1), output_shape)
        self.expected_input_shape = input_shape

        self.kernel = make_kernel(input_shape, output_shape, parallel, dynamic, device, dtype)
        if use_bias:
            self.bias = make_bias(output_shape, parallel, dynamic, device, dtype)
        else:
            self.bias = None

    def forward(self,
                tensor: torch.Tensor,
                superposition_weights: Optional[torch.Tensor] = None
                )->torch.Tensor:


        # Create the kernels accounting for any superposition
        weight_name = "superposition_weights"
        matrix = self.kernel(superposition_weights, weight_name, "setting up superimposed matrix for linear forward")
        if self.bias is not None:
            bias = self.bias(superposition_weights, weight_name, "setting up superimposed bias for linear forward")
        else:
            bias = None

        # Run forward call
        tensor = self.input_map(tensor, "reshaping tensor for linear forward")
        tensor = linear_forward(tensor, matrix, bias, "Executing linear forward")
        tensor = self.output_map(tensor, "reshaping tensor after linear forward")

        return tensor
