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

import src.supertransformerlib.Core.Errors
import src.supertransformerlib.Core.Functions
import src.supertransformerlib.Core.StringUtil
from src.supertransformerlib.Core import Reshape
from src.supertransformerlib import Core

from torch import nn


class LinearForwardException(src.supertransformerlib.Core.Errors.ValidationError):
    """
    Called when catching an error during
    the forward phase
    """

    def __init__(self, reason: str, task: Optional[str] = None):
        typing = "LinearForwardException"
        self.reason = reason
        self.task = task
        super().__init__(typing, reason, task)


class LinearCreationException(src.supertransformerlib.Core.Errors.ValidationError):
    """
    Called when something goes wrong on creating
    a linear layer.
    """

    def __init__(self, reason: str, task: Optional[str] = None):
        typing = "LinearCreationException"
        self.reason = reason
        self.task = task
        super().__init__(typing, reason, task)


class LinearFactoryException(src.supertransformerlib.Core.Errors.ValidationError):
    """
    Called when something goes wrong when making
    the linear closure in the first place
    """

    def __init__(self, reason: str, task: Optional[str] = None):
        typing = "LinearFactory"
        super().__init__(typing, reason, task)


def linear_forward(tensor: torch.Tensor,
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


class LinearClosure:
    """
    The torchscript passible linear operation
    which can be utilized to perform a linear
    call any number of times.

    Contains the majority of the error handling
    methods for linear calls. Can be called on a tensor to perform
    a linear operation. Does not update it's kernel on backprop.
    """

    def validate_primitive(self, tensor: torch.Tensor, task: Optional[str]):
        if tensor.dtype != self.kernel.dtype:
            tensor_dtype = tensor.dtype
            kernel_dtype = self.kernel.dtype
            reason = f"""\
            The parameter 'tensor' was found to have the wrong
            dtype when executing the linear operation. The tensor
            had dtype {tensor_dtype}. However, the linear kernel
            has dtype {kernel_dtype}.
            
            Either move the layer or the tensor to a common dtype
            using .to(dtype)
            """
            reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
            raise LinearForwardException(reason, task)
        if tensor.device != self.kernel.device:
            tensor_device = tensor.device
            kernel_device = self.kernel.device
            reason = f"""\
            The parameter 'tensor' was found to have the 
            wrong device when executing the linear operation. 
            The tensor has device {tensor_device}, but
            the layer is defined on device {kernel_device}
            
            Either move the layer or the tensor to a common
            device using .to(device)
            
            """
            reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
            raise LinearForwardException(reason, task)

    def validate_matmul(self, tensor: torch.Tensor, task: Optional[str]):

        if tensor.shape[-1] != self.kernel.shape[-2]:
            tensor_dim_size = tensor.shape[-1]
            kernel_dim_size = self.kernel.shape[-2]
            reason = f"""\
            Cannot perform linear operation. 'tensor' Tensor's dim -1 has
            size {tensor_dim_size}. However, the process was initialized
            with an input width of {kernel_dim_size}
            """
            reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
            raise LinearForwardException(reason, task)

        if tensor.dim() < self.kernel.dim() - 1:
            tensor_actual_dim = tensor.dim()
            kernel_required_dim = self.kernel.dim() - 1
            reason = f"""\
                 Tensor is insufficient rank for parallel linear execution.
                 The tensor was expected to have rank {kernel_required_dim}
                 due to the parallel kernels. However, it was only found to have
                 rank {tensor_actual_dim}.

                 Reduce the number of parallel kernel dimensions, or increase 
                 the rank of the tensor.
                 """
            reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
            raise LinearForwardException(reason, task)

        kernel_parallel_shape = self.kernel.shape[:-2]
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
            reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
            raise LinearForwardException(reason, task)

    def __init__(self,
                 kernel: torch.Tensor,
                 bias: Optional[torch.Tensor] = None,
                 input_map: Optional[Reshape.ReshapeClosure] = None,
                 output_map: Optional[Reshape.ReshapeClosure] = None,
                 validate: Optional[bool] = None
                 ):

        if validate is None:
            validate = True
        if bias is not None:
            bias = bias.clone()

        self.kernel = kernel.clone()
        self.bias = bias
        self.input_map = input_map
        self.output_map = output_map
        self.validate = validate

    def __call__(self, tensor: torch.Tensor, task: Optional[str] = None) -> torch.Tensor:

        # We need to go and fetch the maps into local
        # memory here for torchscript to compile. Else, torchscript
        # type refinement will fail.

        input_map = self.input_map
        output_map = self.output_map

        # Execute

        if task is None:
            task = "Executing a linear operation."

        if input_map is not None:
            tensor = input_map(tensor)

        if self.validate:
            self.validate_primitive(tensor, task)
            self.validate_matmul(tensor, task)

        tensor = linear_forward(tensor, self.kernel, self.bias)

        if output_map is not None:
            tensor = output_map(tensor, validate=self.validate, task=task)

        return tensor


# Cache the scripted version of the linear closure away while
# we still have the required environmental details around

torch.jit.script(LinearClosure)  # noqa


### Parameters Logic ###
# Here are stored various information about the nn.Module layers
# which may actually contain the parameters and generation
# mechanisms needed to get a closure mechanism setup.

def make_kernel(input_shape: torch.Tensor,
                output_shape: torch.Tensor,
                parallel: Optional[torch.Tensor],
                dynamic: Optional[torch.Tensor],
                device: Optional[torch.device],
                dtype: Optional[torch.dtype]) -> torch.Tensor:
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
    if dynamic is not None:
        shape = torch.concat([dynamic, shape])

    shape = torch.Size(shape)
    kernel = torch.empty(shape, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(kernel)
    return kernel


def make_bias(output_shape: torch.Tensor,
              parallel: Optional[torch.Tensor],
              dynamic: Optional[torch.Tensor],
              device: Optional[torch.device],
              dtype: Optional[torch.dtype]) -> torch.Tensor:
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
    if dynamic is not None:
        shape = torch.concat([dynamic, shape])

    shape = torch.Size(shape)
    bias = torch.empty(shape, device=device, dtype=dtype)
    torch.nn.init.zeros_(bias)
    return bias


def make_dense_superposition(dynamics: torch.Tensor,
                             kernel: torch.Tensor,
                             shape: torch.Tensor) -> torch.Tensor:
    """
    Makes a superposition of the kernel out of the dynamics weights
    """

    length = shape.shape[0]
    dynamics = dynamics.flatten(0, length-1)
    reduction_location = 0
    kernel = kernel.flatten(0, length - 1)
    for _ in range(kernel.dim() - 1):
        dynamics = dynamics.unsqueeze(-1)
    while kernel.dim() < dynamics.dim():
        kernel = kernel.unsqueeze(1)

    weighted_kernel = dynamics * kernel
    superimposed_kernel = weighted_kernel.sum(dim=reduction_location)
    return superimposed_kernel


torch.jit.script(make_dense_superposition)


def make_sparse_superposition(dynamics: torch.Tensor,
                              kernel: torch.Tensor,
                              shape: torch.Tensor):
    """
    Make a superposition out of the kernel when the dynamic
    weights are sparse. The ones which are not present are
    ignored.

    :param dynamics: The dynamic kernel. Expected to be sparse
    :param kernel: The kernel.
    :param shape: The dynamic dynamic_shape. Note that due to hybrid tensor restrictions the dynamic_shape should
        come as [...dynamic_shape, ...batch_shape], since sparse dimensions must remain sparse.
    :return: The superimposed kernel
    """

    # Flatten the tensors so that the dynamic dimensions are
    # found in one lump sum.

    length = shape.shape[0]
    input_shape = shape
    output_shape = torch.tensor([int(shape.prod())])
    dynamics = Reshape.reshape(dynamics, input_shape, output_shape, task="flattening sparse dynamic tensor")
    kernel = kernel.flatten(0, length - 1)

    # Resize the dynamic dimensions. expand the values where needed so
    # sparse mask will not throw a fit. Use memory efficient expansion

    dynamics_shape = dynamics.shape
    kernel_shape = kernel.shape

    dynamic_values = dynamics.values()
    dynamic_expansion = [-1] * dynamic_values.dim()
    dynamic_update_shape = list(dynamics_shape)

    for dim in kernel_shape[1:]:
        dynamic_expansion.append(dim)
        dynamic_update_shape.append(dim)
        dynamic_values = dynamic_values.unsqueeze(-1)

    dynamic_values = dynamic_values.expand(dynamic_expansion)
    dynamics = torch.sparse_coo_tensor(dynamics.indices(), dynamic_values, size=dynamic_update_shape)

    # Resize the kernel dimension so that they have the same
    # dynamic_shape.

    kernel_expansion = [-1] * len(kernel_shape)
    for i, dim in enumerate(dynamics_shape[1:]):
        kernel_expansion.insert(i + 1, dim)
        kernel = kernel.unsqueeze(1)
    kernel = kernel.expand(kernel_expansion)

    # Perform kernel mask, then add and sum resulting in
    # a weighted superposition.

    dynamics = dynamics.coalesce()
    kernel = kernel.sparse_mask(dynamics)

    weighted_kernel = kernel * dynamics

    # The following sum is utilized due to the fact that torch.sparse.sum
    # is broken under torchscript. It would have had to of been compiled
    # on location to be useful.
    superimposed_kernel = torch._sparse_sum(weighted_kernel, 0) #noqa
    return superimposed_kernel

def make_superposition(dynamic: torch.Tensor,
                       kernel: torch.Tensor,
                       shape: torch.Tensor):
    """
    Makes a superposition given the provided
    dynamic spec and kernel.
    """

    if dynamic.is_sparse:
        return make_sparse_superposition(dynamic, kernel, shape)
    else:
        return make_dense_superposition(dynamic, kernel, shape)


class Linear(nn.Module):
    """
    The linear factory layer. This layer will generate
    the linear closures needed to perform a variety
    of operations. It is capable of handling parallel
    layers along with dynamic superpositions.

    --- Usage ---

    This layer may be setup using information such as the
    expected input dynamic_shape, the expected output dynamic_shape, and various
    additional modal information. Doing this will create a linear
    factory layer.

    Once the factory is setup, it can be called to create a linear
    closure. This closure may, depending on the configuration,
    require additional information to create.

    Finally, once the closure is created, you can use it execute
    a linear operation. Notably, the closure will pass elegantly
    and without trouble through torchscript problems.

    --- Tricks ----

    A number of additional tricks are included in the linear layer
    operation. These include, but are not restricted only to,

    * Autoreshaping
    * Parallel Kernels
    * Dynamic Superposition

    ---- Basics ----

    The closure mechanism is needed for torchscript and onnx compiling
    to go smoothly. This means there is a slight alteration in using a
    linear layer. Lets say we are mapping a tensor with dim
    10 to dim 5. We can set that up much like torch as:


    ```
        tensor = torch.randn([10])
        expected_shape = torch.Size([5])

        layer = Linear.Linear(10, 5)
        closure = layer()
        output = closure(tensor)
    ```

    The extra call sets up the closure, which we
    then immediately use.

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
    will automatically reshape to match.

    For example, say you want to project a tensor with dynamic_shape
    [3, 5, 6] into a tensor with dynamic_shape [3, 2]. This can be done
    as:

    ```
        tensor = torch.randn([3, 5, 6])
        input_shape = [3, 5, 6]
        output_shape = [3, 2]

        layer = Linear(input_shape, output_shape)
        closure = layer()
        output = closure(tensor)
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

        tensor = torch.randn([20, 10])
        tensor = tensor.unsqueeze(-2).expand(-1, 10, -1)


        layer = Linear(10, 5, parallel=10)
        closure = layer()
        output = closure(tensor)
    ```

    This is not the limit, however. If multiple dimensions
    worth of ensembles are needed, for whatever reason, that is also
    supported. Suppose you want one ensemble for every location on a
    7 by 7 grid, and your data is coming in binned by x and y.

    This would be supported as

    ```

    data = torch.randn([20, 7, 7, 10])

    layer = Linear(10, 5, parallel=[7, 7])
    closure = layer()
    output = closure(data)
    ```

    ------ dynamic superposition ------

    One way to get around some thorny discrete training issues
    is to try all possibilities. This is the idea behind dynamic
    superposition.

    You may define a kernelspace of arbitrary dynamic_shape to be designated
    "dynamic" and for which it is the case that weights will need
    to be provided. Notably, these weights need a little discussion.
    Unlike the parallel kernels, the dynamic dimensions must
    corrolate with the beginning of the tensor. That is, if you
    have dynamic dimensions of [12, 5], then your weight tensor
    must start with dynamic_shape [12, 5]. This is due to restrictions
    in how torch handles sparse hybrid tensors, which can be
    utilized for the process. The remaining dimensions after
    this should be whatever batch features are needed

    As an example, lets say we have a network of 12 dynamically
    configured layer that are configured per batch entry across
    a batch of dynamic_shape [7, 7]. The linear operation should map
    from size 5 to size 6. The layer should first
    setup a configuration, then generate a dynamic superposition
    and execute linear.



    The layer has a mapping of 10 to 10, a batch size of 11,
    and 12 dynamic kernels to draw from. These kernels may in
    turn be configured by earlier segments of the model. One
    might develop something like the following to execute this

    ```
        batch_shape = [7, 7]
        input_dim = 5
        output_dim = 6
        dynamic = 12


        test_data = torch.randn([*batch_shape, input_dim])

        class Dynamic_Linear_Configuration(nn.Module):

            def __init__(self):
                super().__init__()
                self.configuration_layer = Linear.Linear(input_dim, dynamic)
                self.execution_layer = Linear.Linear(input_dim, output_dim, dynamic=dynamic)
            def forward(self, tensor: torch.Tensor)->torch.Tensor:
                configuration = self.configuration_layer()(tensor)
                configuration = torch.relu(configuration)
                configuration = configuration.movedim(-1, 0) #Notice the move required to place the dynamic dim to the front.
                output = self.execution_layer(configuration)(tensor)
                return output

        instance = Dynamic_Linear_Configuration()
        instance = torch.jit.script(instance) #Optional line. Scripts it
        output = instance(test_data)

    ```

    Notably, this is far from the only way to configure the dynamic layers. Of
    particular note, it is possible do use a sparse weight network instead. As
    long as you provide a hybrid sparse tensor, with the sparse dimensions
    defining the active dynamic kernels, the system will handle the rest.


    """
    ClosureType = LinearClosure
    def validate_dynamic(self,
                         dynamic: torch.Tensor,
                         shape: torch.Tensor,
                         task: Optional[str]):
        """
        Validates that the dynamic tensor is configured correctly, in whatever
        format it may come in.
        """

        dynamic_dim = dynamic.dim()
        shape_dim = shape.shape[0]
        if dynamic_dim < shape_dim:
            reason = f"""\
            The provided dynamic weights tensor had insufficient rank to 
            successfully apply dynamic construction. The tensor was found
            to have a rank of {dynamic_dim}, while the required number
            of dimensions were {shape_dim}
            """
            reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
            raise LinearFactoryException(reason, task)

        shape_as_list: List[int] = shape.tolist()
        if dynamic.shape[:shape_dim] != torch.Size(shape_as_list):
            dynamic_shape = dynamic.shape[:shape_dim]

            reason = f"""\
            The provided dynamics tensor does not properly match
            the dynamic_shape of the defined dynamic kernel and thus
            cannot be assembled. It was expected to start with
            dynamic_shape {torch.Size(shape)}, but what was actually 
            found was a tensor with dynamic_shape {dynamic_shape}            
            """
            reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
            raise LinearFactoryException(reason, task)
        if dynamic.is_sparse and dynamic.sparse_dim() != shape_dim:
            reason = f"""\
            The provided dynamic tensor is a sparse hybrid tensor
            of invalid nature. Due to the defined kernel, sparse
            hybrid tensors must have sparse rank of {shape_dim}
            however found {dynamic.sparse_dim()}
            """
            reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
            raise LinearFactoryException(reason, task)


    def __init__(self,
                 input_shape: src.supertransformerlib.Core.Functions.StandardShapeType,
                 output_shape: src.supertransformerlib.Core.Functions.StandardShapeType,
                 parallel: Optional[src.supertransformerlib.Core.Functions.StandardShapeType] = None,
                 dynamic: Optional[src.supertransformerlib.Core.Functions.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 use_bias: bool = True,
                 do_validation: bool = True,
                 ):
        """

        :param input_shape: What dynamic_shape, in either list or int format, to accept as inputs for linear
        :param output_shape: What dynamic_shape, in either list or int format, to accept as inputs for linear
        :param parallel: What dynamic_shape, in either list or int format, to setup as ensemble dimensions
        :param dynamic: What dynamic_shape, in either list or int format, to setup as dynamic dimensions
        :param dtype: The dtype. Defaults to float64
        :param device: The device. Defaults to CPU
        :param use_bias: Whether to use biass
        :param do_validation: Whether to do validation, or leave it to torch's native error code.
            Generally, the validations attached to the layer are a bit more informative.
        """

        super().__init__()

        task = "Creating a linear layer"
        input_shape = src.supertransformerlib.Core.Functions.standardize_shape(input_shape, 'input_shape', task=task)
        output_shape = src.supertransformerlib.Core.Functions.standardize_shape(output_shape, 'output_shape', task=task)

        if parallel is not None:
            parallel = src.supertransformerlib.Core.Functions.standardize_shape(parallel, 'parallel', task=task)
        if dynamic is not None:
            dynamic = src.supertransformerlib.Core.Functions.standardize_shape(dynamic, 'dynamic', task=task)


        self.input_shape = input_shape
        self.output_shape = output_shape
        self.parallel = parallel
        self.dynamic = dynamic
        self.validate = do_validation

        self.input_map = Reshape.ReshapeFactory(input_shape, input_shape.prod().unsqueeze(-1))
        self.output_map = Reshape.ReshapeFactory(output_shape.prod().unsqueeze(-1), output_shape)

        kernel = make_kernel(input_shape, output_shape, parallel, dynamic, device, dtype)
        self.kernel = nn.Parameter(kernel)

        if use_bias:
            bias = make_bias(output_shape, parallel, dynamic, device, dtype)
            self.bias = bias
        else:
            self.bias = None

    def forward(self, dynamic: Optional[torch.Tensor] = None, task: Optional[str] = None) -> LinearClosure:
        """
        The factory method for getting the linear closure which can
        then be utilized to apply the linear mechanism. Depending on the
        configuration, either requires no tensor, or a tensor of dynamic
        weights.

        :param dynamic:
        :param task:
        :return:
        """

        if self.dynamic is None and dynamic is not None:
            reason = f"""\
            Factory was called with non-None dynamic, but layer 
            was defined with no dynamic dimensions
            """
            reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
            raise LinearFactoryException(reason, task)
        if self.dynamic is not None and dynamic is None:
            reason = f"""\
            Factory was called without a dynamic, but
            it is the case that the linear layer is defined
            with dynamic dimensions
            """
            reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
            raise LinearFactoryException(reason, task)

        dynamic_shape = self.dynamic # Must copy line into local memory or torchscript throws a fit when refining types.
        if dynamic_shape is not None and dynamic is not None:
            torch.jit.annotate(torch.Tensor, dynamic)
            self.validate_dynamic(dynamic, dynamic_shape, task)

            matrix = self.kernel
            matrix = make_superposition(dynamic, matrix, dynamic_shape)

            bias = self.bias
            if bias is not None:
                bias = make_superposition(dynamic, bias, dynamic_shape)
            else:
                bias = None

        else:
            matrix = self.kernel
            bias = self.bias

        input_map = self.input_map()
        output_map = self.output_map()

        output = LinearClosure(matrix,
                               bias,
                               input_map,
                               output_map,
                               self.validate
                               )
        return output


