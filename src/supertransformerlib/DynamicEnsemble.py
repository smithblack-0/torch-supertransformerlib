import math

import torch
from torch import nn
from typing import Union, Optional, List
from src.supertransformerlib import Glimpses
from src.supertransformerlib.Core import KernelSpace

"""
--- Design ----

The standard ensemble consists of a large number of parallel instances
of a model which are executed in parallel and "vote" on the correct result.

These do not do that. 

Instead, each parameter consists of a number of parallel kernels for whatever
process will be utilized, with the restriction that the first dimension must be
equal to parameter "ensemble_width". At some point during call, set, or
creation a "configuration" is passed in. This is a (..., dim_out, ensemble_width)
parameter which can be seen as performing matrix multiplication with the underlying
kernels, adding them together according to the weights in configuation, producing
a sort of superposition of the kernels. 

The configuration may be changed by the program, or even provided upon 
forward call. This provides the model with a variety of kernels which
are each of use in particular circumstances. 

Designwise, it is expected that the program will wish to tune its configuration
for the circumstances of each particular batch.
"""


class Linear(KernelSpace):
    """
    A linear layer allowing a number of tricks to be deployed in parallel.
    These tricks are:

    * Linear mapping
    * Autoreshaping
    * Parallel execution
    * Dynamic Kernel Assembly

    Generally, the resolution order for a arbitrary tensor is

    tensor(...batch_stuff, parallel_dims..., autoreshaping)

    ---- Linear mapping ---

    Linear mapping is what this actually does. It is a linear layer.
    In the simple case of a definition as Linear(3, 5), this executes
    by taking a tensor of shape (3) to a tensor of shape (5) by a web of dense
    connections

    It is also the case that the standard paradynm is followed for broadcasting.
    That is, a tensor of shape tensor(10, 3) will be transformed by Instance(tensor)
    into tensor(10, 5), by a broadcast across the undefined dimensions.

    ---- Autoreshaping ---

    One neat trick, and common situation, is a need to trnasform a
    whole collection of dimension by either flattening or expanding.
    This linear layer has that covered.

    One can provide a list of values into the layer, and it will
    subsequently flatten the sectors shown, and reshape as appropriate,
    so long as the quantity of tensor nodes matches.

    For instance defining Linear([3,4], 15) would be able to transform a tensor of shape
    tensor(5, 3, 4) into a tensor of shape tensor(5, 15). Likewise, one may define
    Linear(15, [12, 4]) to transform the tensor above into tensor(5, 12, 4)

    ---- Parallel execution ----

    Under some circumstances, it is useful to have many
    independent linear layers all operating in parallel
    at the same time. This is what the parallel execution
    is designed for. The command here creates a larger
    kernel capable of independently addressing in the tensor.

    Lets see how it works.

    Consider tensor(10, 23, 5). If we wanted to execute 23 independent linear
    operations in parallel, Linear(5, 7, parallel=23), yielding tensor(10, 23, 7)
    Additionally, we could go deeper. Defining Linear(5, 7, [10, 23]) would make a
    kernel capable of addressing the entire tensor at once.

    This has its greatest utility when designing independent ensembles. However,
    exchange of information has it's purpose. Which brings us to...

    ---- Dynamic Kernel Assembly ----

    Although it is possible to create completely parallel tensor kernels, it is possible
    to do much more. Using the parallel attribute indicated above actually activates the
    dynamic ensemble mechanism built into the class. This mechanism allows correctly
    defined specifications to create a KernelSpace of different Linear kernels, which
    the program may then combine together for various purposes based on a Configuration
    attribute.

    consider a tensor of shape tensor(3, 5, 10). If we were to define instance=Linear(10, 5, dynamic=20),
    it is the case that a randomly configured Linear layer with an ensemble section of width 20
    would be created. Lets say for a moment we want to configure linear so that each ensemble is
    drawn from equally. Then

    instance.configuration = torch.ones([20])

    Would do the trick, by weighting each option equally. It is also possible to specify the configuration
    more directly. Any extra dimensions will pop out right before the batch info starts. See
    KernelSpace for more details.
    """
    def standardize_input(self, input: Union[torch.Tensor, List[int], int])->torch.Tensor:
        """
        Convert an input in one of three formats into a common single format of tensor
        Sanitize and throw errors if there is a problem
        """
        if not isinstance(input, (torch.Tensor, list, int)):
            raise ValueError("Illegal constructor argument")
        if isinstance(input, int):
            input = [input]
        output = torch.tensor(input, dtype=torch.int64)
        return output

    def __init__(self,
                 input_shape: Union[torch.Tensor, List[int], int],
                 output_shape: Union[torch.Tensor, List[int], int],
                 parallel: Optional[Union[torch.Tensor, List[int], int]] = None,
                 dynamics: Optional[Union[torch.Tensor, List[int], int]] = None,
                 use_bias: bool = True,
                 top_k: Optional[int] = None,
                 top_p: Optional[int] = None

                 ):
        """

        :param input_shape: The shape of the input
        :param output_shape: The shape of the output
        :param parallel: What parallel portions of the kernel to create
        :param dynamics: What dynamics to create, or what config to make
        :param use_bias: Whether or not to use bias on the linear layer
        :param top_k: The top k to use. Only active if dynamics is active
        :param top_p: The top p to use. Only active if dynamics is active.
        """

        #Peform standardization
        if parallel is None:
            parallel = []
        if dynamics is None:
            dynamics = []

        input_shape = self.standardize_input(input_shape)
        output_shape = self.standardize_input(output_shape)
        parallel = self.standardize_input(parallel)
        dynamics = self.standardize_input(dynamics)

        #Begin developing kernel shapes and conversions

        matrix_rows = input_shape.prod().unsqueeze(-1)
        matrix_columns = output_shape.prod().unsqueeze(-1)
        matrix_shape = torch.concat([matrix_rows, matrix_columns], dim=0)

        if use_bias:
            bias_shape = matrix_columns

        input_autoshape_mapping = (input_shape, matrix_rows)
        output_autoshape_mapping = (matrix_columns, output_shape)

        #Introduce modifications to account for parallelization.
        #This consists of additional indepedent dimensions
        #at the front of the matrix and bias

        matrix_shape = torch.concat([parallel, matrix_shape], dim=0)
        if use_bias:
            bias_shape = torch.concat([parallel, bias_shape])

        #Handle dynamics.
        #
        #Make sure to set the release dimension
        #flag if dynamics are not being utilized

        if dynamics.dim() > 0:
            ensemble_width = dynamics.shape[-1]
            self.is_ensemble = True
        else:
            ensemble_width = 1

            self.is_ensemble = False
        super().__init__(ensemble_width, top_k=top_k, top_p=top_p)
        matrix_shape = torch.concat([torch.tensor([ensemble_width]), matrix_shape])
        if use_bias:
            bias_shape = torch.concat([torch.tensor([ensemble_width]), bias_shape])

        #Generate actual kernels

        matrix_kernel = torch.empty(matrix_shape)
        torch.nn.init.kaiming_uniform_(matrix_kernel, math.sqrt(5))
        matrix_kernel = nn.Parameter(matrix_kernel)

        if use_bias:
            bias_kernel = torch.zeros(bias_shape)
            bias_kernel = nn.Parameter(bias_kernel)

        #Register kernels and deployment details

        self.use_bias = use_bias

        self.input_map_reference = input_autoshape_mapping
        self.output_map_reference = output_autoshape_mapping

        self.matrix_kernel = matrix_kernel
        self.register_ensemble("matrix_kernel")

        if use_bias:
            self.bias_kernel = bias_kernel
            self.register_ensemble("bias_kernel")

    def forward(self, tensor: torch.Tensor):

        input_shape, row_length = self.input_map_reference
        column_length, output_shape = self.output_map_reference
        assert torch.Size(input_shape) == tensor.shape[-len(input_shape):], "Tensor and kernel shapes not compatible"

        flattened_input = Glimpses.reshape(tensor,input_shape, row_length)
        if self.use_bias:
            flattened_output = torch.matmul(self.matrix_kernel, flattened_input) + self.bias_kernel
        else:
            flattened_output = torch.matmul(self.matrix_kernel, flattened_input)
        restored_output = Glimpses.reshape(flattened_output, column_length, output_shape)

        return restored_output







class Linear(nn.Module):
    """

    A Linear layer allowing head-dependent linear processing of data from shape
    to shape. JIT is supported as an instance.

    An instance is made by providing a list of head_shapes,
    an input_shape tuple, an output_shape tuple.

    This is then used to initialize a head dependent linear remap
    from input shape to output shape. That will then be accessed
    through the instance call

    It is expected that the input format will be in the form of

    [..., heads, input_shape]

    Returning something of format

    [..., heads, output_shape]


    Letting the head_shape parameter be none will disable it, resulting in broadcasting. Input
    shape, output shape, and head_shapes may all be just an integer, in which case it is
    assumed only a single dimension is involved.

    """

    def __init__(self,
                 input_shape: Union[torch.Tensor, List[int], int],
                 output_shape: Union[torch.Tensor, List[int], int],
                 ensemble_shapes: Optional[Union[torch.Tensor, List[int], int]] = None):
        """

        :param input_shape: The shape of the input. May be an int, or a list/tuple of ints,
            or a tensor
        :param output_shape: The shape of the output. May be an int, or a list/tuple of ints,
            or a tensor
        :param ensemble_shapes: The size of the ensemble dimensions.
        :param ensemble_dims: The dimensions on which the ensemble is found.
        """
        # Super call

        super().__init__()

        # Implicit conversion

        if ensemble_shapes is None:
            ensemble_shapes = []
        elif isinstance(ensemble_shapes, int):
            ensemble_shapes = [ensemble_shapes]
        elif torch.is_tensor(ensemble_shapes) and ensemble_shapes.dim() == 0:
            ensemble_shapes = [ensemble_shapes]
        if isinstance(input_shape, int):
            input_shape = [input_shape]
        elif torch.is_tensor(input_shape) and input_shape.dim() == 0:
            input_shape = [input_shape]
        if isinstance(output_shape, int):
            output_shape = [output_shape]
        elif torch.is_tensor(output_shape) and output_shape.dim() == 0:
            output_shape = [output_shape]

        input_shape = torch.tensor(input_shape, dtype=torch.int64)
        output_shape = torch.tensor(output_shape, dtype=torch.int64)
        head_shapes = torch.tensor(ensemble_shapes, dtype=torch.int64)

        # Create kernel and bias. These include head dimensions if provided.

        if head_shapes is not None:

            kernel_shape = [*head_shapes, output_shape.prod(), input_shape.prod()]
            bias_shape = [*head_shapes, output_shape.prod()]
        else:
            kernel_shape = [output_shape.prod(), input_shape.prod()]
            bias_shape = [output_shape.prod()]

        kernel = torch.zeros(kernel_shape, requires_grad=True)
        kernel = torch.nn.init.kaiming_uniform_(kernel, a=math.sqrt(5))

        bias = torch.zeros(bias_shape, requires_grad=True)
        bias = torch.nn.init.zeros_(bias)

        # Store shapes and kernels

        self._input_shape = input_shape
        self._output_shape = output_shape

        self._kernel = nn.Parameter(kernel)
        self._bias = nn.Parameter(bias)

    def forward(self, tensor: torch.Tensor):
        """

        :param tensor: The tensor to perform linear operations with. Given in [..., ensemble, d_model] or [..., d_model]
        :return:
        """

        # Flatten the relevent dimensions

        tensor = Glimpses.reshape(tensor, self._input_shape, int(self._input_shape.prod()))

        # Perform primary processing. Add an extra dimension on the end
        # of the input tensor to handle the matrix multiply, perform
        # matrix multiply, then add bias

        tensor = tensor.unsqueeze(-1)
        tensor = self._kernel.matmul(tensor)
        tensor = tensor.squeeze(-1)
        tensor = tensor + self._bias

        # Restore the dimensions, then return
        tensor = Glimpses.reshape(tensor, int(self._output_shape.prod()), self._output_shape)
        return tensor