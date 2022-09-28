"""

The module for the ensemble
extended linear process.

"""
import math
from typing import Union, List, Optional

import torch
import torch.nn
from torch import nn

from . import Glimpses


class Utility:
    """ A place for utility methods to belong"""

    def standardize_input(self, input: Union[torch.Tensor, List[int], int]) -> torch.Tensor:
        """
        Convert an input in one of three formats into a common single format of tensor
        Sanitize and throw errors if there is a problem
        """
        if not isinstance(input, (torch.Tensor, list, int)):
            raise ValueError("Illegal constructor argument. Type cannot be %s" % type(input))
        if isinstance(input, int):
            input = [input]
        output = torch.tensor(input, dtype=torch.int64)
        return output


class Linear(Utility, nn.Module):
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

    ---- Combination ----

    when both parallization and dynamic configuration is present, the order of resolution across dimensions,
    from right to left, is first the autoshaping specifications, then the parallelization, and finally the
    dynamic specifications.

    """

    def __init__(self,
                 input_shape: Union[torch.Tensor, List[int], int],
                 output_shape: Union[torch.Tensor, List[int], int],
                 parallel: Optional[Union[torch.Tensor, List[int], int]] = None,
                 use_bias: bool = True,
                 ):
        """

        :param input_shape: The shape of the input
        :param output_shape: The shape of the output
        :param parallel: What parallel portions of the kernel to create
        :param dynamics: If defined, what the dynamics width is.
        :param use_bias: Whether or not to use bias on the linear layer
        :param tags: Any tags to restrict config updates to. Passed on to our kernels.
        """

        # Peform standardization
        if parallel is None:
            parallel = []

        input_shape = self.standardize_input(input_shape)
        output_shape = self.standardize_input(output_shape)
        parallel = self.standardize_input(parallel)

        # Begin developing kernel shapes and conversions

        matrix_rows = input_shape.prod().unsqueeze(-1)
        matrix_columns = output_shape.prod().unsqueeze(-1)
        matrix_shape = torch.concat([matrix_columns, matrix_rows], dim=0)

        if use_bias:
            bias_shape = matrix_columns

        input_autoshape_mapping = (input_shape, matrix_rows)
        output_autoshape_mapping = (matrix_columns, output_shape)

        # Introduce modifications to account for parallelization.
        # This consists of additional indepedent dimensions
        # at the front of the matrix and bias

        matrix_shape = torch.concat([parallel, matrix_shape], dim=0)
        if use_bias:
            bias_shape = torch.concat([parallel, bias_shape])

        super().__init__()

        # Generate actual kernels

        matrix_kernel = torch.empty(matrix_shape.tolist())
        torch.nn.init.kaiming_uniform_(matrix_kernel, math.sqrt(5))
        matrix_kernel = nn.Parameter(matrix_kernel)

        if use_bias:
            bias_kernel = torch.zeros(bias_shape.tolist())
            bias_kernel = nn.Parameter(bias_kernel)

        # Register kernels and deployment details

        self.use_bias = use_bias

        self.input_map_reference = input_autoshape_mapping
        self.output_map_reference = output_autoshape_mapping

        self.matrix_kernel = matrix_kernel
        if use_bias:
            self.bias_kernel = bias_kernel

    def forward(self, tensor: torch.Tensor):

        input_shape, row_length = self.input_map_reference
        column_length, output_shape = self.output_map_reference

        flattened_input = Glimpses.reshape(tensor, input_shape, row_length)
        flattened_input = flattened_input.unsqueeze(-1)

        if self.use_bias:
            flattened_output = torch.matmul(self.matrix_kernel, flattened_input).squeeze(-1)
            flattened_output = flattened_output + self.bias_kernel
        else:
            flattened_output = torch.matmul(self.matrix_kernel, flattened_input)
        restored_output = Glimpses.reshape(flattened_output, column_length, output_shape)
        return restored_output
