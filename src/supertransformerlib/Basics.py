import math
from collections import namedtuple
from typing import Tuple, Optional, Union, List

import torch
import torch.jit
from torch import nn

from src.supertransformerlib import Glimpses
from src.supertransformerlib.Core import standardize_shape


@torch.jit.script
class _Linear_Forward:
    """
    The linear forward mechanism.

    Accepts the input map, the output map,
    kernel, and bias. Produced by kernel
    loader.

    The batch mask parameter can be used to
    tell the layer to not run calculations
    on certain batch dimensions.
    """

    def __init__(self,
                 input_map: Glimpses.Reshape,
                 output_map: Glimpses.Reshape,
                 kernel: torch.Tensor,
                 bias: torch.Tensor,
                 ):
        self.input_map = input_map
        self.output_map = output_map
        self.kernel = kernel
        self.bias = bias

    def __call__(self,
                tensor: torch.Tensor,
                batch_mask: Optional[torch.Tensor] = None,
                ):
        input_shape, row_length = self.input_map
        column_length, output_shape = self.output_map

        flattened_input = self.input_map(tensor)
        flattened_input = flattened_input.unsqueeze(-1)

        flattened_output = torch.matmul(self.kernel, flattened_input).squeeze(-1)
        flattened_output = flattened_output + self.bias

        restored_output = self.output_map(tensor)
        return restored_output


ReshapeStub = namedtuple("reshape_map", ["input_shape", "output_shape"])
MatrixShape = namedtuple("MatrixShapeStub", ["rows", "columns"])


class Linear(nn.Module):
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

    ---- Dynamic Kernel Assembly ----



    """
    ForwardType = _Linear_Forward
    def make_stubs(self,
                           input_shape: torch.Tensor,
                           output_shape: torch.Tensor
                           )->Tuple[MatrixShape, Glimpses.Reshape, Glimpses.Reshape]:

        """
        Makes the various shape stubs needed
        to calculate map and shape parameters.
        :param input_shape:
        :param output_shape:
        :return:
        """

        rows = input_shape.prod()
        columns = output_shape.prod()
        shape = MatrixShape(rows, columns)

        InputMap = Glimpses.Reshape(input_shape, rows)
        OutputMap = Glimpses.Reshape(columns, output_shape)
        return shape, InputMap, OutputMap


    def __init__(self,
                 input_shape: Union[torch.Tensor, List[int], int],
                 output_shape: Union[torch.Tensor, List[int], int],
                 parallel: Optional[Union[torch.Tensor, List[int], int]] = None,
                 dynamics: Optional[Union[torch.Tensor, List[int], int]] = None,
                 use_bias: bool = True,
                 ):
        """

        :param input_shape: The shape of the input
        :param output_shape: The shape of the output
        :param parallel: What parallel portions of the kernel to create
        :param dynamics: If defined, what parallel dimensions to define as "dynamic".
        :param use_bias: Whether or not to use bias on the linear layer
        """


        input_shape = standardize_shape(input_shape)
        output_shape = standardize_shape(output_shape)
        if parallel is not None:
            parallel = standardize_shape(parallel)
        if dynamics is not None:
            dynamics = standardize_shape(dynamics)

        matrix_spec, input_map, output_map = self.make_stubs(input_shape, output_shape)


        # Begin developing kernel shapes and conversions

        matrix_rows: torch.Tensor = matrix_spec.rows.unsqueeze(-1)
        matrix_columns: torch.Tensor = matrix_spec.columns.unsqueeze(-1)
        matrix_shape = torch.concat([matrix_columns, matrix_rows], dim=0)

        if use_bias:
            bias_shape = matrix_columns

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
        self.input_map = input_map
        self.output_map = output_map

        self.matrix_kernel = matrix_kernel
        if use_bias:
            self.bias_kernel = bias_kernel
        else:
            self.bias_kernel = torch.zeros([0])

    def setup_forward(self)->_Linear_Forward:
        """
        Returns (basically) a torchscript passible linear forward function
        call.

        """


        if self.use_bias:
            return _Linear_Forward(self.input_map_reference,
                                   self.output_map_reference,
                                   self.matrix_kernel,
                                   self.bias_kernel)
        else:
            return _Linear_Forward(self.input_map_reference,
                                   self.output_map_reference,
                                   self.matrix_kernel,
                                   torch.zeros([1], device=self.matrix_kernel.device))

    def forward(self, tensor: torch.Tensor):
        """
        Execute the forward mechanism.
        :param tensor:
        :return:
        """
        # Torchscript does not like to pass around layers, but I do.
        # To get around this, the majority of the feedforward mechanism is
        # located in the _Linear_Forward class.

        forward_call = self.setup_forward()
        return forward_call(tensor)


@torch.jit.script
class ViewPoint:
    """
    A callable function for performing a viewpoint operation.

    Is given a number of views, a view width,
    a weight tensor, and an index tensor. May then
    be called to create views of segments of a text.

    Should be created by the ViewPointFactory.
    """
    def __init__(self,
                 views: int,
                 view_width: int,
                 weights: torch.Tensor,
                 index: torch.Tensor,
                 ):
        """

        :param views: The number of views
        :param view_width: The view width
        :param weights: The weights tensor. In shape [..., views, query, top_k]
        :param index: The index tensor. In shape [..., views, query, top_k]
        """
        self.view_width = view_width
        self.views = views
        self.weights = weights
        self.index = index

    def __call__(self, tensor: torch.Tensor)->torch.Tensor:


        #Generate the draw source. This will be a memory efficient strided
        #view of the input tensor

        strided_source = tensor.unsqueeze(0).transpose(0, -1).squeeze(-1)
        strided_source = Glimpses.dilocal(strided_source,
                                          self.view_width,
                                          1,
                                          [1]) # (..parallel), viewpoint, item, local, viewpoint_dim)
        strided_source = strided_source.squeeze(-3)
        strided_source = strided_source.unsqueeze(-1).transpose(-1, 0).squeeze(0)

        #The following code sets up a gather
        #
        #This consists of getting the index and strided source the same shape,
        #then using expand to ensure gather will select all required index.
        #
        #Gather basically expects, along each non-gathered dimension, a
        #list indicating what elements to grab.

        index = self.index.unsqueeze(-1).unsqueeze(-1)
        strided_source = strided_source.unsqueeze(-4).unsqueeze(-4)

        index_expansion = [-1]*index.dim()
        source_expansion = [-1]*strided_source.dim()

        source_expansion[-4] = index.shape[-4]
        source_expansion[-5] = index.shape[-5]
        index_expansion[-1] = strided_source.shape[-1]
        index_expansion[-2] = strided_source.shape[-2]

        index = index.expand(index_expansion)
        strided_source = strided_source.expand(source_expansion)
        gathered_viewpoint = torch.gather(strided_source, dim=-3, index=  index)

        #Weight the viewpoints, combine them together, then return the result

        gathered_viewpoint = gathered_viewpoint*self.weights.unsqueeze(-1).unsqueeze(-1)
        output = gathered_viewpoint.sum(-3)
        return output


class ViewPointFactory(nn.Module):
    """
    A factory for making a ViewPoint, which is
    a map that returns a sequence of tensors with each
    tensor being corrolated to a distinct section of
    text.
    """

    def __init__(self,
                 d_query,
                 d_key,
                 viewpoints: int,
                 viewpoint_width: int,
                 top_k: int,
                 parallelization: Optional[Union[torch.Tensor, List[int], int]] = None,
                 ):
        super().__init__()
        d_viewpoint = d_query // viewpoints
        self.viewpoints = viewpoints
        self.query_projector = Linear(d_query, [viewpoints, d_viewpoint], parallelization)
        self.key_projector = Linear(d_key, [viewpoints, d_viewpoint], parallelization)
        self.width = viewpoint_width
        self.top_k = top_k

    def forward(self, query, key) -> ViewPoint:
        # Generate the viewpoint dims
        query = query.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, , (..parallel), embedding)
        key = key.unsqueeze(0).transpose(-2, 0).squeeze(-2)  # (item, , (..parallel), embedding)

        viewpoint_query = self.query_projector(query)  # (item ..., , (..parallel), viewpoint, viewpoint_dim)
        viewpoint_key = self.key_projector(key)  # (item ..., , (..parallel), viewpoint, viewpoint_dim)

        viewpoint_query = viewpoint_query.unsqueeze(-2).transpose(0, -2).squeeze(
            0)  # ...,, (..parallel), viewpoint, item, viewpoint_dim)
        viewpoint_key = viewpoint_key.unsqueeze(-2).transpose(0, -2).squeeze(
            0)  # ..., , (..parallel), viewpoint, item, viewpoint_dim)

        # Sort out the candidates. Generate the index.

        score = torch.matmul(viewpoint_query, viewpoint_key.transpose(-1, -2))
        score, index = torch.sort(score, dim=-1, descending=True)
        index = index[..., :self.top_k]
        score = score[..., :self.top_k]
        score = torch.softmax(score, dim=-1)

        #Return the viewpoint.

        return ViewPoint(self.viewpoints,
                         self.width,
                         score,
                         index)