"""

The module for the ensemble
extended linear process.

"""
import math
import warnings
from typing import Union, List, Optional, Tuple

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
                 input_map: Tuple[torch.Tensor, torch.Tensor],
                 output_map: Tuple[torch.Tensor, torch.Tensor],
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

        flattened_input = Glimpses.reshape(tensor, input_shape, row_length)
        flattened_input = flattened_input.unsqueeze(-1)

        flattened_output = torch.matmul(self.kernel, flattened_input).squeeze(-1)
        flattened_output = flattened_output + self.bias

        restored_output = Glimpses.reshape(flattened_output, column_length, output_shape)
        return restored_output

class NullPtr(Exception):
    """
    A exception class for null pointers in the address book
    """
    def __init__(self,
                 task: str,
                 ids: torch.Tensor,
                 ):

        msg = "Null (Python level, not C) Pointer Exception\n"
        msg += "This occurred while doing task: %s \n" % task
        msg += "This was found to occur at pointers in tensor indices: \n\n"
        msg += "%s" % ids

        self.ids = ids
        self.task = task
        super().__init__(msg)

class BadPtr(Exception):
    """
    An exception class for bad pointers, whatever that may mean
    """
    def __init__(self,
                 task: str,
                 reason: str,
                 ids: torch.Tensor):
        msg = "Bad pointer (Python level, not C) Exception encountered\n"
        msg += "This occurred while doing task: %s \n" % task
        msg += "The reason it is bad is: %s" % reason
        msg += "This was found to occur at pointers in tensor indices: \n\n"
        msg += "%s" % ids



class AddressBook(nn.Module):
    """
    A small helper class, the address book
    allows for the elegant management of reservoir addresses.

    It contains the addresses available and knows which are active
    and which are not. It displays related statistics when requested,
    and will find a free address to place information when requested

    Pointers can be assigned and looked up by an integer variable.
    """

    # Data structure Addresses is split into three parts.
    #
    # Layer 1 is simply an index
    # Layer 2 is the pointer key
    # Layer 3 is the pointed at location.

    def _get_used_address_indices(self) -> torch.Tensor:
        # Figures out what addresses are used, and returns their index
        #
        # An address is used if the key is greater than zero

        index = self.addresses[0]
        ids = self.addresses[1]
        used = ids >= 0
        return index.masked_select(used)

    def _get_free_address_indices(self) -> torch.Tensor:
        # Figures out what addresses are free. Then return
        # the associated index.
        #
        # An address is free if its key is zero.

        index = self.addresses[0]
        ids = self.addresses[1]
        free = ids < 0
        return index.masked_select(free)

    def _get_pointer_indices(self, pointer_ids: torch.Tensor, task: str):
        used_addr_indices = self._get_used_address_indices()
        used_keys = self.addresses[1, used_addr_indices]

        # Check all combinations of indices and pointer ids
        # at once. Do this by letting dimension -1 be the dimension
        # representing the stored key, dimension -2 represent the
        # the pointer id, and the intersection represent whether or not
        # the key and pointer id are the same.

        broadcast_keys = used_keys.unsqueeze(-2)
        broadcast_pointers = pointer_ids.unsqueeze(-1)
        usage_mesh = broadcast_keys == broadcast_pointers

        # Handle error conditions, then dereference. Throw
        # errors if more than one copy of the pointer exists,
        # or if the pointer is null.
        #
        # (..., pointer_dim, keys_dim)

        reference_count = usage_mesh.sum(dim=-1)
        if torch.any(reference_count == 0):
            bad_indices = torch.argwhere(reference_count == 0)
            raise NullPtr(task, bad_indices)

        if torch.any(reference_count >= 2):
            reason = "Pointer pointing to more than one address"
            bad_indices = torch.argwhere(reference_count > 1)
            raise BadPtr(task, reason, bad_indices)

        chosen_addr_indices = used_addr_indices.masked_select(usage_mesh).view(pointer_ids.shape)
        return chosen_addr_indices

    @torch.jit.export
    def peek(self, pointer_ids: torch.Tensor)->torch.Tensor:
        """
        Checks for every id in pointer ids, whether or not that id
        is currently pointing to an address. In other words, checks if
        pointer is not

        :param pointer_ids: A int64 tensor of arbitrary shape, representing possible pointer variables.
        :return: A bool tensor of shape pointer_ids, indicating whether or not it is contained within that variable.
        :raises BadPtr: If something has gone horribly wrong in the memory management.

        """

        assert pointer_ids.dim() > 0
        assert pointer_ids.dtype == torch.int64
        assert torch.all(pointer_ids >= 0)

        used_addr_indices = self._get_used_address_indices()
        used_keys = self.addresses[1, used_addr_indices]

        # Check all combinations of indices and pointer ids
        # at once. Do this by letting dimension -1 be the dimension
        # representing the stored key, dimension -2 represent the
        # the pointer id, and the intersection represent whether or not
        # the key and pointer id are the same.

        broadcast_keys = used_keys.unsqueeze(-2)
        broadcast_pointers = pointer_ids.unsqueeze(-1)
        usage_mesh = broadcast_keys == broadcast_pointers

        reference_count = usage_mesh.sum(dim=-1)
        if torch.any(reference_count >= 2):
            task = "Peeking at allocated addresses"
            reason = "Pointer pointing to more than one address"
            bad_indices = torch.argwhere(reference_count > 1)
            raise BadPtr(task, reason, bad_indices)

        return reference_count == 1

    @torch.jit.export
    def malloc(self, pointer_ids: torch.Tensor):
        """
        Take in a list of ints, called pointer ids. These represent
        variables which can be connected to a pointer. Connect each of these
        pointers to a concrete, free memory kernel address.

        :param pointer_ids: A int64 tensor of arbitrary shape but dim > 0. Represents required maps.
        :return: An into list as a tensor, representing the addresses of concern.
        :raise: Runtime error, if insufficient addresses remain.
        """

        assert pointer_ids.dim() == 1
        assert pointer_ids.dtype == torch.int64
        assert torch.all(pointer_ids >= 0)

        #Handle preassigned element ignoring.
        preassigned = self.peek(pointer_ids)
        if torch.any(preassigned):
            msg = " Error. In AddressBook. Attempting to initialize a pointer which has already been initialized. \n"
            raise RuntimeError(msg)

        required_num_addresses = pointer_ids.numel()
        free_addresses = self._get_free_address_indices()
        if free_addresses.shape[0] < required_num_addresses:
            msg = "AddressBook of Reservoir could not reserve address. Needed %s addresses , but only %s left"
            msg = msg % (pointer_ids.shape[0], free_addresses.shape[0])
            raise RuntimeError(msg)
        reserving = free_addresses[:required_num_addresses]
        self.addresses[1, reserving] = pointer_ids

    @torch.jit.export
    def dereference(self, pointer_ids: torch.Tensor):
        """
        Take a list of pointer ids. Dereferences them, telling us
        what memory address it is pointing to. Do it quickly.

        :param pointer_ids: An int64 tensor representing the pointers to dereference. May be any shape
        :return: The dereferenced addresses, whatever they may be.
        :raise NullPtr: If pointer is not found.
        :raise BadPtr: If something goes terribly wrong.
        """

        assert pointer_ids.dtype == torch.int64
        assert torch.all(pointer_ids >= 0)

        task = "Dereferencing_pointers"
        index = self._get_pointer_indices(pointer_ids, task)
        chosen_addresses = self.addresses[2, index]
        return chosen_addresses

    @torch.jit.export
    def free(self, pointer_ids: torch.Tensor):
        """
        Free up the addresses and thus memory
        that these pointers were previously pointing to.

        :param pointer_ids: An int64 tensor. Represents the pointers to free. Pointers freed will no longer derefernce.
        :raise: NullPtrException - When attempting to free a pointer that was never reserved
        """

        assert pointer_ids.dim() == 1
        assert pointer_ids.dtype == torch.int64
        assert torch.all(pointer_ids >= 0)

        task = "Freeing Addresses"
        index = self._get_pointer_indices(pointer_ids, task)
        self.addresses[1, index] = -1

    def __init__(self,
                 memory_addresses: torch.int64,
                 warn_on_overwrite: bool = True):

        super().__init__()

        assert memory_addresses.dim() == 1, "On constructing AddressBook: Addresses must be given as a 1d tensor"
        assert memory_addresses.dtype == torch.int64, "On constructing AddressBook: Addresses must be a torch.int64"

        length = memory_addresses.shape[0]

        index = torch.arange(length, device=memory_addresses.device, dtype=torch.int64)
        key = -torch.ones_like(memory_addresses, device=memory_addresses.device, dtype=torch.int64)
        address_book = torch.stack([index, key, memory_addresses], dim=0)

        addresses = address_book
        self.register_buffer('addresses', addresses)
        self.warn = warn_on_overwrite


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



    """
    ForwardType = _Linear_Forward
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

        matrix_rows: torch.Tensor = input_shape.prod().unsqueeze(-1)
        matrix_columns: torch.Tensor = output_shape.prod().unsqueeze(-1)
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


