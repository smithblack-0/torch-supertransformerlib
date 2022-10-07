"""

The module for the ensemble
extended linear process.

"""
import math
import warnings
from typing import Union, List, Optional, Tuple, Callable

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

def torchFalse(device: torch.device):
    """
    Torchscript compatible false.

    Under certain cercumstances, tensor = True or tensor = False
    does not work. This is a quick way to ge a tensor for broadcast

    :return: Tensor false
    """
    return torch.tensor(True, dtype=torch.bool, device=device)

def torchTrue(device: torch.device):
    """
    Torchscript compatible true

    Under certain cercumstances, tensor = True or tensor = False
    does not work. This is a quick way to ge a tensor for broadcast

    :param device:
    :return: Tensor true
    """
    return torch.tensor(False, dtype=torch.bool, device=device)

def standardize_input(input: Union[torch.Tensor, List[int], Tuple[int, ...], int]) -> torch.Tensor:
    """
    Convert an input in one of three formats into a common single format of tensor
    Sanitize and throw errors if there is a problem
    """
    if not isinstance(input, (torch.Tensor, list, int, tuple)):
        raise ValueError("Illegal constructor argument. Type cannot be %s" % type(input))
    if isinstance(input, int):
        input = [input]
    output = torch.tensor(input, dtype=torch.int64)
    return output

def validate_shape_tensor(shape_tensor: torch.Tensor):
    """
    Validates that a given tensor represents a valid shape
    :param input: The shape tensor
    :raises: ValueError, if the shape tensor is invalid
    """
    if not isinstance(shape_tensor, torch.Tensor):
        raise ValueError("Shape tensor was not tensor")
    if torch.is_floating_point(shape_tensor):
        raise ValueError("Expected shape to be an int type. Got %s" % shape_tensor.dtype )
    if shape_tensor.dim() != 1:
        raise ValueError("shape was not 1d")
    if torch.any(shape_tensor < 0):
        raise ValueError("shape had negative dimensions, which are not allowed")

def validate_string_in_options(string: str,
                               stringname: str,
                               options: List[str],
                               options_name: str):
    """
    Validates that a string belongs to the
    given list. Raises useful error messages

    :param string: The string to test validity of
    :param stringname: What to call the string in error messages.
    :param options: The options the string could belong to
    :param options_name: What to call the options in error messages.

    :raises: ValueError if string not str
    :raises: ValueError, if string not in options mode
    """

    if not isinstance(string, str):
        msg = "The provided %s was not a string" % stringname
        raise ValueError(msg)
    if string not in options:
        msg = "The provide %s was '%s' \n" % (stringname, string)
        msg += "However, it could only be among %s  '%s'" % (options_name, str(options))
        raise ValueError(msg)

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





class AddressSpace(nn.Module):
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
            msg = " Error. In AddressSpace. Attempting to initialize a pointer which has already been initialized. \n"
            raise RuntimeError(msg)

        required_num_addresses = pointer_ids.numel()
        free_addresses = self._get_free_address_indices()
        if free_addresses.shape[0] < required_num_addresses:
            msg = "AddressSpace of Reservoir could not reserve address. Needed %s addresses , but only %s left"
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

        assert memory_addresses.dim() == 1, "On constructing AddressSpace: Addresses must be given as a 1d tensor"
        assert memory_addresses.dtype == torch.int64, "On constructing AddressSpace: Addresses must be a torch.int64"

        length = memory_addresses.shape[0]

        index = torch.arange(length, device=memory_addresses.device, dtype=torch.int64)
        key = -torch.ones_like(memory_addresses, device=memory_addresses.device, dtype=torch.int64)
        address_book = torch.stack([index, key, memory_addresses], dim=0)

        addresses = address_book
        self.register_buffer('addresses', addresses)
        self.warn = warn_on_overwrite


class ReservoirError(Exception):
    """

    An exception class for things going wrong
    while working with a dynamic kernel reservoir

    """
    def __init__(self,
                 type: str,
                 details: str,
                 task: str):
        msg = "A %s occurred while working with a dynamic kernel\n" % type
        msg += "This happened while doing task: %s \n" % task
        msg += "Additional details, if available, are: \n %s" % details



        super().__init__(msg)

class ReservoirKeyError(ReservoirError):
    """
    Occurs when something goes wrong when using a reservoir
    key or similar
    """
    def __init__(self,
                 task: str,
                 details: str,
                 key: Optional[torch.Tensor] = None,
                 ):


        self.task = task
        self.details = details
        self.key = key
        type = "ReservoirKeyError"
        super().__init__(type, details, task)

class ReservoirKernelError(ReservoirError):
    """
    Triggered when something goes wrong
    when assigning or initializing kernels
    directly
    """

    def __init__(self,
                 task: str,
                 details: str,
                 kernel: Optional[torch.Tensor] = None,
                 ):
        self.task = task
        self.details = details
        self.kernel = kernel
        type = "ReservoirKernelError"
        super().__init__(type, details, task)

class DynamicKernelReservoir(nn.Module):
    """
    The Dynamic Kernel Assembly Mechanism.

    When enabled, produces a block of parameters which can be
    selected from or superimposed together as though existing in a bunch
    of different ensembles. Each group in the assemble is identified by
    an integer starting at zero. The kernel accepts the configuration
    for the current round, superimposes its pieces together and otherwise
    selects and resets, and sets up for the current round.

    ----- construction ----

    Due to torchscript limitations, along with efficiency considerations, the kernel
    reservoir will allocate its memory all at once. As a result, two parameters
    are required to setup the reservoir. These are the number of parallel kernels
    tp allow, and the shape of the kernel. Optionally, the dtype and device may be
    specified

    ------ setup ------

    As originally setup, an undifferentiated block of empty kernels is
    developed. To actually usefully use them, the kernel blocks must be setup.
    This can be done in several different manners. One can setup using a
    string representing the init function to utilize. Alternatively, one
    can retrieve a block of the kernel of the right shape, then set the initialized
    value. Whatever the case, attempting to use a section which is not setup will
    cause an error.

     ----- Retrieval ----

    Various methods can then be utilized to retrieve information from
    the reservoir. Regardless of the method, one must request kernel
    dimensions by id. Attempting to retrieve an id that is not yet setup
    will result in an error. One can directly request a kernel to be at
    a particular location, or request a superposition of kernels instead.
    """

    #Data structure
    #
    # The internal data structure, called "_kernel_, consists of a tensor of shape
    # [num_allowed_kernels, shape...] where shape... is the shape
    # passed during init. Working alongside this is a bool of shape
    # [num_allowed_kernels], which keeps track of what has and has not been setup
    #
    # The raw manipulation is done using the __getitem__ and __setitem__ methods,
    # and is performed by providing the appropriate index when making a call.



    #Define status checking functions
    @torch.jit.export
    def num_reservoirs(self)->int:
        """ Return the total reservoirs available """
        return self._active.shape[0]

    @torch.jit.export
    def num_free(self)->int:
        """ Return the number of free reservoirs """
        free = self._active == torchFalse(self._active.device)
        return int(free.sum())

    @torch.jit.export
    def num_used(self)->int:
        """ Return the number of used reservoirs """
        used = self._active == torchTrue(self._active.device)
        return int(used.sum())

    @torch.jit.export
    def index_free(self)->torch.Tensor:
        """ Returns a tensor list of the indices free. Those not setup"""
        free = self._active == torchFalse(self._active.device)
        free_index = torch.arange(self.num_reservoirs(), device=self._active.device).masked_select(free)
        return free_index

    @torch.jit.export
    def index_used(self)->torch.Tensor:
        """ returns a tensor list of the indices used. Those that have been setup """
        used = self._active == torchTrue(self._active.device)
        used_index = torch.arange(self.num_reservoirs(), device=self._active.device).masked_select(used)
        return used_index

    #Define helper functions

    def _validate_raw_key(self, task: str, key: torch.Tensor):
        """
        Validates that a key will successfully
        access the kernel and active tensor.

        Raise an error if key will not work
        :raise: ReservoirKeyError - Keys are outsize of reservoir width
        :raise: ReservoirKeyError - Keys are wrong type.
        :raise: ReservoirKeyError - Keys are a float dtype
        :raise: ReservoirKeyError - Keys are not on the same device as layer.
        """

        if not isinstance(key, torch.Tensor):
            details = "Key was not a tensor"
            raise ReservoirKeyError(task, details)
        if key.dtype != torch.int64:
            details = "Key was not dtype of int64, but instead was %s" % key.dtype
            raise ReservoirKeyError(task, details, key)
        if key.device != self._kernel.device:
            details = "Key was not on the same device as kernel. Please use .to(...) to move it onto it first"
            raise ReservoirKeyError(task, details, key)
        if torch.any(key > self.num_reservoirs()):
            details = "Key pointed to kernels in reservoir greater than %s. \n " % self.num_reservoirs()
            details += "However, the reservoir only goes up to %s" % self.num_reservoirs()
            raise ReservoirKeyError(task, details, key)
        if torch.any(key < 0):
            details = "Key asked for kernels at negative indices \n"
            details += "However, the count for the reservoirs starts at zero"
            return ReservoirKeyError(task, details, key)

    def _validate_key_active(self,
                             task: str,
                             more_details: str,
                             key: torch.Tensor):
        """
        Validates that a key, which should be valid, is referring to an
        active kernel section.
        :raises: ReservoirKeyError if accessing active
        """
        if not torch.all(self._active[key]):
            details = "Key was attempting to access inactive indices. \n"
            details += more_details
            raise ReservoirKeyError(task, details, key)

    def _validate_key_inactive(self,
                               task: str,
                               more_details: str,
                               key: torch.Tensor):
        """
        Validates all elements in the key are pointing to inactive
        reservoirs.

        :param task: The current task
        :param more_details: Anything else to add to the error message
        :param key: The key to test.
        :raises: ReservoirKeyError
        """
        if not torch.all(torch.logical_not(self._active[key])):
            details = "Key was attempting to access active indices. \n"
            details += more_details
            raise ReservoirKeyError(task, details, key)

    def _validate_kernel(self,
                         task: str,
                         kernel: torch.Tensor):
        """
        Validates that a new block of tensors that will be
        assigned to our kernel is valid.

        :raises: ReservoirKernelError if type is wrong.
        :raises: ReservoirKernelError if dtype is wrong
        :raises: ReservoirKernelError if device is wrong
        :raises: ReservoirKernelError if shape is wrong
        """
        if not isinstance(kernel, torch.Tensor):
            details = "Provided kernel is not a tensor"
            raise ReservoirKernelError(task, details)
        if kernel.dtype != self._kernel.dtype:
            details = "Expected kernel dtype to be %s, but got %s \n" % (self._kernel.dtype, kernel.dtype)
            details += "You can use .to(...) to fix this"
            raise ReservoirKernelError(task, details, kernel)
        if kernel.device != self._kernel.device:
            details = "Expected kernel device to be %s, but got %s" % (self._kernel.device, kernel.device)
            details += "You can use .to(...) to fix this"
            raise ReservoirKernelError(task, details, kernel)
        if kernel.shape[1:] != self._kernel.shape[1:]:
            details = "Kernel shape invalid. Kernel provided of shape %s \n" % torch.tensor(kernel.shape[1:])
            details += "But class was initialized with shape %s" % torch.tensor(self._kernel.shape[1:])
            raise ReservoirKernelError(task, details, kernel)

    #Setup primary configuration methods for manipulation of kernel.

    def __getitem__(self, key: torch.Tensor)->torch.Tensor:
        """
        Allows direct manipulation of the kernel by id
        Returns a tuple of the associated kernel, and whether
        it has been initialized yet.
        """

        if self.validate:
            task = "While getting the raw kernel through __getitem__"
            self._validate_raw_key(task, key)
        return self._kernel[key].clone()

    def __setitem__(self, key: torch.Tensor, kernel: Optional[torch.Tensor]):
        """
        Allows direct manipulation of the kernel reservoir by
        addressing by index. Setting to a kernal with a tensor
        will automatically activate it. Meanwhile, setting to it
        with None will deactivate it.

        :param key: A int64 tensor containing the reservoirs to set to
        :param value: One of two things.
            * A tensor of shape [key_shape, kernel_shape] of kernel dtype and device
            * None,
            Either a tensor of kernel dtype of same shape
            as key
        :return:
        """
        with torch.no_grad():
            if kernel is None:
                if self.validate:
                    task = "Deleting Kernels from reservoirs by __setitem__"
                    self._validate_raw_key(task, key)
                    self._validate_key_active(task, key)
                self._active[key] = torchFalse(self._active.device)
            else:
                if self.validate:
                    task = "Setting Kernels to reservoirs by __setitem__"
                    self._validate_raw_key(task, key)
                    self._validate_kernel(task, kernel)
                    self._active[key] = torchTrue(self._active.device)
                    self._kernel[key] = kernel

    # Setup retrieval mechanisms
    def get_kernel_by_id(self, ids: torch.Tensor)->torch.Tensor:
        return self.__getitem__(ids)

    def __init__(self,
                 shape: Union[torch.Tensor, int, List[int]],
                 reservoir_size: int,
                 mode: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 validate: bool = True,
                 ):
        """

        :param shape: The shape of the kernel to setup
        :param reservoir_size: The maximum number of reservoir kernels to store
        :param mode: One of "id", "dense", or "sparse"
            * Id: When forward is called, the kernel for each id is found and returned. Default
            * Dense: When forward is called, the ids are looked up, multiplied by the weights, and superimposed.
            * Sparse: Almost the same as dense. But underlying logic is based on sparse logic, which can be more efficient when many weights are zero.
        :param dtype: The dtype of the kernel. Defaults to float32
        :param device: The device. Defaults to cpu
        """

        super().__init__()
        legal_modes = ["id", "sparse", "dense"]

        #Default filling.
        if mode is None:
            mode = "id"

        #Validation

        typing = "ConstructionError"
        task = "Validating reservoir size"
        if not isinstance(reservoir_size, int):
            details = "Reservoir size was not an int"
            raise ReservoirError(typing, details, task)
        if reservoir_size < 0:
            details = "Reservoir size is negative. This is not allowed"
            raise ReservoirError(typing, details, task)

        task = "Validating kernel shape"
        try:
            shape = standardize_input(shape)
            validate_shape_tensor(shape)
        except ValueError as err:
            raise ReservoirError(typing, str(err), task)
        except Exception as err:
            raise ReservoirError(typing, str(err), task)

        task = "Validating mode"
        try:
            validate_string_in_options(mode, "mode", legal_modes, "modes")
        except ValueError as err:
            details = str(err)
            raise ReservoirError(typing, details, task)

        #Setup
        compound_shape = torch.concat([torch.tensor([reservoir_size]), shape])
        kernel = torch.empty(torch.Size(compound_shape), device=device, dtype=dtype)
        kernel = nn.Parameter(kernel)

        self.shape = shape
        self.mode = mode
        self._reservoir_width = reservoir_size
        self._kernel = kernel
        self._active = torch.full([reservoir_size], False, dtype=torch.bool, device=device)
        self.validate = validate






class old:








    # Define status checkers
    @torch.jit.export
    def num_kernels(self)->int:
        """Returns the total number of kernels managed here"""
        return self._addressbook.num_addresses()

    @torch.jit.export
    def num_free(self)->int:
        """Returns the remaining free kernels"""
        return self._addressbook.num_free_addresses()

    @torch.jit.export
    def num_used(self)->int:
        """Returns"""
        return self._addressbook.num_used_addresses()

    #Define main methods
    @torch.jit.unused
    def setup_kernels(self, ids: torch.Tensor, init: Callable[[torch.Tensor], torch.Tensor]):
        """

        Sets up new kernels using the indicated ids and the provided init function.

        WARNING: Not torchscript compatible, instead you must use reserve kernels

        :param ids: A 1d int64 tensor consisting of various unique ids to assign things to
        :param init: The init function to call when initializing the kernel
        """

        assert isinstance(ids, torch.Tensor)
        assert ids.dim() == 1

        with torch.no_grad():
            self._addressbook.malloc(ids)
            addresses = self._addressbook.dereference(ids)

            num_ids = ids.shape[-1]
            new_kernels = torch.empty([num_ids] + list(self.shape),
                                      device = self._kernel.device,
                                      dtype = self._kernel.dtype)
            init(new_kernels)

            self._kernel[addresses, ...] = new_kernels

    @torch.jit.export
    def setup_blank_kernels(self, ids: torch.Tensor)->torch.Tensor:
        """
        Part of the dynamic torchscript mechanism. One can hand in
        a list of ids, and will be given an uninitialized sequence
        of kernels of the appropriate shape.

        This can then be initialized by the callee, and passed into
        reserve kernels.

        :param ids: A 1d int64 tensor consisting of various unique ids to assign things to
        """

        assert isinstance(ids, torch.Tensor)
        assert ids.dim() == 1

        with torch.no_grad():
            num_ids: int = ids.shape[-1]
            shape: List[int] = self.shape.tolist()
            shape = [num_ids] + shape
            size = torch.Size(shape)
            new_kernels = torch.empty(size,
                                      device = self._kernel.device,
                                      dtype = self._kernel.dtype)
        return new_kernels

    @torch.jit.export
    def reserve_kernels(self, ids: torch.Tensor, kernels: torch.Tensor):
        """

        Reserves kernels of the correct shape to be associated with the
        given ids.

        :param ids: A 1d int64 tensor representing unused ids to map kernels to
        :param kernels: A tensor of shape [ids_length, ...] consisting of kernels to store away
        """
        with torch.no_grad():
            self._addressbook.malloc(ids)
            addresses = self._addressbook.dereference(ids)
            self._kernel[addresses] = kernels

    @torch.jit.export
    def get_kernel_from_id(self, ids: torch.Tensor):
        """
        Gets the appropriate kernel directly from the ids.

        :param ids: The ids to retrieve from. A Nd int64 tensor
        :return: A ND + kernel_shape, int64 tensor
        """

        addresses = self._addressbook.dereference(ids)
        return self._kernel[addresses]

    @torch.jit.export
    def get_kernel_from_weights_sparse(self, ids: torch.Tensor, weights: torch.Tensor):

        """
        Accepts two ND tensors of the same shape. Uses sparse logic to multiply the weights
        by the associated kernels, and superimposes them together. Notably, weights with a
        value of zero do not contribute due to being sparse.


        :param ids: The ids to select from. The last dimension will be corrolated with the weights
            All else acts as batches.
        :param weights: The weights. A float or such tensor. Will multiply each of the resulting
            kernels and then add them together.
        :return: The kernel gotten by the superposition of the indicated ids by the indicated weights.


        """

        #Develop the raw index selections. These will be indices
        #pointing to the contigous locations containing
        #nonzero weights and ids.

        raw_index = torch.arange(ids.numel())
        is_not_zero = weights != 0.0
        raw_index = raw_index.masked_select(is_not_zero.flatten())

        #Develop the COO tensor index

        meshprep: List[torch.Tensor] = [torch.arange(i, device=self._kernel.device) for i in ids.shape]
        index = torch.meshgrid(meshprep, indexing='ij')
        index = torch.stack(index, dim=-1)
        index = index.flatten(0, -2)
        index = index[raw_index, :]
        index = index[..., :-1]
        output_shape = list(ids.shape[:-1]) + list(self._kernel.shape[1:])

        #Perform lookup.

        ids = ids.masked_select(is_not_zero)
        weights = weights.masked_select(is_not_zero)

        addresses = self._addressbook.dereference(ids)
        kernels = self._kernel[addresses]
        kernels =( kernels.transpose(0, -1) * weights).transpose(0, -1)



        output = torch.sparse_coo_tensor(indices=index.transpose(0, 1), values=kernels, size=output_shape)
        output = output.coalesce() #Sum is performed here
        output = output.to_dense()

        return output

    @torch.jit.export
    def get_kernel_from_weights_dense(self, ids: torch.Tensor, weights: torch.Tensor):
        """
        Accepts two ND tensors of the same shape.
        :param ids: The ids to select from. The last dimension will be corrolated with the weights
            All else acts as batches.
        :param weights: The weights. A float or such tensor. Will multiply each of the resulting
            kernels and then add them together.
        :return: The kernel gotten by the superposition of the indicated ids by the indicated weights.
        """

        addresses = self._addressbook.dereference(ids)
        kernels = self._kernel[addresses]
        while weights.dim() < kernels.dim():
            weights = weights.unsqueeze(-1)
        kernels = kernels*weights
        kernels = kernels.sum(dim = ids.dim() - 1)
        return kernels

    @torch.jit.export
    def free(self, ids: torch.Tensor):
        """
        Frees up ids that are currently attached to kernels, allowing for
        the parameters to be reused.

        :param ids: The ids to free
        :raises NullPtr: If the id was not found or has already been freed.
        """

        self._addressbook.free(ids)



    def forward(self, ids: torch.Tensor, weights: Optional[torch.Tensor] = None)->torch.Tensor:
        if self.mode == "id":
            if weights is not None:
                raise RuntimeError("Passing weight when in id mode")

            return self.get_kernel_from_id(ids)
        elif self.mode == "dense":
            if weights is None:
                raise RuntimeError("Not passing weights when in dense mode")
            return self.get_kernel_from_weights_dense(ids, weights)
        elif self.mode == "sparse":
            if weights is None:
                raise RuntimeError("Not passing weights when in sparse mode")
            return self.get_kernel_from_weights_sparse(ids, weights)

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

    ---- Dynamic Kernel Assembly ----



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


