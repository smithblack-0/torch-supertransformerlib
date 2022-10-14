"""

The module for the ensemble
extended linear process.

"""
import textwrap
from typing import Union, List, Tuple, Optional, Dict

import torch
import torch.nn
from torch import nn

class ValidationError(Exception):
    """
    An error class for validation problems
    """
    def __init__(self,
                 type: str,
                 reason: str,
                 tasks: Optional[List[str]] = None
                 ):

        msg = ""
        msg += "A validation error occurred: %s \n" % type
        msg += "The error occurred because: \n\n %s\n" %reason
        if tasks is not None:
            tasks = "\n".join(tasks)
            msg += "This occurred while doing tasks: \n\n%s" % tasks
        super().__init__(msg)

# String manipulation.
#
# Torchscript shuts down a lot of string manipulation tools. This
# revives the ones I need.

@torch.jit.script
def dedent(string: str)->str:
    """
    A torchscript compatible dedent, since textwrap's dedent
    does not work properly. Takes and eliminates common among
    the beginning of each line. Required to prevent error messages
    from looking weird.

    Quick and dirty. That is okay. This is only utilized when
    raising error messages.

    Edge cases involving tab or other such nasties are not explicitly handled,
    beware

    :param string: The string to dedent
    :return: The dedented string
    """
    lines = string.split("\n")

    #Figure out how much whitespace to removed by looking through
    #all the lines and keeping the shortest amount of whitespace.
    #
    #Then shorten all dimensions by that amount
    has_viewed_a_line_flag = False
    amount_of_whitespace_to_remove = 0
    for line in lines:
        whitespace = len(line) - len(line.lstrip())
        if not has_viewed_a_line_flag:
            amount_of_whitespace_to_remove = whitespace
            has_viewed_a_line_flag = True
        else:
            amount_of_whitespace_to_remove = min(whitespace, amount_of_whitespace_to_remove)

    output: List[str] = []
    for line in lines:
        updated_line = line[amount_of_whitespace_to_remove:]
        output.append(updated_line)

    output = "\n".join(output)
    return output

@torch.jit.script
def format(string: str, substitutions: Dict[str, str])->str:
    """
    Performs a formatting action on a string in a torchscript
    compatible manner. Does not support positional substitutions
    or escape sequences.

    :param string: The string to perform substitutions on
    :param substitutions: The substitutions to perform as would be done with str.format(keywords)
    :return: The formatted string
    """
    for key, value in substitutions.items():
        key = "{" + key +"}"
        string = string.replace(key, value)
    return string


def update_or_start_tasks(update: str, tasks: Optional[List[str]])->List[str]:
    """
    Updates the error handling task list. This is a stack of
    information on the current task. If there is none, start
    one.

    :param update: Whatever should be included in the tasks
    :param tasks: The current task chain. Or none
    :return: The updated task chain
    """

    if tasks is None:
        tasks = [update]
    else:
        tasks = tasks.copy()
        tasks.append(update)

    return tasks


class StandardizationError(ValidationError):
    """
    Called when something cannot be converted to a
    shape tensor for whatever reason
    """
    def __init__(self, reason: str, tasks: Optional[List[str]] = None ):
        type = "StandardizationError"
        super().__init__(type, reason, tasks)


StandardShapeType = Union[torch.Tensor, List[int], int]

@torch.jit.script
def standardize_shape(input: StandardShapeType,
                      input_name: str,
                      allow_negatives: bool = False,
                      allow_zeros: bool = False,
                      tasks: Optional[List[str]] = None,
                      ) -> torch.Tensor:
    """
    Converts a sequence of things representing the shape
    of a tensor in one of several formats into a single
    standard format of a 1D tensor. Performs some
    validation as well


    :param input: One of the possible input formats
    :param input_name: The name of the thing being standardized. Used to generate helpful error messages
    :param allow_negatives: Whether negative elements are allowed in the tensor shape
    :param allow_zeros: Whether zero elements are allowed in the tensor shape
    :param tasks: The task trace, used to make nice error messages.
    :return: A 1D tensor consisting of a valid definition of a tensor's shape.
    """
    tasks = update_or_start_tasks(f"Standardizing %s" % input_name, tasks)
    #Turn options into tensors
    if isinstance(input, int):
        output = torch.tensor([input], dtype=torch.int64)
    elif isinstance(input, list):
        output = torch.tensor(input, dtype=torch.int64)
    elif isinstance(input, torch.Tensor):

        if input.dim() != 1:
            dims = input.dim()
            reason = f"""\
            Expected parameter '{input_name}' to receive 1d tensor representing
            shape. Number of dimensions was actually %{dims}
            """
            reason = dedent(reason)
            raise StandardizationError(reason, tasks)
        if torch.is_floating_point(input):
            tensor_type = input.dtype
            reason = f"""\
            Expected parameter '{input_name}' to receive an integer tensor type. 
            However, actually a floating point tensor of type {tensor_type} 
            """
            reason = dedent(reason)
            raise StandardizationError(reason, tasks)
        if torch.is_complex(input):
            tensor_type = input.dtype
            reason = f"""\
            Expected parameter '{input_name}' to receive an integer of an integer tensor
            type. However, actually recieved a complex type of {tensor_type}
            """
            reason = dedent(reason)
            raise StandardizationError(reason, tasks)
        output = input
    else:
        input_type = type(input)
        reason = f"""\
        Expected parameter '{input_name}' to be one of 
        int, List[int], or torch.Tensor. But instead found {input_type}
        """
        reason = dedent(reason)
        raise StandardizationError(reason, tasks)
    if not allow_negatives and torch.any(output < 0):
        reason = f"""\
        Expected parameter '{input_name}' to consist of no elements
        less than zero. This was not satisfied. 
        """
        reason = dedent(reason)
        raise StandardizationError(reason, tasks)
    if not allow_zeros and torch.any(output == 0):
        reason = f"""\
        Expected parameter '{input_name}' to consist of no elements
        equal to zero. This was not satisfied
        """
        reason = dedent(reason)
        raise StandardizationError(reason, tasks)

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





