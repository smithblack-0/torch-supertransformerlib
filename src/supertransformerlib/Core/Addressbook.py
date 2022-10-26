import torch
import torch.jit
from torch import nn


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

        :param pointer_ids: A int64 tensor of arbitrary dynamic_shape, representing possible pointer variables.
        :return: A bool tensor of dynamic_shape pointer_ids, indicating whether or not it is contained within that variable.
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

        :param pointer_ids: A int64 tensor of arbitrary dynamic_shape but dim > 0. Represents required maps.
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

        :param pointer_ids: An int64 tensor representing the pointers to dereference. May be any dynamic_shape
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