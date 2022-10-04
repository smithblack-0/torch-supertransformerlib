"""


Dynamic module

Dynamic consists of methods for allowing the program to configure
its own architecture and data structures, generally by means of
attention based adaptive pointers.

It is designed to basically allow many different ensembles
to exist and be utilized in parallel while only running
the ensembles actually immediately in use.


# Design

Dynamic layers consist of a reservour and execution class,
and a compatible configuration tensor. It is designed to
maximize computational efficiency at the cost of memory.




Each element of the

"""

import torch
from torch import nn
from typing import List, Callable, Optional




class ReservoirKernel(nn.Module):
    """
    A small helper class to make life a little bit more easy on the
    programmer. The ReservoirKernel represents a series of kernels of
    a particular shape under the hood.

    The kernel must first be initialized with a shape and an init
    function. This sets up the class.

    After this point, it is possible to tell the kernel to grow reservour
    subkernels that can be drawn upon by passing an unused id into the
    grow function. This will create a new tensor of the right shape,
    initialize it, then store it away under the corresponding id.

    After growing a reservoir, it is possible to use it. Passing
    in a Long or Int tensor will return, for each element, a new
    tensor in which the element was sampled from the corrosponding
    reservoir location. This allows per element configuration.

    It is entirely possible to delete reservoir kernels as well. One
    can pass the requested id into delete to free up that slot.

    It should be noted that in order to ensure torch plays nice
    with this system and operates efficiently, memory is preallocated.
    As a result, you must set on initialization the maximum number of
    allowable reservoir kernels, and cannot exceed that number at any point.

    As is to be expected, it is the case that the device and
    the datatype must be set on initialization.
    """
    def __init__(self,
                 shape: torch.Tensor,
                 init_function: Callable,
                 max_kernels: int,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        Sets up the limitations of the reservoir

        :param shape: The shape each kernel should be
        :param init_function: The function used to grow new kernels.
        :param dtype: The dype of the kernel
        :param device: The device of the kernel.
        """
        #Store away important info
        self.shape = shape
        self.dtype = dtype

        #Reserve all space for the entire kernel.

        compound_shape = torch.concat([torch.tensor([max_kernels]), shape])
        kernel = torch.zeros(compound_shape, device=device, dtype = dtype)
        kernel = nn.Parameter(kernel)
        self.kernel = kernel


        #Reserve the address space. Addresses which are zero

        addresses = torch.zeros([kernel])

        #Reserve the

        self._init = init_function


class BaseReservoir(nn.Module):
    """
    The Reservoir consists of all of the tensor kernels
    which we are trying to train. Multiple of these tensor kernels
    exist in parallel within the reservoir, representing the different
    configurations which may be called upon. Each different configuration
    is numbered starting at 0 and going to n-1. When the

    A Reservoir must implement one function. This function
    is get_forward. get_forward should accept a long tensor and
    look through the
    """