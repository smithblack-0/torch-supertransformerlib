"""

A place for the core superposition driver kernel
to lay. This is a mechanism which allows for
superpositions to be elegantly defined with
minimal fuss, and executed later on.

"""
from typing import Callable, Optional

import torch
from torch import nn
from . import Reshape
from . import Core


class KernelSetupError(Core.ValidationError):
    def __init__(self, reason: Optional[str], task: Optional[str] ):
        type = "Kernel Setup Error"
        super().__init__(type, reason, task)

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
    :param shape: The dynamic shape. Note that due to hybrid tensor restrictions the shape should
        come as [...shape, ...batch_shape], since sparse dimensions must remain sparse.
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
    # shape.

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

torch.jit.script(make_sparse_superposition)

class Parameter(nn.Module):
    """
    A container for holding a parameter when doing
    superposition logic. When called, it returns
    the parameter itself. It should be setup with
    the specifications and an init function, then
    can be called with appropriate weights to get
    the superposition

    Once the parameter is setup, it may be called. Depending
    on the setup, it either needs to be called with superposition weight,
    or nothing. Regardless, the return will be the parameter underlying the
    mechanism which will have shape "kernel_shape"
    """

    def __init__(self,
                 init_func_: Callable[[torch.Tensor], torch.Tensor],
                 kernel_shape: Core.StandardShapeType,
                 superposition_shape: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        :param init_func: An init function accepting a tensor and performing in-place initialization of the values
        :param kernel_shape: The shape of the kernel we are making
        :param superposition_shape: The shape of the superposition parts of the kernel. May be none
        :param dtype: The dtype. can be none
        :param device: The device. can be none
        """

        task = "setting up parameter"
        kernel_shape = Core.standardize_shape(kernel_shape, "kernel_shape", task=task)
        if superposition_shape is not None:
            superposition_shape = Core.standardize_shape(superposition_shape, "superposition_shape", task = task)

        if superposition_shape is not None:
            final_shape = torch.concat([superposition_shape, kernel_shape], dim =0)
        else:
            final_shape = kernel_shape

        parameter = torch.zeros(final_shape, dtype=dtype, device=device)
        init_func_(parameter)
        parameter = nn.Parameter(parameter)

        self.Superposition_Shape = superposition_shape
        self.Kernel_Shape = kernel_shape
        self.Parameter = parameter


    def forward(self, superposition_weights: Optional[torch.Tensor] = None):
        pass
