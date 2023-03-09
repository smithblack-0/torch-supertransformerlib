"""

A place for the core superposition driver kernel
to lay. This is a mechanism which allows for
superpositions to be elegantly defined with
minimal fuss, and executed later on.

"""
from typing import Callable, Optional

import torch
from torch import nn

from . import errors as Errors
from . import reshape_module as Reshape
from . import string_util as StringUtil
from . import functions
class KernelSetupError(Errors.ValidationError):
    """
    An error which occurred while setting
    up the kernel
    """
    def __init__(self,
                 reason: Optional[str],
                 task: Optional[str],
                 weights: Optional[torch.Tensor],
                 kernel: torch.Tensor
                 ):
        typing = "Kernel Setup Error"
        self.weights = weights
        self.kernel = kernel
        super().__init__(typing, reason, task)

def make_dense_superposition(dynamics: torch.Tensor,
                             kernel: torch.Tensor,
                             dynamic_shape: torch.Tensor,
                             ) -> torch.Tensor:
    """
    Makes a superposition of the kernel out of the dynamics weights
    """

    length = dynamic_shape.shape[0]
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
                              shape: torch.Tensor,
                              task: Optional[str]):
    """
    Make a superposition out of the kernel when the dynamic
    weights are sparse. The ones which are not present are
    ignored.

    :param dynamics: The dynamic kernel. Expected to be sparse
    :param kernel: The kernel.
    :param shape: The dynamic dynamic_shape. Note that due to hybrid tensor restrictions
           the dynamic_shape should
        come as [...dynamic_shape, ...batch_shape], since sparse dimensions must remain sparse.
    :return: The superimposed kernel
    """

    # Flatten the tensors so that the dynamic dimensions are
    # found in one lump sum.

    length = shape.shape[0]
    input_shape = shape
    output_shape = torch.tensor([int(shape.prod())])
    dynamics = Reshape.reshape(dynamics, input_shape, output_shape, task=task)
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
    dynamics = torch.sparse_coo_tensor(dynamics.indices(),
                                       dynamic_values,
                                       size=dynamic_update_shape)

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
    mechanism which will have dynamic_shape "kernel_shape"
    """

    def __init__(self,
                 init_func_: Callable[[torch.Tensor], torch.Tensor],
                 kernel_shape: functions.StandardShapeType,
                 superposition_shape: Optional[functions.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        :param init_func: An init function accepting a tensor and performing in-place
                          initialization of the values
        :param kernel_shape: The dynamic_shape of the kernel we are making
        :param superposition_shape: The dynamic_shape of the superposition parts of the kernel.
               May be none
        :param dtype: The dtype. can be none
        :param device: The device. can be none
        """

        super().__init__()
        task = "setting up parameter"
        kernel_shape = functions.standardize_shape(kernel_shape, "kernel_shape", task=task)
        if superposition_shape is not None:
            superposition_shape = functions.standardize_shape(superposition_shape,
                                                              "superposition_shape",
                                                            task = task)

        if superposition_shape is not None:
            final_shape = torch.concat([superposition_shape, kernel_shape], dim =0)
        else:
            final_shape = kernel_shape
        final_shape = torch.Size(final_shape)

        parameter = torch.zeros(final_shape, dtype=dtype, device=device)
        init_func_(parameter)
        parameter = nn.Parameter(parameter)

        self.Superposition_Shape = superposition_shape
        self.Kernel_Shape = kernel_shape
        self.Parameter = parameter


    def forward(self,
                superposition_weights: Optional[torch.Tensor] = None,
                weights_name: Optional[str] = "superposition_weights",
                task: Optional[str] = None):

        if self.Superposition_Shape is None:
            # Boring case.
            #
            # Just verify that we are not providing a superposition when unneeded, then
            # go return the kernel
            if superposition_weights is not None:
                reason = f"""\
                The kernel was defined with no superposition dimensions.
                However, it is the case parameter '{weights_name}' was provided 
                with something during the forward call. To prevent hard to debug errors,
                this is not allowed.
                
                Either stop passing in a weights tensor, or define a superposition shape. 
                """
                reason = StringUtil.dedent(reason)
                raise KernelSetupError(reason, task, superposition_weights, self.Parameter)
            return self.Parameter

        else:
            # Exciting case
            # Go ahead and do a bunch of validation. Once we are at the end
            # of the validation chain, either do a sparse or a dense
            # superposition construction

            if superposition_weights is None:
                reason = f"""\
                The Parameter is being built while defined to have a superposition
                shape of {self.Superposition_Shape}. However, it is in
                fact the case that no parameter '{weights_name}' was
                provided when forward was called.
                
                This is not allowed.
                """
                reason = StringUtil.dedent(reason)
                raise KernelSetupError(reason, task, superposition_weights, self.Parameter)

            interesting_length = self.Superposition_Shape.shape[0]
            size = torch.Size(functions.get_shape_as_list(self.Superposition_Shape))
            if superposition_weights.shape[:interesting_length] != size:
                if superposition_weights.shape[-interesting_length:] == size:
                    front = torch.tensor(superposition_weights.shape[:-interesting_length],
                                         dtype=torch.int64)
                    back = torch.tensor(superposition_weights.shape[-interesting_length:],
                                        dtype=torch.int64)
                    error_shape = functions.get_shape_as_list(torch.concat([back, front], dim=0))
                    right_shape = torch.Size(error_shape)


                    reason = f"""\
                    The superposition weights are almost valid. Due to limitations with 
                    hybrid tensors, it is the case batch dimensions need to be placed after
                    the weight dimensions. 
                    
                    You provided a tensor '{weights_name}' of shape {torch.Size(superposition_weights.shape)} which
                    is attempting to build the superposition defined by {torch.Size(functions.get_shape_as_list(self.Superposition_Shape))}. However, 
                    it is the case that the tensor should have in fact had a shape of {right_shape}. That is, 
                    the superposition should have been defined, then the batch elements. 
                    
                    This format is necessary due to torch hybrid tensor placing sparse dimensions 
                    first. It is less confusing to just require all weights to be written in this format
                    than to have to learn a new format when using sparse calls. Please simply reorder 
                    your dimensions to be compatible.
                    """
                    reason = StringUtil.dedent(reason)
                    raise KernelSetupError(reason, task, superposition_weights, self.Parameter)
                else:
                    shaping = functions.get_shape_as_list(self.Superposition_Shape)

                    reason = f"""\
                    The kernel is being built with an expected weight dynamic_shape 
                    of {torch.Size(shaping)}. However, it is the 
                    the case provided parameter '{weights_name}' under the forward
                    call had a size of {superposition_weights.shape}.
                    
                    Since these do not match, kernel construction failed
                    """
                    reason = StringUtil.dedent(reason)
                    raise KernelSetupError(reason, task, superposition_weights, self.Parameter)
            if superposition_weights.dtype != self.Parameter.dtype:
                reason = f"""\
                The kernel is being build under an expected weights 
                datatype of {self.Parameter.dtype}. However, the 
                superposition weights had a dtype of {superposition_weights.dtype}
                
                Since these are not the same, kernel construction failed. Use
                Tensor.to(dtype={self.Parameter.dtype}) to fix this, but make 
                sure you are not passing in the wrong tensor first!
                """
                reason = StringUtil.dedent(reason)
                raise KernelSetupError(reason, task, superposition_weights, self.Parameter)
            if superposition_weights.device != self.Parameter.device:
                reason = f"""\
                The kernel is being built under the assumption it is 
                located on device {self.Parameter.device}. However, 
                the parameter {weights_name} are instead located on device
                {superposition_weights.device}. 
                
                Since these are not the same, kernel construction failed.
                
                Use Tensor.to(device={self.Parameter.device}) to fix this,
                but make sure you are passing in the right tensor first!
                """
                reason = StringUtil.dedent(reason)
                raise KernelSetupError(reason, task, superposition_weights, self.Parameter)

            # Do either sparse or dense superposition construction
            if task is not None:
                task = task + ": Superimposing using superposition weights"
            if superposition_weights.is_sparse:
                return make_sparse_superposition(superposition_weights,
                                                 self.Parameter,
                                                 self.Superposition_Shape,
                                                 task)
            else:
                return make_dense_superposition(superposition_weights,
                                                self.Parameter,
                                                self.Superposition_Shape,
                                                )
