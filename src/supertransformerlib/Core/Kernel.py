"""

A place for the core superposition driver kernel
to lay. This is a mechanism which allows for
superpositions to be elegantly defined with
minimal fuss, and executed later on.

"""
from typing import Callable, Optional

import torch
from torch import nn

import src.supertransformerlib.Core.Errors as Errors
import src.supertransformerlib.Core.Functions as Functions
import src.supertransformerlib.Core.Reshape as Reshape
import src.supertransformerlib.Core.StringUtil as StringUtil


class KernelSetupError(Errors.ValidationError):
    def __init__(self,
                 reason: Optional[str],
                 task: Optional[str],
                 weights: Optional[torch.Tensor],
                 kernel: torch.Tensor
                 ):
        type = "Kernel Setup Error"
        self.weights = weights
        self.kernel = kernel
        super().__init__(type, reason, task)

def make_dense_superposition(dynamics: torch.Tensor,
                             kernel: torch.Tensor,
                             dynamic_shape: torch.Tensor) -> torch.Tensor:
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
                              shape: torch.Tensor):
    """
    Make a superposition out of the kernel when the dynamic
    weights are sparse. The ones which are not present are
    ignored.

    :param dynamics: The dynamic kernel. Expected to be sparse
    :param kernel: The kernel.
    :param shape: The dynamic dynamic_shape. Note that due to hybrid tensor restrictions the dynamic_shape should
        come as [...dynamic_shape, ...batch_shape], since sparse dimensions must remain sparse.
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
                 kernel_shape: Functions.StandardShapeType,
                 superposition_shape: Optional[Functions.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        :param init_func: An init function accepting a tensor and performing in-place initialization of the values
        :param kernel_shape: The dynamic_shape of the kernel we are making
        :param superposition_shape: The dynamic_shape of the superposition parts of the kernel. May be none
        :param dtype: The dtype. can be none
        :param device: The device. can be none
        """

        super().__init__()
        task = "setting up parameter"
        kernel_shape = Functions.standardize_shape(kernel_shape, "kernel_shape", task=task)
        if superposition_shape is not None:
            superposition_shape = Functions.standardize_shape(superposition_shape, "superposition_shape", task = task)

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


    def forward(self, superposition_weights: Optional[torch.Tensor] = None, task: Optional[str] = None):

        if self.Superposition_Shape is None:
            # Boring case.
            #
            # Just verify that we are not providing a superposition when unneeded, then
            # go return the kernel
            if superposition_weights is not None:
                reason = f"""\
                The kernel was defined with no superposition dimensions.
                
                However, it is the case a superposition weight was provided
                on construction. This is not allowed
                """
                raise KernelSetupError(reason, task, superposition_weights, self.Parameter)
            return self.Parameter

        else:
            # Exciting case
            # Go ahead and do a bunch of validation. Once we are at the end
            # of the validation chain, either do a sparse or a dense
            # superposition construction

            if superposition_weights is None:
                reason = f"""\
                The kernel is being built with a superposition weight
                of dynamic_shape {self.Superposition_Shape}. However, it is in
                fact the case that no superposition weights were provided.
                
                This is not allowed.
                """
                reason = StringUtil.dedent(reason)
                raise KernelSetupError(reason, task, superposition_weights, self.Parameter)

            interesting_length = self.Superposition_Shape.shape[0]
            if superposition_weights.shape[:interesting_length] != torch.Size(self.Superposition_Shape):
                if superposition_weights.shape[-interesting_length:] == torch.Size(self.Superposition_Shape):
                    front = torch.tensor(superposition_weights.shape[:-interesting_length], dtype=torch.int64)
                    back = torch.tensor(superposition_weights.shape[-interesting_length:], dtype=torch.int64)
                    right_shape = torch.Size(torch.concat([back, front], dim=0))

                    reason = f"""\
                    The superposition weights are almost valid. Due to limitations with 
                    hybrid tensors, it is the case batch dimensions need to be placed after
                    the weight dimensions. 
                    
                    You provided a tensor 'superposition_weights' of shape {torch.Size(superposition_weights.shape)} which
                    is attempting to build the superposition defined by {torch.Size(self.Superposition_Shape)}. However, 
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
                    reason = f"""\
                    The kernel is being built with an expected weight dynamic_shape 
                    of {torch.Size(self.Superposition_Shape)}. However, it is the 
                    the case provided parameter 'superposition_weights' under the forward
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
                Tensor.to(dtype) to fix this
                """
                reason = StringUtil.dedent(reason)
                raise KernelSetupError(reason, task, superposition_weights, self.Parameter)
            if superposition_weights.device != self.Parameter.device:
                reason = f"""\
                The kernel is being built under the assumption it is 
                located on device {self.Parameter.device}. However, 
                the superposition weights are instead located on device
                {superposition_weights.device}. 
                
                Since these are not the same, kernel construction failed.
                
                Use Tensor.to(device) to fix this
                """
                reason = StringUtil.dedent(reason)
                raise KernelSetupError(reason, task, superposition_weights, self.Parameter)

            # Do either sparse or dense superposition construction
            if superposition_weights.is_sparse:
                return make_sparse_superposition(superposition_weights, self.Parameter, self.Superposition_Shape)
            else:
                return make_dense_superposition(superposition_weights, self.Parameter, self.Superposition_Shape)
