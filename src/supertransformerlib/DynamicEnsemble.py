import torch
from torch import nn
from typing import Union, Optional, List

"""

--- Design ----

The standard ensemble consists of a large number of parallel instances
of a model which are executed in parallel and "vote" on the correct result.

These do not do that. 

Instead, each parameter consists of a number of parallel kernels for whatever
process will be utilized, with the restriction that the first dimension must be
equal to parameter "ensemble_width". At some point during call, set, or
creation a "configuration" is passed in. This is a (..., dim_out, ensemble_width)
parameter which can be seen as performing matrix multiplication with the underlying
kernels, adding them together according to the weights in configuation, producing
a sort of superposition of the kernels. 

The configuration may be changed by the program, or even provided upon 
forward call. This provides the model with a variety of kernels which
are each of use in particular circumstances. 

Designwise, it is expected that the program will wish to tune its configuration
for the circumstances of each particular batch.
"""

class EnsembleSpace(nn.Module):
    """
    The base ensemble module.

    Contains methods designed to allow
    easy manipulation of ensemble constructs
    using broadcast mechanics.
    """
    @property
    def configuration(self)->torch.Tensor:
        return self._configuration


    def register_kernel(self, name: str, attribute: Optional[torch.Tensor] = None):
        """
        Registers an attribute as an ensemble kernel tensor.

        -- specifications --
        The attribute must be a tensor.
        The attributes first dimension must equal the length of ensemble.
        The attribute, when retrieved, will be returned according to configuration.

        --- params ---
        :param name: The name of the instance attribute
        :param attribute: The attribute to be assigned. If already assigned on class, do not have to specify again
        """
        #Sanity checks the attribute, then goes ahead and
        #registers the attribute with the registry parameter
        if attribute is None:
            #Retrieve the attribute
            if not hasattr(self, name):
                raise AttributeError("Attempt to access attribute of name %s which does not exist" % name)
            else:
                attribute = self.__getattribute__(name)
        if not isinstance(attribute, torch.Tensor):
            raise AttributeError("Ensemble attribute of name %s is not a tensor" % name)
        if attribute.dims() < 2:
            raise AttributeError("Ensemble attribute of name %s must have an ensemble dimension" % name)
        if attribute.shape[0] != self.native_ensemble_width:
            raise AttributeError("Ensemble attribute of name %s had first dim of length %s, expecting %s" % (
                name,
                attribute.shape[0],
                self.native_ensemble_width
            ))
        self.__setattr__(name, attribute)
        self.__registry.append(name)
    def construct_kernel(self, name)->torch.Tensor:
        """
        Construct an kernel ensemble based on the named attribute and
        the current configuration

        :param name: The attribute to construct
        :return: The constructed attribute
        """
        # This functions by performing a matrix multiplication
        # across the ensemble dimension. Notably, it is the case
        # that broadcasting is utilized to ensure that a configuration
        # of any shape, but ending in ensemble_width shape, is compatible.

        configuration = self.configuration #(..., ensemble_width)
        attribute: torch.Tensor = self.__getattribute__(name) #(ensemble_width, ..., dim1, dim0)

        if attribute.dim() == 2:
            #attribute shape: #(ensemble_width, dim0)
            output = torch.matmul(configuration, attribute)
        else:
            attribute = attribute.swapdims(-2, 0) #attribute: (dim1, ...., ensemble_width, dim0)
            required_broadcast_additions = attribute.dim() - 1
            for _ in range(required_broadcast_additions):
                configuration = configuration.unsqueeze(-2)
            output = torch.matmul(configuration, attribute)
            output = output.swapdims(-2, -attribute.dims())
            output = output.squeeze(dim= -attribute.dims())
        return output

    def __getattribute__(self, item):
        """
        Gets the given attribute.
        If the attribute is registered as a ensemble, applies the current configuration
        Else, does nothing."""
        if item in ("__registry", "configuration", "construct_ensemble", "__dict__"):
            return super().__getattribute__(item)
        if item in self.__registry:
            return self.construct_ensemble(item)
        else:
            return super().__getattribute__(item)
    def __init__(self, ensemble_width):
        super().__init__()
        self.native_ensemble_width = ensemble_width
        self.__registry = []

instance = EnsembleSpace(3)
intance = torch.jit.script(instance)