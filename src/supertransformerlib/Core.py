"""

The module for the ensemble
extended linear process.

"""
import math
from typing import Union, List, Optional, Dict

import torch
from torch import nn

from . import Glimpses




class EnsembleSpace(nn.Module):
    """
    The base ensemble module.

    Contains methods designed to allow
    easy manipulation of ensembles into a dynamically
    configurable kernel.
    """
    @property
    def configuration(self)->torch.Tensor:
        return self._configuration
    @configuration.setter
    def configuration(self, config: torch.Tensor):
        if not isinstance(config, torch.Tensor):
            raise ValueError("Configuration must be a tensor")
        if config.dtype != torch.float:
            raise ValueError("Configuration was not float")
        if config.dim() < 2:
            raise ValueError("Configuration must have two or more dimesions")
        elif config.shape[-1] != self.native_ensemble_width:
            raise ValueError("Configuration last dimension bad. Expected %s, got %s"
                             % (self.native_ensemble_width, self.configuration.shape[-1]))
        #Handle construction, whether it be a top-p, top-k, or standard instance

        configuration = config
        if self.top_k is not None:
            # Create the top k sparse configuration
            #
            # Do this by finding the top k indices, making
            # a mask, then masking the configuration and performing
            # the probabilistic change.
            _, indices = torch.topk(configuration, self.top_k, dim=-1)
            indices = indices.unsqueeze(-1)
            mask = torch.arange(self.native_ensemble_width)
            mask = indices == mask
            mask = torch.any(mask, dim=-2)
            configuration = configuration * mask
            configuration = torch.softmax(configuration, dim=-1)

        elif self.top_p is not None:
            ### Develop the top-p driven case
            #
            # This will consist of finding by cumulative sums the
            # sorted layer at which we transition above the top
            # p, then generating and applying a mask based on these
            # indices.

            #Extract the relevant indices in an efficient manner

            configuration = torch.softmax(configuration, dim=-1)
            sorted, indices = torch.sort(configuration, descending=True, dim=-1)

            #Promote interesting dimension to front for zippiing
            sorted = sorted.unsqueeze(0).transpose(0, -1).squeeze(-1)
            indices = indices.unsqueeze(0).transpose(0, -1).squeeze(-1)

            #Setup and extract indices
            cumulative_probability = torch.zeros(configuration.shape[:-1])
            mask_construction_indices: List[torch.Tensor] = []
            off = torch.full(cumulative_probability.shape, -1)
            for probabilities, index_layer in zip(sorted, indices):
                cumulative_probability_lower_then_p = cumulative_probability < self.top_p
                mask_update = torch.where(cumulative_probability_lower_then_p, index_layer, off)
                mask_construction_indices.append(mask_update)
                cumulative_probability += probabilities
                if torch.all(cumulative_probability >= self.top_p):
                    break

            mask_indices = torch.stack(mask_construction_indices, dim=-1).unsqueeze(-1)
            mask = mask_indices == torch.arange(self.native_ensemble_width)
            mask = torch.any(mask, dim=-2)
            configuration = configuration*mask
            configuration = torch.softmax(configuration, dim=-1)
        self._configuration = configuration

    def set_configuration(self, configuration: torch.Tensor):
        """
        Recursively sets this layers configuration and
        the configuration of all detected child layers

        This can be utilized to have a large synced
        group.

        :param configuration: The configuration to set
        """
        for child in self.children():
            if isinstance(child, EnsembleSpace):
                child.set_configuration(configuration)

    def register_ensemble(self, name: str, attribute: Optional[torch.Tensor] = None):
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
                print(self.matrix_kernel)
                print(getattr(self, "matrix_kernel"))
                attribute = super().__getattribute__(name)
                attribute = self.__getattribute__(name)
        if not isinstance(attribute, torch.Tensor):
            raise AttributeError("Ensemble attribute of name %s is not a tensor" % name)
        if attribute.dim() < 2:
            raise AttributeError("Ensemble attribute of name %s must have an ensemble dimension" % name)
        if attribute.shape[0] != self.native_ensemble_width:
            raise AttributeError("Ensemble attribute of name %s had first dim of length %s, expecting %s" % (
                name,
                attribute.shape[0],
                self.native_ensemble_width
            ))
        self._registry[name] = attribute

    def construct_ensemble(self, name)->torch.Tensor:
        """
        Construct a kernel in which all but the last
        configuration entry is broadcast onto the end
        of the named tensor.

        Perform matrix multiplication to collapse the last
        tensor.

        :param name: The attribute to construct
        :return: The constructed attribute
        """
        attribute = self._registry[name]#(ensemble_width, ..., dim1, dim0)
        configuration = self.configuration #(..., ensemble_width)


        # This functions by performing a matrix multiplication
        # across the ensemble dimension. Notably, it is the case
        # that broadcasting is utilized to ensure that a configuration
        # of any shape, but ending in ensemble_width shape, is compatible.

        #We flatten and run the matmul as a multiplication between
        #two matrices. This is because torch.sparse.mn does not like
        #tensors

        restoration_shape = torch.Size(list(configuration.shape[:-1]) + list(attribute.shape[1:]))
        attribute = attribute.flatten(1)
        configuration = configuration.flatten(0, -2)
        configuration = configuration.masked_fill(configuration < self.sparse_epsilon, 0.0)
        configuration = configuration.to_sparse_coo()
        output = torch.sparse.mm(configuration, attribute)
        output = output.view(restoration_shape)
        return output



    def __getattribute__(self, item):
        """
        Gets the given attribute.
        If the attribute is registered as a ensemble, applies the current configuration
        Else, does nothing."""
        if item in ("_registry", "configuration", "construct_ensemble", "__dict__"):
            return super().__getattribute__(item)
        if hasattr(self, "_registry") and item in self._registry:
            return self.construct_ensemble(item)
        else:
            return super().__getattribute__(item)

    def __setattr__(self, key, value):
        """Sets the given attribute. Redirects registered items into registry"""
        if not hasattr(self, "_registry"):
            return super().__setattr__(key, value)
        if key in self._registry:
            self._registry[key] = value
        super().__setattr__(key, value)

    def __init__(self,
                 ensemble_width: int,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 configuration: Optional[torch.Tensor] = None,
                 sparse_epsilon=0.0001):
        """

        :param ensemble_width: The width that the ensemble kernels will be. Must be provided
        :param top_k: Starts top-k mode. This will, for each dimension in configuration,
                      construct an output matrix out of only the top-k most probable config values.
                        Exclusive with regards to top-p mode

                        This engages sparse logic.
        :param top_p: Starts top-p mode. This will, for each dimension in config, only bother to
                        evaluate with the top-p probable kernels.

                        This engages sparse logic.
        :param configuration: Optional. A configuration.
        """

        assert not (top_k is not None and top_p is not None)
        self.top_k = top_k
        self.top_p = top_p
        self.sparse_epsilon = sparse_epsilon
        self.native_ensemble_width = ensemble_width
        self._registry: Dict[str, torch.Tensor] = {}

        super().__init__()
        #Some sort of configuration is required for torchscript to be happy.
        #
        #Fortunately, we cna replace it with something of a different shape
        #later.

        if configuration is None:
            configuration = torch.softmax(torch.ones([1, ensemble_width]), dim=-1)
        self.configuration = configuration

class Linear(EnsembleSpace):
    """
    A linear layer allowing a number of tricks to be deployed in parallel.
    These tricks are:

    * Linear mapping
    * Autoreshaping
    * Parallel execution
    * Dynamic Ensemble Assembly

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

    This has its greatest utility when designing independent ensembles. However,
    exchange of information has it's purpose. Which brings us to...

    ---- Dynamic Ensemble Assembly ----

    Although it is possible to create completely parallel tensor kernels, it is possible
    to do much more. Using the parallel attribute indicated above actually activates the
    dynamic ensemble mechanism built into the class. This mechanism allows correctly
    defined specifications to create a EnsembleSpace of different Linear kernels, which
    the program may then combine together for various purposes based on a Configuration
    attribute.

    consider a tensor of shape tensor(3, 5, 10). If we were to define instance=Linear(10, 5, dynamic=20),
    it is the case that a randomly configured Linear layer with an ensemble section of width 20
    would be created. Lets say for a moment we want to configure linear so that each ensemble is
    drawn from equally. Then

    instance.configuration = torch.ones([20])

    Would do the trick, by weighting each option equally. It is also possible to specify the configuration
    more directly. Any extra dimensions will pop out right before the batch info starts. See
    EnsembleSpace for more details.
    """
    def standardize_input(self, input: Union[torch.Tensor, List[int], int])->torch.Tensor:
        """
        Convert an input in one of three formats into a common single format of tensor
        Sanitize and throw errors if there is a problem
        """
        if not isinstance(input, (torch.Tensor, list, int)):
            raise ValueError("Illegal constructor argument")
        if isinstance(input, int):
            input = [input]
        output = torch.tensor(input, dtype=torch.int64)
        return output

    def __init__(self,
                 input_shape: Union[torch.Tensor, List[int], int],
                 output_shape: Union[torch.Tensor, List[int], int],
                 parallel: Optional[Union[torch.Tensor, List[int], int]] = None,
                 dynamics: Optional[Union[torch.Tensor, List[int], int]] = None,
                 use_bias: bool = True,
                 top_k: Optional[int] = None,
                 top_p: Optional[int] = None

                 ):
        """

        :param input_shape: The shape of the input
        :param output_shape: The shape of the output
        :param parallel: What parallel portions of the kernel to create
        :param dynamics: What dynamics to create, or what config to make
        :param use_bias: Whether or not to use bias on the linear layer
        :param top_k: The top k to use. Only active if dynamics is active
        :param top_p: The top p to use. Only active if dynamics is active.
        """

        #Peform standardization
        if parallel is None:
            parallel = []
        if dynamics is None:
            dynamics = []

        input_shape = self.standardize_input(input_shape)
        output_shape = self.standardize_input(output_shape)
        parallel = self.standardize_input(parallel)
        dynamics = self.standardize_input(dynamics)

        #Begin developing kernel shapes and conversions

        matrix_rows = input_shape.prod().unsqueeze(-1)
        matrix_columns = output_shape.prod().unsqueeze(-1)
        matrix_shape = torch.concat([matrix_rows, matrix_columns], dim=0)

        if use_bias:
            bias_shape = matrix_columns

        input_autoshape_mapping = (input_shape, matrix_rows)
        output_autoshape_mapping = (matrix_columns, output_shape)

        #Introduce modifications to account for parallelization.
        #This consists of additional indepedent dimensions
        #at the front of the matrix and bias

        matrix_shape = torch.concat([parallel, matrix_shape], dim=0)
        if use_bias:
            bias_shape = torch.concat([parallel, bias_shape])

        #Handle dynamics.
        #
        #Make sure to set the release dimension
        #flag if dynamics are not being utilized

        if dynamics.dim() > 0:
            ensemble_width = dynamics.shape[-1]
            self.is_ensemble = True
        else:
            ensemble_width = 1

            self.is_ensemble = False
        super().__init__(ensemble_width, top_k=top_k, top_p=top_p)
        matrix_shape = torch.concat([torch.tensor([ensemble_width]), matrix_shape])
        if use_bias:
            bias_shape = torch.concat([torch.tensor([ensemble_width]), bias_shape])

        #Generate actual kernels

        matrix_kernel = torch.empty(matrix_shape.tolist())
        torch.nn.init.kaiming_uniform_(matrix_kernel, math.sqrt(5))
        matrix_kernel = nn.Parameter(matrix_kernel)

        if use_bias:
            bias_kernel = torch.zeros(bias_shape.tolist())
            bias_kernel = nn.Parameter(bias_kernel)

        #Register kernels and deployment details

        self.use_bias = use_bias

        self.input_map_reference = input_autoshape_mapping
        self.output_map_reference = output_autoshape_mapping

        self.matrix_kernel = matrix_kernel
        self.register_ensemble("matrix_kernel")

        if use_bias:
            self.bias_kernel = bias_kernel
            self.register_ensemble("bias_kernel")

    def forward(self, tensor: torch.Tensor):

        input_shape, row_length = self.input_map_reference
        column_length, output_shape = self.output_map_reference
        assert torch.Size(input_shape) == tensor.shape[-len(input_shape):], "Tensor and kernel shapes not compatible"

        flattened_input = Glimpses.reshape(tensor,input_shape, row_length)
        if self.use_bias:
            flattened_output = torch.matmul(self.matrix_kernel, flattened_input) + self.bias_kernel
        else:
            flattened_output = torch.matmul(self.matrix_kernel, flattened_input)
        restored_output = Glimpses.reshape(flattened_output, column_length, output_shape)

        return restored_output

