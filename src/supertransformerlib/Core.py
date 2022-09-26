"""

The module for the ensemble
extended linear process.

"""
import math
from typing import Union, List, Optional, Dict, overload

import torch
import torch.nn
from torch import nn


from . import Glimpses

@torch.jit.script
class Config():
    """
    A class for representating a configuration
    an ensemble may be placed into.

    This is initialized with certain parameters,
    does sanity checking and processing, and may
    subsequently be passed or assigned to compatible
    layers

    --- configuration ---

    The configuration may be viewed and set using the configuration
    property. A configuration should be set as a raw bundle of
    logits. Don't worry, it is differentable. It is converted into
    a probability, possibly revised using top-p or top-k analysis,
    and then stored away for use assembling kernels.

    Notably, the configuration must be designed such that the last
    dimension is of width ensemble width. One has complete freedom
    regarding additional dimensions, which will broadcast and
    then pushed onto the front dimensions of the returned kernel.

    The last dimension of the configuration is treated as if
    it contains weights by which to use to assemble the final
    kernel. Notably, it is the case that there exists both
    a logits and a probability mode, and the configuration
    can be set using .set_config

    As an example, consider a registered kernel defined as a parameter of shape
    Parameter(10, 3, 5,20) with name "test_kernel". This has, naturally, an ensemble width of
    10. If you define the follow configurations with the following shapes :


    configuration1 = tensor(1, 10)
    configuration2 = tensor(13, 10)
    configuration3 = tensor(3, 5, 12, 10)

    The kernel shapes you will get if you perform self.test_kernel would be

    result1 = tensor(1, 3, 5, 12, 10)
    result2 = tensor(13, 3, 5, 12, 10)
    result3 = tensor(3, 5, 12, 3,5, 12, 10)

    Notice that only the last dimension is reduced, and the configuration is
    broadcast across the rest of the tensor. This allows, for example, batch
    dependent setup and processing.


    """


    def rebalance_probability(self, tensor: torch.Tensor)->torch.Tensor:
        """Takes a tensor which consists of positive weights and turns it into something that sums up to one"""
        return tensor / tensor.sum(dim=-1).unsqueeze(-1)

    def _process_config_options(self,
                                configuration: torch.Tensor,
                                top_k: Optional[int],
                                top_p: Optional[float])->torch.Tensor:
        """
        A support function. It handles probabity conversion for
        top_p and top_k.
        """

        if top_k is not None and top_p is not None:
            raise ValueError("Top_k and top_p cannot both be active at the same time. Please leave one as None")


        if top_k is not None:
            # Create the top k sparse configuration
            #
            # Do this by finding the top k indices, making
            # a mask, then masking the configuration and performing
            # the probabilistic change. Then we rebalance the probabilites

            _, indices = torch.topk(configuration,top_k, dim=-1)
            indices = indices.unsqueeze(-1)
            mask = torch.arange(self.ensemble_width)
            mask = indices == mask
            mask = torch.any(mask, dim=-2)
            configuration = configuration * mask
            configuration = self.rebalance_probability(configuration)

        elif top_p is not None:
            ### Develop the top-p driven case
            #
            # This will consist of finding by cumulative sums the
            # sorted layer at which we transition above the top
            # p, then generating and applying a mask based on these
            # indices.

            # Extract the relevant indices in an efficient manner

            configuration = torch.softmax(configuration, dim=-1)
            sorted, indices = torch.sort(configuration, descending=True, dim=-1)

            # Promote interesting dimension to front for zippiing
            sorted = sorted.unsqueeze(0).transpose(0, -1).squeeze(-1)
            indices = indices.unsqueeze(0).transpose(0, -1).squeeze(-1)

            # Setup and extract indices
            cumulative_probability = torch.zeros(configuration.shape[:-1])
            mask_construction_indices: List[torch.Tensor] = []
            off = torch.full(cumulative_probability.shape, -1)
            for probabilities, index_layer in zip(sorted, indices):
                cumulative_probability_lower_then_p = cumulative_probability < top_p
                mask_update = torch.where(cumulative_probability_lower_then_p, index_layer, off)
                mask_construction_indices.append(mask_update)
                cumulative_probability += probabilities
                if torch.all(cumulative_probability >= top_p):
                    break

            mask_indices = torch.stack(mask_construction_indices, dim=-1).unsqueeze(-1)
            mask = mask_indices == torch.arange(self.ensemble_width)
            mask = torch.any(mask, dim=-2)
            configuration = configuration * mask
            configuration = self.rebalance_probability(configuration)
        return configuration

    def sanitize_config(self, config: torch.Tensor, logits: bool, dtype: torch.dtype):
        """Sanity checks the configuration """
        if not isinstance(config, torch.Tensor):
            raise ValueError("Configuration must be a tensor")
        if not isinstance(logits, bool):
            raise ValueError("logits must be a bool")
        if config.dim() < 2:
            raise ValueError("Configuration must have two or more dimesions")
        elif config.shape[-1] != self.ensemble_width:
            raise ValueError("Configuration last dimension bad. Expected %s, got %s"
                             % (self.ensemble_width, config.shape[-1]))
        config = config.to(dtype=dtype)
        if logits:
            config = torch.softmax(config, dim=-1)

        return config
    def __init__(self,
                 config: torch.Tensor,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 logits: bool = True,
                 dtype: torch.dtype = torch.float32,
                 tags: Optional[List[str]] = None,
                 ):
        """

        :param config: The tensor, in logit format or otherwise, to turn into the config. Last dim is ensemble dim
        :param top_k: Whether to use top-k, and how many to keep
        :param top_p: Whether to use top-p, and how many to keep
        :param logits: Whether this is a logit or probability
        :param dtype: The dtype to make the config
        :param tags: Any tags to include. If none, this is a generic config. If tags are included, only
            configs with a matching tag will be updated. when passed into a kernelspace.
        """

        self.ensemble_width = config.shape[-1]
        config = self.sanitize_config(config, logits, dtype)
        config = self._process_config_options(config, top_k, top_p)
        self.config = config
        self.dtype = dtype
        self.tags = tags

    def __call__(self):
        return self.config


class Kernel(nn.Module):
    """
    A small storage layer to a torch kernel of some
    sort. The kernel may be an ensemble, in which case
    a configuration must eventually be provided
    to construct anything, or just a regular old
    parameter.

    The class will be captured when seen and provided
    with configurations when updates go by.

    By default, it will be setup as an ensemble that spits out the
    same kernel it is given. It can then be further modified using update_config.
    """
    @property
    def ensemble_width(self):
        return self.kernel.shape[0]

    @torch.jit.export
    def update_config(self, config: Config):
        """
        Update the given kernel with the given config.
        """

        #Temporary variables needed for torchscript to work right.
        my_tags = self.tags
        config_tags = config.tags
        if torch.jit.isinstance(my_tags, List[str]) and torch.jit.isinstance(config_tags, List[str]):
            #Finding if the two tag groups share an element is a great
            #canidate for set. Unforunately, torchscript does not support it.
            #so we are using a loop for now. Do not use too many tags I guess?


            is_contained = False
            for tag in my_tags:
                if tag in config_tags:
                    is_contained = True
                    break
            if not is_contained:
                return None

        if not self.is_ensemble:
            raise ValueError("Attempt to set config on non ensemble kernel")
        if config.ensemble_width != self.ensemble_width:
            raise ValueError("Attempt to set config when config ensemble width does not match kernel width")
        self.config = config

    def __init__(self,
                 kernel: torch.Tensor,
                 is_ensemble: bool = True,
                 tags: Optional[List[str]] = None,
                 sparsity_exclusion: float = 0.001,
                 dtype: torch.dtype = torch.float32):
        """
        :param kernel: The kernel to store away
        :param is_ensemble: Whether or not this is an ensemble, or a standard kernel
        :param dtype: the dtype to cast to.
        :param sparsity_exclusion:
            A parameter that controls how small a config entry should be before it is evaluated as sparse.
        :param tags: Used to control when a config is applied. None means any passing config is applied.
            If a list of strings is provided, only configs with one or more of the matching tags will be applied.

        """

        super().__init__()
        if isinstance(kernel, nn.Parameter):
            self.register_parameter("kernel", kernel)
        else:
            self.register_buffer("kernel", kernel)

        self.tags = tags
        self.is_ensemble = is_ensemble
        self.sparisty_exlusion = sparsity_exclusion
        self.config = Config(torch.eye(self.ensemble_width, dtype=dtype))

    def forward(self):
        kernel = self.kernel
        kernel = kernel.to(self.config.dtype)
        if self.is_ensemble:
            configuration = self.config.config
            restoration_shape = torch.Size(list(configuration.shape[:-1]) + list(kernel.shape[1:]))
            kernel = kernel.flatten(1)
            configuration = configuration.flatten(0, -2)
            configuration = configuration.masked_fill(configuration < self.sparisty_exlusion, 0.0)
            configuration = configuration.to_sparse()
            output = torch.sparse.mm(configuration, kernel)
            output = output.view(restoration_shape)
        else:
            output = kernel
        return output



class _KernelSpace(nn.Module):
    """
    This exists only to get
    around certain torchscript limitations.
    """

class KernelSpace(_KernelSpace):
    """
    Being able to dynamically combine together kernels
    into ensembles is the point of this class. It is
    the case that one can use this, and the accompanying
    kernel type, to easily and elegantly create an
    environment in which configurations can act with
    minimal annoyance to the programmer.

    After setup, kernels are caught as they
    are assigned and registered against an internal registry.
    They can then be retrieved using get_kernel, and the kernel
    name.

    While an ideal world would see the kernel retrievable the
    same way it is assigned, torchscript has no idea about
    __getattr__ making this impossible.


    ---- Examples ----

    Lets go ahead and look into an example. Suppose we want
    to make a set of two layers. The first will be a linear
    layer and creates a configuration. The second adds a kernel
    constructed from that configuration to the tensor passing through,
    with it being the case each batch may have a different configuration.
    What would that look like? Lets go see

    ```
    import torch
    from torch import nn
    from src.supertransformerlib import Core


    # Setup
    ensemble_width = 20
    d_model = 32
    batch_size = 16

    mockup_data_tensor = torch.randn([batch_size, d_model])

    # Define the layers


    class AddBias(Core.KernelSpace):
        def __init__(self, ensemble_width, d_model):
            super().__init__(ensemble_width)
            bias_kernel = torch.zeros([ensemble_width, d_model])
            bias_kernel = nn.Parameter(bias_kernel)
            self.register_ensemble("bias_kernel", bias_kernel)
        def forward(self, tensor):
            # Note that the raw kernel would be 20 x 32, not the required 16x32
            # It MUST be rebuilding the kernel using the ensemble if this works.
            kernel = self.get_kernel("bias_kernel")
            return tensor + kernel

    class ConfigureThenAdd(nn.Module):
        def __init__(self):
            super().__init__()
            self.configuration_generator = nn.Linear(d_model, ensemble_width)
            self.add_layer = AddBias(ensemble_width, d_model)
        def forward(self, tensor):
            configuration = self.configuration_generator(tensor)
            self.add_layer.configuration = configuration
            return self.add_layer(tensor)

    instance = ConfigureThenAdd()
    instance = torch.jit.script(instance) #Notice torchscript works. Yay!
    instance(mockup_data_tensor)

    ```
    """
    # The actual responsibilities of this particular layer are actually
    # extremely limited. Ultimately, all that it is responsible for doing
    # is tracking down further sublayers to update when given particular
    # parameters

    @torch.jit.export
    def update_children(self, config: Config):
        """Updates the config for all immediate children"""
        #This does a recursive search for kernels
        #and sub kernelspaces. The kernelspaces see
        #their update_config method called, causing recursion. The
        #kernels will see their update config method called, updating
        #the actual config
        #
        #Config is stored entirely on the kernels. This layer just
        #tracks down where it should go.
        for child in self.children():
            if isinstance(child, Kernel):
                if child.is_ensemble:
                    child.update_config(config)

    @torch.jit.export
    def update_descendents(self, config: Config):
        """Updates my own children, along with all descendent kernelspace layers as well. """
        for child in self.children():
            if isinstance(child, Kernel):
                child.update_config(config)
            if hasattr(child, "_is_KernelSpace"): #We cannot use isinstance(child, KernelSpace) due to torchscript limitations.
                child.update_descendents(config)



    def __init__(self):
        super().__init__()
        self._is_KernelSpace = True

class Utility:
    """ A place for utility methods to belong"""
    def get_group(self, ensemble_width, ):
        """"""


    def standardize_input(self, input: Union[torch.Tensor, List[int], int])->torch.Tensor:
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

class Linear(Utility, KernelSpace):
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

    This has its greatest utility when designing independent ensembles. However,
    exchange of information has it's purpose. Which brings us to...

    ---- Dynamic Kernel Assembly ----

    Although it is possible to create completely parallel tensor kernels, it is possible
    to do much more. Using the parallel attribute indicated above actually activates the
    dynamic ensemble mechanism built into the class. This mechanism allows correctly
    defined specifications to create a KernelSpace of different Linear kernels, which
    the program may then combine together for various purposes based on a Configuration
    attribute.

    consider a tensor of shape tensor(3, 5, 10). If we were to define instance=Linear(10, 5, dynamic=20),
    it is the case that a randomly configured Linear layer with an ensemble section of width 20
    would be created. Lets say for a moment we want to configure linear so that each ensemble is
    drawn from equally. Then

    instance.configuration = torch.ones([20])

    Would do the trick, by weighting each option equally. It is also possible to specify the configuration
    more directly. Any extra dimensions will pop out right before the batch info starts. See
    KernelSpace for more details.

    ---- Combination ----

    when both parallization and dynamic configuration is present, the order of resolution across dimensions,
    from right to left, is first the autoshaping specifications, then the parallelization, and finally the
    dynamic specifications.

    --- configuration ---

    set_config will configure everything properly

    """

    def __init__(self,
                 input_shape: Union[torch.Tensor, List[int], int],
                 output_shape: Union[torch.Tensor, List[int], int],
                 parallel: Optional[Union[torch.Tensor, List[int], int]] = None,
                 dynamics: Optional[int] = None,
                 use_bias: bool = True,
                 tags: Optional[List[str]] = None,
                 ):
        """

        :param input_shape: The shape of the input
        :param output_shape: The shape of the output
        :param parallel: What parallel portions of the kernel to create
        :param dynamics: If defined, what the dynamics width is.
        :param use_bias: Whether or not to use bias on the linear layer
        :param tags: Any tags to restrict config updates to. Passed on to our kernels.
        """

        #Peform standardization
        if parallel is None:
            parallel = []
        if dynamics is None:
            dynamics = 0

        input_shape = self.standardize_input(input_shape)
        output_shape = self.standardize_input(output_shape)
        parallel = self.standardize_input(parallel)

        #Begin developing kernel shapes and conversions

        matrix_rows = input_shape.prod().unsqueeze(-1)
        matrix_columns = output_shape.prod().unsqueeze(-1)
        matrix_shape = torch.concat([matrix_columns, matrix_rows], dim=0)

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


        if dynamics > 0:
            ensemble_width = dynamics
            matrix_shape = torch.concat([torch.tensor([ensemble_width]), matrix_shape])
            if use_bias:
                bias_shape = torch.concat([torch.tensor([ensemble_width]), bias_shape])
            is_ensembled = True
        else:
            is_ensembled = False


        super().__init__()

        #Generate actual kernels

        matrix_kernel = torch.empty(matrix_shape.tolist())
        torch.nn.init.kaiming_uniform_(matrix_kernel, math.sqrt(5))
        matrix_kernel = nn.Parameter(matrix_kernel)
        matrix_kernel = Kernel(matrix_kernel, is_ensembled)

        if use_bias:
            bias_kernel = torch.zeros(bias_shape.tolist())
            bias_kernel = nn.Parameter(bias_kernel)
            bias_kernel = Kernel(bias_kernel, is_ensembled)

        #Register kernels and deployment details

        self.use_bias = use_bias
        self.is_ensembled = is_ensembled

        self.input_map_reference = input_autoshape_mapping
        self.output_map_reference = output_autoshape_mapping

        self.matrix_kernel = matrix_kernel
        if use_bias:
            self.bias_kernel = bias_kernel

    def forward(self, tensor: torch.Tensor):

        input_shape, row_length = self.input_map_reference
        column_length, output_shape = self.output_map_reference

        flattened_input = Glimpses.reshape(tensor,input_shape, row_length)
        flattened_input = flattened_input.unsqueeze(-1)

        if self.use_bias:
            flattened_output = torch.matmul(self.matrix_kernel(), flattened_input).squeeze(-1)
            flattened_output = flattened_output + self.bias_kernel()
        else:
            flattened_output = torch.matmul(self.matrix_kernel(), flattened_input)
        restored_output = Glimpses.reshape(flattened_output, column_length, output_shape)
        return restored_output

