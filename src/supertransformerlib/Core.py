"""

The module for the ensemble
extended linear process.

"""
import math
from typing import Union, List, Optional, Dict, overload

import torch
from torch import nn

from . import Glimpses


class EnsembleSpace(nn.Module):
    """
    Purpose:

    Dynamic assembly of the pieces of an ensemble into
    the optimum option is the idea behind this class.

    One may setup during init the expected ensemble
    width along with assembly features such as
    whether to use top-p or top-k. Then, at some
    point along the line one can provide a configuration,
    which is discussed more below.

    Once all ensembles are registered, they may be used
    transparently in normal logic in torch. It is the case
    that when one goes to access something registered as
    an ensemble, you get given the superimposed version of it.

    How the class actually works is that the configuration determines
    which of the ensemble slices are most useful for the given problem.
    These slices are then weighted by the configuration and added together.


    --- init ---

    Of importance for this dicussion, during the init
    step one must set the ensemble_width. This sets the
    instance expectations regarding the size of ensemble
    kernels and the shape of the configuration.

    One may at this point go with the default configuration,
    or alternatively setup a top-k or top-p parameter situation.

    Notably, an ensemble_length of 0 disables the forward logic.

    --- register_ensemble ---

    This is used by the subclass to handle squirreling away ensembles.
    See it for details

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

    ---- Tagging  ----

    During the __init__ process, it is possible to define a list of
    strings to associate with the layer. These strings are known as
    tags, and can be used for sorting purposes or even for grouping
    bunches of layers together.

    ---- Grouping ----

    It is possible to place multiple EnsembleSpace modules together into
    a so called "group". A group is simply a group of EnsembleSpace layers
    that are tied together in a so called "EnsembleGroup" object, and which
    promise to have the same configuration and related properties in common
    at the same time. Additionally, setting the associated property on the
    group will update the properties on all members of a group.

    A layer cannot belong to more than one group at a time.

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


    class AddBias(Core.EnsembleSpace):
        def __init__(self, ensemble_width, d_model):
            super().__init__(ensemble_width)
            bias_kernel = torch.zeros([ensemble_width, d_model])
            bias_kernel = nn.Parameter(bias_kernel)
            self.bias_kernel = bias_kernel
            self.register_ensemble("bias_kernel")
        def forward(self, tensor):
            # Note that the raw kernel would be 20 x 32, not the required 16x32
            # It MUST be rebuilding the kernel using the ensemble if this works.
            kernel = self.bias_kernel
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

    @property
    def ensemble_enabled(self)->bool:
        return self.native_ensemble_width > 0


    ### Set and get configuration ###

    @staticmethod
    def rebalance_probability(tensor: torch.Tensor)->torch.Tensor:
        """Takes a tensor which consists of positive weights and turns it into something that sums up to one"""
        return tensor / tensor.sum(dim=-1).unsqueeze(-1)

    def _process_config_options(self, configuration: torch.Tensor)->torch.Tensor:
        """
        A support function for set configuration. Converts a raw
        logit into a probability.

        :param config: The provided configuration. Expected to
        have been sanity checked by this point
        :return: The probabilistic configuration. In the right dtype as
        """


        if self._top_k is not None:
            # Create the top k sparse configuration
            #
            # Do this by finding the top k indices, making
            # a mask, then masking the configuration and performing
            # the probabilistic change. Then we rebalance the probabilites
            _, indices = torch.topk(configuration, self._top_k, dim=-1)
            indices = indices.unsqueeze(-1)
            mask = torch.arange(self.native_ensemble_width)
            mask = indices == mask
            mask = torch.any(mask, dim=-2)
            configuration = configuration * mask
            configuration = self.rebalance_probability(configuration)

        elif self._top_p is not None:
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
                cumulative_probability_lower_then_p = cumulative_probability < self._top_p
                mask_update = torch.where(cumulative_probability_lower_then_p, index_layer, off)
                mask_construction_indices.append(mask_update)
                cumulative_probability += probabilities
                if torch.all(cumulative_probability >= self._top_p):
                    break

            mask_indices = torch.stack(mask_construction_indices, dim=-1).unsqueeze(-1)
            mask = mask_indices == torch.arange(self.native_ensemble_width)
            mask = torch.any(mask, dim=-2)
            configuration = configuration * mask
            configuration = self.rebalance_probability(configuration)

        return configuration

    @torch.jit.export
    def get_config(self)->torch.Tensor:
        """
        The get configuration function returns the current configuration.
        """
        # Due to torchscript not handling properties on modules
        # we have to use an explicit getter and setter. Unfortunately.
        return self._configuration

    @torch.jit.export
    def set_config(self, config: torch.Tensor, logits: bool = False):
        """
        Set a logit as a configuration, with no fancy work
        :param config: The configuration logit
        :param logits: whether or not the input is a logit
        """

        # Do a few sanity checks.
        if not isinstance(config, torch.Tensor):
            raise ValueError("Configuration must be a tensor")
        if config.dim() < 2:
            raise ValueError("Configuration must have two or more dimesions")
        elif config.shape[-1] != self.native_ensemble_width:
            raise ValueError("Configuration last dimension bad. Expected %s, got %s"
                             % (self.native_ensemble_width, config.shape[-1]))

        # Cast and process the config into a probability.
        #
        # Handle probability options such as top-p or top-k
        config = config.to(dtype=self.dtype)
        if logits:
            config = torch.softmax(config, dim=-1)
        config = self._process_config_options(config)
        self._configuration = config

    #Set top p, set top k

    def set_top_p(self, top_p: Optional[float]):
        if top_p is not None:
            if not 0 <= top_p <= 1.0:
                raise ValueError("Top_p is not between 0 and 1")
            if self._top_k is not None:
                self._top_k = None
        self._top_p = top_p

    def get_top_p(self)-> Optional[float]:
        return self._top_p

    def set_top_k(self, top_k: Optional[int]):
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("Top k not greater than or equal to one")
            if self._top_p is not None:
                self._top_p = None
            self._top_k = top_k
        else:
            self._top_k = None
    def get_top_k(self)->Optional[int]:
        return self._top_k

    #### Grouping logic ###
    #
    # Grouping consists of tying
    # together sets




    #Registration logic.

    def register_ensemble(self, name: str, attribute: Optional[Union[nn.Parameter, torch.Tensor]] = None):
        """

        Registers an attribute as an ensemble kernel tensor.
        Will automatically convert tensor to the datatype defined at init.

        -- specifications --
        The attribute should be a tensor or a parameter.
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
        if attribute.dim() < 2:
            raise AttributeError("Ensemble attribute of name %s must have an ensemble dimension" % name)
        if attribute.shape[0] != self.native_ensemble_width:
            raise AttributeError("Ensemble attribute of name %s had first dim of length %s, expecting %s" % (
                name,
                attribute.shape[0],
                self.native_ensemble_width
            ))
        if name in self.__dict__:
            del self.__dict__[name]

        attribute = attribute.to(self.dtype)
        if isinstance(attribute, nn.Parameter):
            self.register_parameter(name, attribute)
            self._registry[name] = "parameter"
        else:
            self.register_buffer(name, attribute)
            self._registry[name] = "buffer"

    def construct_ensemble(self, kernel: Union[torch.Tensor, nn.Parameter])->torch.Tensor:
        """
        Construct a kernel in which all but the last
        configuration entry is broadcast onto the end
        of the named tensor.

        Perform matrix multiplication to collapse the last
        tensor.

        :param name: The attribute to construct
        :return: The constructed attribute
        """
        #attribute: (ensemble_width, ..., dim1, dim0)
        configuration = self.get_config() #(..., ensemble_width)
        # In the case of a ensemble_width of 0, the ensemble is disabled
        # and should just return the kernel directly. This supports
        # simpler code using register ensemble

        if self.native_ensemble_width == 0:
            return kernel

        # This functions by performing a matrix multiplication
        # across the ensemble dimension. Notably, it is the case
        # that broadcasting is utilized to ensure that a configuration
        # of any shape, but ending in ensemble_width shape, is compatible.

        # It is the case that with top-p and top-k calculations, many
        # of the configuration entries will be empty. Additionally, it is
        # highly likely if the configuration is generated by the program itself
        # that some entries will be very small. As a result, we run the matrix
        # multiplication in sparse format.

        # Sparse operations in torch are highly restricted. In particular, it is
        # the case that backpropogation is only supported using torch.sparse.mm
        # and a few similar operations. These operations only work when using
        # matrices, not tensors.

        # As a result, we flatten everything into matrices, mask,
        # perform a sparse matrix multiplication, then restore
        # the tensor dimensions.

        restoration_shape = torch.Size(list(configuration.shape[:-1]) + list(kernel.shape[1:]))
        kernel = kernel.flatten(1)
        configuration = configuration.flatten(0, -2)
        configuration = configuration.masked_fill(configuration < self.sparse_epsilon, 0.0)
        configuration = configuration.to_sparse_coo()
        output = torch.sparse.mm(configuration, kernel)
        output = output.view(restoration_shape)
        return output
    def __getattr__(self, item):
        """
        Runs the kernel rebuilt if it is found within
        the registry. Else, hands off to torch
        """
        outcome = super().__getattr__(item)
        if hasattr(self, "_registry") and item in self._registry:
            return self.construct_ensemble(outcome)
        return super().__getattr__(item)

    def __getattribute__(self, item):

            # Torchscript functions in such a way that
            # the getattr function does not work.
            #
            # As a result, we must find alternatives.
            #
            # The following code explictly catches and
            # then calls __getattr__, allowing attribute lookup
            # to occur in a torchscript context by using
            # self.__getattribute__(name), in leu of getattr(self, "name")
            #
            # DO NOT OPTIMIZE AWAY
            try:
                return super().__getattribute__(item)
            except AttributeError:
                return self.__getattr__(item)

    def __init__(self,
                 ensemble_width: int,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 configuration: Optional[torch.Tensor] = None,
                 tags: Optional[List[str]] = None,
                 sparse_epsilon=0.0001,
                 dtype: torch.dtype = torch.float32,
                 ):
        """
        :param ensemble_width: The width that the ensemble kernels will be.
            * Must be provided
            * Length of zero disables ensemble and just returns the kernel directly.
        :param top_k: Starts top-k mode. This will, for each dimension in configuration,
                      construct an output matrix out of only the top-k most probable config values.
                        Exclusive with regards to top-p mode

                        This engages sparse logic.
        :param top_p: Starts top-p mode. This will, for each dimension in config, only bother to
                        evaluate with the top-p probable kernels.

                        This engages sparse logic.
        :param configuration: Optional. A passed configuration
                        Must follow the configuration rules outlined above
        :param sparse_epsilon:
                    The configuration probability below which to exclude the
                    configuration as an option during ensemble construction.

                    0.0 disables
        :param tags: Used with groups to narrow down what is captured and what is not.
        :param dtype: What to cast the configuration and kernels to.
        """

        if top_k is not None and top_p is not None:
            raise ValueError( "Cannot use both top-p and top-k at same time")

        self._top_p = None
        self._top_k = None

        self.set_top_p(top_p)
        self.set_top_k(top_k)
        self.sparse_epsilon = sparse_epsilon
        self.native_ensemble_width = ensemble_width
        self._registry: Dict[str, str] = {}
        self.dtype = dtype
        self.debug_bit = 0
        super().__init__()

        #Some sort of configuration is required for torchscript to be happy.
        #
        #Fortunately, we cna replace it with something of a different shape
        #later. We initialize it to a basic configuration compatible with the
        #problem.

        if tags is None:
            tags = []
        self.tags = tags

        if configuration is None:
            configuration = torch.softmax(torch.ones([1, ensemble_width], dtype=dtype), dim=-1)
        self.set_config(configuration)


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


class Linear(Utility, EnsembleSpace):
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

    ---- Combination ----

    when both parallization and dynamic configuration is present, the order of resolution across dimensions,
    from right to left, is first the autoshaping specifications, then the parallelization, and finally the
    dynamic specifications.


    """

    def __init__(self,
                 input_shape: Union[torch.Tensor, List[int], int],
                 output_shape: Union[torch.Tensor, List[int], int],
                 parallel: Optional[Union[torch.Tensor, List[int], int]] = None,
                 dynamics: Optional[int] = None,
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
        #
        #Make sure to set the release dimension
        #flag if dynamics are not being utilized.
        #
        #This flag is called "is_ensemble", and has
        #an effect on the forward method.

        if dynamics > 0:
            ensemble_width = dynamics
            matrix_shape = torch.concat([torch.tensor([ensemble_width]), matrix_shape])
            if use_bias:
                bias_shape = torch.concat([torch.tensor([ensemble_width]), bias_shape])
        else:
            ensemble_width = 0

        super().__init__(ensemble_width, top_k=top_k, top_p=top_p)

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
        if self.ensemble_enabled:
            self.register_ensemble("matrix_kernel")

        if use_bias:
            self.bias_kernel = bias_kernel
            if self.ensemble_enabled:
                self.register_ensemble("bias_kernel")

    def forward(self, tensor: torch.Tensor):

        input_shape, row_length = self.input_map_reference
        column_length, output_shape = self.output_map_reference
        input_shape_as_list: List[int] = input_shape.tolist() #Required for torchscript assert
        test_assertion = torch.Size(input_shape_as_list) == tensor.shape[-len(input_shape):]

        assert test_assertion, "Tensor and kernel shapes not compatible"

        flattened_input = Glimpses.reshape(tensor,input_shape, row_length)
        flattened_input = flattened_input.unsqueeze(-1)
        if self.use_bias:
            flattened_output = torch.matmul(self.matrix_kernel, flattened_input).squeeze(-1)
            flattened_output = flattened_output + self.bias_kernel
        else:
            flattened_output = torch.matmul(self.matrix_kernel, flattened_input)
        restored_output = Glimpses.reshape(flattened_output, column_length, output_shape)
        return restored_output

