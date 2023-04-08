
import torch
from typing import Optional, Dict, List, Tuple, Union
from supertransformerlib import Core


class BundleTensorError(Exception):
    """ The general bundle error class"""
    pass

class ConstructorError(BundleTensorError):
    """ Triggered when an error occurs during constructor validation"""
    pass

class ArithmeticError(BundleTensorError):
    """ Triggered when an error is caught during arithmetic action"""
    pass

class ConstructorNoTensorsError(ConstructorError):
    def __str__(self):
        msg = """\
        An issue has occurred when creating a bundle tensor. The 
        bundle tensor was expected to be initialized with a collection
        of tensors. However, no tensors were actually provided in the
        dictionary.
        """
        msg = Core.dedent(msg)
        return msg


class ConstructorBadRankError(ConstructorError):
    def __init__(self, violating_name: str, expected_rank: int, got_rank: int):
        self.violating_name = violating_name
        self.expected_rank = expected_rank
        self.got_rank = got_rank

    def __str__(self):
        msg = f"""\
        An issue has occurred when creating a state tensor. The rank
        for tensor {self.violating_name} is bad. The expected rank, to
        match the batch dims, was {self.expected_rank}. However, it 
        was found instead to have rank {self.got_rank}.

        Since {self.got_rank} is less than {self.expected_rank}, construction failed.
        """
        msg = Core.dedent(msg)
        return msg

class ConstructorBadTensorDtypeError(ConstructorError):
    def __init__(self,
                 violating_name: str,
                 violating_type: torch.dtype,
                 base_name: str,
                 base_type: torch.dtype):
        self.violating_name = violating_name
        self.violating_type = violating_type
        self.base_name = base_name
        self.base_type = base_type
    def __str__(self):
        msg = f"""\
        An issue has been encountered during construction. The tensor feature of name
        {self.violating_name} had dtype {self.violating_type}. However, the tensor feature of
        name {self.base_name} had dtype {self.base_type}. Differing dtypes are not allowed. 
        """
        msg = Core.dedent(msg)
        return msg

class ConstructorBadTensorDeviceError(ConstructorError):
    def __init__(self, violating_name: str, violating_device: torch.device, base_name: str, base_device: torch.device):
        self.violating_name = violating_name
        self.violating_device = violating_device
        self.base_name = base_name
        self.base_device = base_device

    def __str__(self):
        msg = f"""\
        An issue has been encountered during construction. The tensor feature of name
        {self.violating_name} had device {self.violating_device}. However, the tensor feature of
        name {self.base_name} had device {self.base_device}. Differing devices are not allowed. 
        """
        msg = Core.dedent(msg)
        return msg

class ConstructorBadTensorBatchShape(ConstructorError):
    def  __init__(self,
                    violating_name: str,
                    total_shape: List[int],
                    violating_shape: List[int],
                    expected_name: str,
                    expected_shape: List[int],
                    ):

        self.violating_name = violating_name
        self.total_shape = total_shape
        self.violating_shape = violating_shape
        self.expected_name = expected_name
        self.expected_shape = expected_shape

    def __str__(self):
        msg = f"""\
        An issue has been encountered while making a new state tensor. The tensor feature
        of name {self.violating_name} had shape {self.total_shape}.

        Its batch shape was {self.violating_shape}. However, the expected batch shape, based 
        on {self.expected_name}, was {self.expected_shape}.

        Since these are not the same, the state tensor failed to syncronize and could
        not be made.
        """
        msg = Core.dedent(msg)
        return msg

class  ConstructorTensorsViolateDimensionalConstraint(ConstructorError):
    def __init__(self,
            violated_constraint: str,
            tensor_a_name: str,
            tensor_a_dim_num: int,
            tensor_a_dim_len: int,
            tensor_b_name: str,
            tensor_b_dim_num: int,
            tensor_b_dim_len: int
            ):

        self.violated_constraint = violated_constraint
        self.tensor_a_name = tensor_a_name
        self.tensor_a_dim_num = tensor_a_dim_num
        self.tensor_a_dim_len = tensor_a_dim_len
        self.tensor_b_name = tensor_b_name
        self.tensor_b_dim_num = tensor_b_dim_num
        self.tensor_b_dim_len = tensor_b_dim_len

    def __str__(self)->str:
        msg = f"""\
        An issue has been encountered while making a new bundle tensor. The
        constraints established in tensor {self.tensor_a_name} do not match those
        in {self.tensor_b_name} for constraint {self.violated_constraint}.
        
        In particular, a constraint was placed on '{self.tensor_a_name}' for dimension
        {self.tensor_a_dim_num} which indicated the length should be {self.tensor_a_dim_len}.
        
        However, tensor '{self.tensor_b_name}''s dimension {self.tensor_b_dim_num} had length
        of {self.tensor_b_dim_len}.
        
        Since these are not the same, a bundle tensor could not be made.
        """
        msg = Core.dedent(msg)
        return msg

class  ArithmeticBundleTensorNotFound(ArithmeticError):
    def __str__(self):
        msg = """\
        An issue occurred when performing bioperand arithmetic. It was 
        expected that at least one of the two operands was a bundle tensor,
        but neither was found to be one.
        """
        msg = Core.dedent(msg)
        return msg

class ArithmeticBadBatchRanks(ArithmeticError):

    def __init__(self,
                 preoperand_batch_rank: int,
                 postoperand_batch_rank: int):

        self.preoperand_rank = preoperand_batch_rank
        self.postoperand_rank = postoperand_batch_rank

    def __str__(self)->str:
        msg = f"""\
            An issue occurred when performing bioperand arithmetic. It was 
            expected that when doing operations between two state tensors the 
            state tensors will have the same batch rank. However, the first operand
            has batch rank {self.preoperand_rank} while the second has batch rank 
            {self.postoperand_rank}
            """
        msg = Core.dedent(msg)
        return msg

class ArithmeticBadBatchShapes(ArithmeticError):
    def __init__(self,
                 preoperand_shape: List[int],
                 postoperand_shape: List[int]
                 ):

        self.preoperand_shape = preoperand_shape
        self.postoperand_shape = postoperand_shape
    def __str__(self):
        msg = f"""\
            An issue occurred when performing bioperand arithmetic. It was 
            expected that when doing operations between two bundle tensors the 
            bundle tensors will have the same batch rank. However, the first operand
            has batch shape {self.preoperand_shape} while the second has batch shape 
            {self.postoperand_shape}. Since these are not the same, the operation
            failed.
            """
        msg = Core.dedent(msg)
        return msg


class ArithmeticBadStateKeys(ArithmeticError):
    def __str__(self):
        msg = f"""\
             An issue occurred when performing bioperand state arithmetic. It was 
             expected that arithmetic between two states will involve the states
             sharing keys. However, the keys in the preoperand did not match the
             keys in the post operand.
             """
        msg = Core.dedent(msg)
        return msg

class ArithmeticCannotReverseBroadcastError(ArithmeticError):
    def  __init__(self,
                    key_name: str,
                    preoperand_shape: List[int],
                    postoperand_shape: List[int],
                    location: int):

        self.key_name = key_name
        self.preoperand_shape = preoperand_shape
        self.postoperand_shape = postoperand_shape
        self.location = location

    def __str__(self) -> str:
        msg = f"""\
        An issue occurred when performing bioperand state arithmetic. It was not possible
        to reverse broadcast across key {self.key_name} at dimension {self.location}. The preoperand
        shape was shape {self.preoperand_shape} and had value {self.preoperand_shape[self.location]} at
        the location of interest. However, the postoperand shape was {self.postoperand_shape}
        and had value of {self.postoperand_shape[location]} at the same place.
        """
        msg = Core.dedent(msg)
        return msg





class BundleTensor:
    """
    The batch bundle tensor is a tensor designed to carry around within it
    a bundle of tensors which may be said to be coupled together based on
    having common batch or ensemble dimensions, along with associated
    tensor based logic for manipulating the entire bundle at once. The
    pay off is the ability to do things like automatically catch insane
    errors, and automatically mix together state information in, for
    example, a router transformer context.

    ---- A few words of warning ---

    Note that this class is entirely immutable. This means that any
    operation which makes a change will in turn return a new updated instance.

    This means you should not use, for example:

    ```
    bundle.set(...)
    ```

    but instead

    ```
    bundle = bundle.set(...)
    ```
    ------ constructor ----

    The constructor is one of the most important things to understand
    when working with a batch bundle tensor. The constructor accepts a parameter
    indicating how many batch dimensions to enforce, a dictionary for containing
    the data tensors we will be using, and another optional dictionary for
    channel constraints.

    The tensor is designed to operate similarly to a dictionary. Tensors
    may be named, inserted using set, or retrieved. However, unlike a
    dictionary considerable restraints are placed upon the contents based
    on the tensor status.


    ------ constructor: syncronization of batch dimensions.  ----


    First, all tensors in the dictionary are required to have common batch dimensions. These
    are the dimensions at the beginning of the tensor, and the number of batch dimensions is
    a parameter settable on construction in case you wish to do something clever, like use ensembles.
    Conceptually, manipulation of the entire bundle occurs across the batch dimensions. Attempts to
    set tensors into the bundle which do not share the current batch dimension will fail, as will
    attempts to make a bundle with none matching batch dimensions

    For example, if you have a tensor named "a" and "b" as follows, you could make a batch bundle

    ```
    a = torch.randn(5, 3, 7)
    b = torch.rand(5, 3, 9)

    # This works. Note how one batch dimension is specified. The constructor notices it
    # is '5' for both.

    batch_bundle = state_tensors.BatchBundleTensor(1, {'a' : a, 'b' : b})
    # This also works. Two dimensions, '5, 3', must be matched

    batch_bundle = state_tensors.BatchBundleTensor(2, {'a' : a, 'b' : b})

    ```

    However, since the third dimension is NOT common, the following would not work

    ```
    # Does not work, due to 3 not being common between a and b
    batch_bundle = state_tensors.BatchbundleTensor(3, {'a' : a, 'b' : b})
    ```

    It is also the case that all tensors in the future must satisfy the common batch shape,
    or alternatively must all be changed to a new shape at once.

    ```
    batch_bundle = state_tensors.BatchBundleTensor(2, {'a' : a, 'b' : b})

    c = torch.rand([5, 3, 2, 9])
    d = torch.rand([3, 2, 7, 9])

    # works
    batch_bundle = batch_bundle.set("c", c)

    # does NOT work
    batch_bundle = batch_bundle.set("d", d)
    ```

    ---- constructor: Additional restrictions with dimensional constraints ----

    Support is also provided for what are referred to as dimensional constraints. These
    are defined with respect the last dimensions as a list of names per initialized tensor,
    working from the last dimension to the left. It is designed to ensure that all tensors
    with the same dimensional constraint have the same length.

    Lets first look at how the constructor associates and constructs constraints. Lets suppose
    you have a tensor of shape [5, 4, 3, 2, 7]. Passing in a constraint with ["items", "embedding"] would
    associated dim "2" with items and "7" with embeddings, and assert any future items, embeddings must
    have the same dimensionality. This would look something like as follows if we are setting up a
    bundle with only this as an entry

    ```
    data_dict = {"a" : torch.rand([5, 4, 3, 2, 7])
    constraint_dict = {"a" : ["items", "embedding"]}
    batch_bundle = state_tensors.BatchBundleTensor(1, data_dict, constraints_dict)
    ```

    Notice how the constraints dict matches the key in the data dict, and how the constructor will
    now corrolate those two. The payoff is that if you try to insert something later on, or create a
    dict, with nonmatching dimensional constraints an error is thrown and the bundle says this will not
    work as well as why. For example

    ```

    # Works. Note dimensions are the same
    batch_bundle = batch_bundle.set("b", torch.rand(5, 2, 7), ["items", "embedding"])

    # Works. Note no constraints are coupled

    batch_bundle = batch_bundle.set("c", torch.randn(5, 4, 7))

    # Does NOT work. Raises error

    batch_bundle = batch_bundle.set("d", torch.rand(5, 2, 8), ["items", "embedding"])
    ```


    ---- manipulation ----

    the class can be manipulated almost exactly like a dictionary, with a
    few exceptions with regards to how to perform updates. First, to just
    set a new tensor of the same shape. Namely, the following functions:

    * set:
    * replace all
    * update
    * remove

    Are functional and instead return a new instance.

    --- arithmetic ----

    Arithmetic is also possible here. It comes in three flavors. One can perform scalar
    arithmetic with floats or ints, tensor arithmetic using reverse broadcasts, state arithmetic
    using reverse broadcasts. Consider the two operands.
    One operand must be a state tensor. The other may be:

    * scalar: Fairly standard stuff. You can multiply by int or float, it distributes
    * tensor: Any tensor must be reverse broadcastable with regards to the batch shape. Attempts
              to perform tensor operations beyond the batch shape will throw an error.
    * reverse broadcastable state:
             A state tensor containing keys which are reverse broadcastable for each individual
             entry in the dictionary. This allows more fine grained control. Of course, it
             also could just end up working element by element.

    The supported arithmetic operations are

    * negate
    * add
    * subtract
    * multiply
    * divide
    * power

    As an example, lets suppose you have a routing transformer and only want to update your
    state information according to what ensemble was being emphasized this round. It mightl
    look like below

    ```

    # A routing transformer with a batch size of 10 and 8 ensembles
    routing_weights = torch.rand(10, 8)

    # state

    state  = ... # original tensor bundle. Has batch size of 2, and info of dim 10, 8 for batch state
    new_state = ... # new state. Same size
    state = routing_weights * state + (1-routing_weights)*new_state
    ```

    Observe how no matter what the contents were, it is elegantly combined together into the new state
    tensor. Very useful.

    """


    # Define dictionary like helpers

    @property
    def dtype(self)->torch.dtype:
        return self._dtype

    @property
    def device(self)->torch.device:
        return self._device

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def __len__(self) -> int:
        return len(self.tensors)

    def __repr__(self) -> str:
        return str(self.tensors)

    def __str__(self) -> str:
        return str(self.tensors)

    def __eq__(self, other: 'BundleTensor') -> bool:
        if not isinstance(other, BundleTensor):
            return False
        if other.keys() != self.keys():
            return False
        for key, my_vals, their_vals in zip(self.keys(),
                                            self.values(),
                                            other.values()):
            if my_vals.shape != their_vals.shape:
                return False
            if not torch.all(my_vals == their_vals):
                return False
        return True

    def __ne__(self, other: 'BundleTensor') -> bool:
        return not self.__eq__(other)

    def __iter__(self):
        return self.tensors.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.tensors.keys()

    def set(self, key: str, tensor: torch.Tensor, dim_names: Optional[List[str]] = None) -> 'BundleTensor':
        """
        Sets a particular key to a particular tensor value. Will throw an
        error if you are now violating constraints. It can be used to add
        entries or constraints to the state tensor.

        Note that if you reshape a tensor using this set, and the new shape
        violates the constraint on another tensor, you will raise an error. In
        that case, you will need to use replace_all to change all the tensors
        with constraints at once.

        :param key: The key to set
        :param value: The new tensor to set here.
        :return: A new BatchStateTensor
        """
        if dim_names is None:
            dim_names = self.constraints_spec[key]

        new_tensor_data = dict(self.tensors)
        new_dim_data = dict(self.constraints_spec)

        new_tensor_data[key] = tensor
        new_dim_data[key] = dim_names

        return BundleTensor(
            self.batch_dim,
            new_tensor_data,
            new_dim_data,
            self.validate)

    def replace_all(self, tensors: Dict[str, torch.Tensor])-> 'BundleTensor':
        """
        Sets a group of keys to particular new values. This may be utilized to modify
        things with several constraints on them. It may NOT be used to modify constraints

        :param tensors: Dict[str, torch.Tensor]. The tensors to replace.
        :return: A new batchStateTensor
        :raises: ValueError from the constructor if your tensors or constraints are bad.
        """
        new_data = dict(self.tensors)
        for key, value in tensors.items():
            assert key in new_data, f"tensor of key {key} was never in state tensor"
            new_data[key] = tensors[key]
        return BundleTensor(self.batch_dim,
                            new_data,
                            self.constraints_spec,
                            self.validate)


    def remove(self, key: str)-> 'BundleTensor':
        """
        Removes a tensor entry completely.

        :param key: The key to remove
        :return: A state tensor with the key removed.
        """

        new_tensors = self.tensors.copy()
        new_constraints = self.constraints_spec.copy()

        new_tensors.pop(key)
        new_constraints.pop(key)

        return BundleTensor(self.batch_dim,
                            new_tensors,
                            new_constraints,
                            self.validate)

    def __setitem__(self, key, value):
        msg = "Cannot modify a state tensor directly\n"
        msg += "You should instead use the set methods provided on the main class"
        raise TypeError("Cannot modify an immutable dictionary.")

    def tensor_hash(self, tensor: torch.Tensor) -> int:
        # Flatten the tensor to a list
        data: List[float] = tensor.flatten().tolist()
        str_data = [str(item) for item in data]

        # Call the script function with the list
        return hash(''.join(str_data))

    def __hash__(self) -> int:
        h = 0
        for k, v in self.tensors.items():
            h = hash(31 * h + hash(k) + self.tensor_hash(v))
        return h

    def items(self) -> List[Tuple[str, torch.Tensor]]:
        return [(key, value) for key, value in self.tensors.items()]

    def keys(self) -> List[str]:
        return list(self.tensors.keys())

    def values(self) -> List[torch.Tensor]:
        return list(self.tensors.values())

    def update(self, other: 'BundleTensor') -> 'BundleTensor':

        assert other.batch_dim == self.batch_dim
        new_tensors = self.tensors.copy()
        new_constraints = self.constraints_spec.copy()

        new_tensors.update(other.tensors)
        new_constraints.update(other.constraints_spec)

        return BundleTensor(
            self.batch_dim,
            new_tensors,
            new_constraints,
            self.validate
        )

    def to(self,
           dtype: Optional[torch.dtype] = None,
           device: Optional[torch.device] = None)->'BundleTensor':

        if dtype is None:
            dtype = self.dtype


    ## Constructor validation helpers ##
    def _constructor_validate_batch_details(self,
                               batch_dims: int,
                               tensors: Dict[str, torch.Tensor]):
        """
        Validates that batch shape, dtypes, device are all sane.
        """

        # Verify that we actually were handled tensors to make
        if len(tensors) == 0:
            raise ConstructorNoTensorsError()

        # Verify the rank is satisfactory for all entries
        for key, tensor in tensors.items():
            if batch_dims > tensor.dim():
                raise ConstructorBadRankError(key, batch_dims, tensor.dim())

        # Verify that the dtype, device, and batch shape is satisfactory for all entries.
        #
        # Also, check if there are any named tensors I need to worry about.

        base_key = list(tensors.keys())[0]
        base_dtype = tensors[base_key].dtype
        base_device = tensors[base_key].device
        batch_shape = tensors[base_key].shape[:batch_dims]
        for key, tensor in tensors.items():
            # Check dtype is sane
            if tensor.dtype != base_dtype:
                raise ConstructorBadTensorDtypeError(key,
                                                     tensor.dtype,
                                                     base_key,
                                                     base_dtype)
            # Check device is sane
            if tensor.device != base_device:
                raise ConstructorBadTensorDeviceError(key,
                                                      tensor.device,
                                                      base_key,
                                                      base_device)

            # Check shape is common
            if tensor.shape[:batch_dims] != batch_shape:
                raise ConstructorBadTensorBatchShape(key,
                                                     list(tensor.shape),
                                                     list(tensor.shape[:batch_dims]),
                                                          base_key,
                                                          batch_shape)

    def _constructor_validate_constraints(self,
                                          tensors: Dict[str, torch.Tensor],
                                          dim_names: Dict[str, List[str]],
                                          ):
        """
        Validate that the constraints are all sane
        """

        # Will contain key constrant, then value constraint source, constraint dim, dim len
        validation_table: Dict[str, Tuple[str, int, int]] = {}

        for key, tensor in tensors.items():
            names = dim_names[key]
            locs = range(tensor.dim() - len(names), tensor.dim())
            # Go through each name and it's associated dimension for this tenso
            for name, loc in zip(names, locs):
                dim = tensor.shape[loc]
                # If we have not seen this before, add it to the table.
                if name not in validation_table:
                    validation_table[name] = (key, loc, dim)
                # We have seen it before. Ensure
                else:
                    source_name, source_loc, expected_len = validation_table[name]
                    if expected_len != dim:
                        raise ConstructorTensorsViolateDimensionalConstraint(name,
                                                                             source_name,
                                                                             source_loc,
                                                                             expected_len,
                                                                             key,
                                                                             loc,
                                                                             dim)

    def _constructor_get_constraint_lengths(self,
                                            tensors: Dict[str, torch.Tensor],
                                            dim_names: Dict[str, List[str]]

                                            ) -> Dict[str, int]:

        # Gets the constraints and their corrolated values
        output: Dict[str, int] = {}

        for key, tensor in tensors.items():
            names = dim_names[key]
            locs = range(tensor.dim() - len(names), len(names))
            # Go through each name and it's associated dimension for this tenso
            for name, loc in zip(names, locs):
                output[name] = tensor.size(loc)
        return output

    def _constructor_get_constraint_index_corrolations(self,
                                                       tensors: Dict[str, torch.Tensor],
                                                       dim_names: Dict[str, List[str]],
                                                       ) -> Dict[str, Dict[str, int]]:
        # Gets a datastructure that indicates first the indicates tensor feature,
        # then the constraints within, and finally the index that this constraint connects to

        state_corrolations: Dict[str, Dict[str, int]] = {}

        for key, tensor in tensors.items():
            tensor_corrolations: Dict[str, int] = {}
            names = dim_names[key]
            locs = range(tensor.dim() - len(names), len(names))
            # Go through each name and it's associated dimension for this tenso
            for name, loc in zip(names, locs):
                tensor_corrolations[name] = loc
            state_corrolations[key] = tensor_corrolations
        return state_corrolations

    ## Constructor ##
    def __init__(self,
                 # Essential features
                 batch_dims: int,
                 tensors: Dict[str, torch.Tensor],
                 dim_names: Optional[Dict[str, List[str]]] = None,
                 validate: bool = True,
                 ):
        """

        Primary constructor parameters

        :param batch_shape: The shape of the batch and ensemble dimensions at the first dimensions of the model
                            For example, [3, 5] would expect all tensors to start with shape [3, 5]
        :param tensors: The actual tensors which will be stored away. They may have named dimensions, in which
                        case said dimensions MUST have the same length.
        :param dim_names: Dimensions are named in order to provide additional constraints against this
                      tensor. A tensor is named in broadcast format from left to right, and may have
                      unnamed dimensions.

                      For example, for a tensor of shape [5, 3, 4, 2] with
                      passed dict entry names["tensor"] = ["items", "embedding"] would associate
                      'items' with 4 and 'embedding' with 3.

        :param validate: Whether to run validation.
        """

        assert batch_dims >= 0
        if dim_names is None:
            updates: Dict[str, List[str]] = {}
            for key in tensors.keys():
                # We have to hop through some hoops for torchscript here.
                associated_list: List[str] = []
                updates[key] = associated_list
            dim_names = updates

        if validate:
            self._constructor_validate_batch_details(batch_dims, tensors)
            self._constructor_validate_constraints(tensors, dim_names)

        self.constraints_spec = dim_names
        self.constraint_lengths = self._constructor_get_constraint_lengths(tensors, dim_names)
        self.constraint_indices = self._constructor_get_constraint_index_corrolations(tensors, dim_names)
        self.tensors = tensors
        self.batch_dim = batch_dims

        key_zero = list(tensors.keys())[0]
        entry = tensors[key_zero]

        self.batch_shape = entry.shape[:batch_dims]
        self._dtype = entry.dtype
        self._device = entry.device
        self.validate = validate

    # Torchscripts type refinement is a bitch and a half. There is no way
    # around massive code duplication with differing overrides.

    def execute_bioperand_arithmetic(self,
                                     feature_name: str,
                                     preoperand: torch.Tensor,
                                     operator: str,
                                     postoperand: torch.Tensor):
        """
        Basically, this is responsible for knowing exactly how to execute two operands when called upon.
        :return: The executed operation
        """
        # We must reverse broadcast in some cases. We will detect here if it is needed,
        # and if so unsqueeze

        # Identify and comment on any deficiencies between the tensors


        if isinstance(preoperand, torch.Tensor) and isinstance(postoperand, torch.Tensor):
            # Comment on any deficiencies in tensor shape if existant
            #
            # Recall that zip will only proceed as far as the shortest list, so this works
            for i, (pre_dim, post_dim) in enumerate(zip(preoperand.shape, postoperand.shape)):
                if pre_dim == post_dim:
                    continue
                if pre_dim == 1:
                    continue
                if post_dim == 1:
                    continue
                raise ArithmeticCannotReverseBroadcastError(feature_name,
                                                            preoperand.shape,
                                                            postoperand.shape,
                                                            i)

            while preoperand.dim() < postoperand.dim():
                preoperand = preoperand.unsqueeze(-1)
            while postoperand.dim() < preoperand.dim():
                postoperand = postoperand.unsqueeze(-1)


        if operator == "add":
            return preoperand + postoperand
        elif operator == "subtract":
            return preoperand - postoperand
        elif operator == "multiply":
            return preoperand * postoperand
        elif operator == "divide":
            return preoperand / postoperand
        elif operator == "power":
            return preoperand ** postoperand
        else:
            raise ValueError(f"Illegal operator type: {operator} not supported")

    def process_operand_stack(self,
                              keys: List[str],
                              preoperands: Dict[str, torch.Tensor],
                              operator: str,
                              postoperands: Dict[str, torch.Tensor]
                              )->Dict[str, torch.Tensor]:
        tensor_updates = {}
        for key in keys:
            preoperand = preoperands[key]
            postoperand = postoperands[key]
            tensor_updates[key] = self.execute_bioperand_arithmetic(key,
                                                                    preoperand,
                                                                    operator,
                                                                    postoperand)
        return tensor_updates


    def perform_bioperand_arithmetic(self,
                                     operand_one: Union[int, float, torch.Tensor, 'BundleTensor'],
                                     operator: str,
                                     operand_two: Union[int, float, torch.Tensor, 'BundleTensor'])\
                                -> 'BundleTensor':
        """
        This is the base function used to impliment arithmetic for the magic methods. It
        is defined in terms of two operands and the accompanying operator. It is a helper method,
        and is used by feeding in the operation conditions and the operator when implimenting
        a magic methods.
        :param operand_one: The first operator to use. May be among int, float, tensor ,state tensor
        :param operator: The operator to use. Must be among "add", "subtract", "multiply", "divide", or "power"
        :param operand_two: The second operand to use
        :return: The result of performing arithmetic among the entries
        """

        # We prepare the operation here by making pairs of
        # syncronous key, value tensors with an entry for every
        # item in the state tensor. Due to the format of the
        # input, a lot of work has to go into torchscript type refinement

        if isinstance(operand_one, (int, float, torch.Tensor)):
            if not isinstance(operand_two, BundleTensor):
                raise ArithmeticBundleTensorNotFound()

            # These if statements perform torchscript type refinement. They are needed.
            if isinstance(operand_one, int):
                auxilary_tensor = torch.tensor(operand_one)
            elif isinstance(operand_one, float):
                auxilary_tensor = torch.tensor(operand_one)
            elif isinstance(operand_one, torch.Tensor):
                auxilary_tensor = operand_one
            else:
                # Really just here to make torchscript happy. Can never happen
                raise ValueError("Illegal state. Hit the programmer")

            root = operand_two
            operand_one_stack = {key : auxilary_tensor for key in operand_two.keys()}
            operand_two_stack = operand_two.tensors.copy()

        elif isinstance(operand_two, (int, float, torch.Tensor)):
            if not isinstance(operand_one, BundleTensor):
                raise ArithmeticBundleTensorNotFound()

            # These if statements perform torchscript type refinement. They are needed
            if isinstance(operand_two, int):
                auxilary_tensor = torch.tensor(operand_two)
            elif isinstance(operand_two, float):
                auxilary_tensor = torch.tensor(operand_two)
            elif isinstance(operand_two, torch.Tensor):
                auxilary_tensor = operand_two
            else:
                raise ValueError("Impossible condition reached") # here to keep torchscript and IDE happy

            root = operand_one
            operand_one_stack = operand_one.tensors.copy()
            operand_two_stack =  {key: auxilary_tensor for key in operand_one.keys()}

        elif isinstance(operand_one, BundleTensor) and isinstance(operand_two, BundleTensor):
            if operand_one.batch_dim != operand_two.batch_dim:
                raise ArithmeticBadBatchRanks(operand_one.batch_dim,
                                              operand_two.batch_dim)
            if operand_one.batch_shape != operand_two.batch_shape:
                raise ArithmeticBadBatchShapes(list(operand_one.batch_shape),
                                               list(operand_two.batch_shape))
            if operand_one.keys() != operand_two.keys():
                raise ArithmeticBadStateKeys()

            root = operand_one
            operand_one_stack = operand_one.tensors.copy()
            operand_two_stack = operand_two.tensors.copy()


        else:
            raise ValueError("Illegal operand type")

        updates = self.process_operand_stack(root.keys(),
                                             operand_one_stack,
                                             operator,
                                             operand_two_stack)

        return BundleTensor(root.batch_dim,
                            updates,
                            root.constraints_spec,
                            root.validate)

    # Define the arithmetic magic methods

    def __add__(self, other):
        return self.perform_bioperand_arithmetic(self, "add", other)
    def __radd__(self, other):
        return self.perform_bioperand_arithmetic(other, "add", self)
    def __sub__(self, other):
        return self.perform_bioperand_arithmetic(self, "subtract", other)
    def __rsub__(self, other):
        return self.perform_bioperand_arithmetic(other, "subtract", self)
    def __mul__(self, other):
        return self.perform_bioperand_arithmetic(self, "multiply", other)
    def __rmul__(self, other):
        return self.perform_bioperand_arithmetic(other, "multiply", self)
    def __truediv__(self, other):
        return self.perform_bioperand_arithmetic(self, "divide", other)
    def __rtruediv__(self, other):
        return self.perform_bioperand_arithmetic(other, "divide", self)
    def __pow__(self, other):
        return self.perform_bioperand_arithmetic(self, "power", other)
    def __rpow__(self, other):
        return self.perform_bioperand_arithmetic(other, "power", self)

torch.jit.script(BundleTensor)


