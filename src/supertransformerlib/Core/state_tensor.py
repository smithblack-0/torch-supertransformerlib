"""
A small module to contain classes designed to ease syncronization between collections of
state tensors

This module contains logic for working with this sort of operation. Emphasis is placed on
the ability to perform arithmetic with respect to tensors, scalars, and other state
tensors.

State tensors are designed to be as functional as possible. Modification attempts
will throw if not being set through the proper channels.

Prototyping:

* Builder exists to make prototype creator.
* Prototype can then be build by specifying feature name, feature dim labels,
* Prototype can be finished
* Prototype can be called with batch_dims, Dict[str, torch.Tensor] and will
  attempt to build a state tensor to match. It will throw if something is detected as going
  wrong.

State Tensors:

* Arithmetic
    * State tensors can perform arithmetic with scalars. This broadcasts across the contents.
    * State tensors can perform arithmetic with tensors. This broadcasts across the batch dims
    * State tensors can perform arithmetic with other state tensors. This operates as follows:
        * If keys differ, throw.
        * If the tensors have the same shape, just proceed.
        * If the tensors have different shapes, throw

*
"""

import torch
from typing import Optional, Dict, List, Tuple, Union
from supertransformerlib import Core

class ErrorMessageBuilder:
    """
    Primarily a storage helper for error messages,
    to help restrict my namespace.
    """
    @staticmethod
    def constructor_no_tensors_message()->str:
        msg = """\
        An issue has occurred when creating a state tensor. The 
        state tensor was expected to be initialized with a collection
        of tensors. However, no tensors were actually provided in the
        dictionary.
        """
        msg = Core.dedent(msg)
        return msg

    @staticmethod
    def constructor_bad_rank_message(violating_name: str,
                                     expected_rank: int,
                                     got_rank: int)->str:
        msg = f"""\
        An issue has occurred when creating a state tensor. The rank
        for tensor {violating_name} is bad. The expected rank, to
        match the batch dims, was {expected_rank}. However, it 
        was found instead to have rank {got_rank}.
        
        Since {got_rank} is less than {expected_rank}, construction failed.
        """
        msg = Core.dedent(msg)
        return msg

    @staticmethod
    def constructor_bad_tensor_dtype_message(violating_name: str,
                                             violating_type: torch.dtype,
                                             base_name: str,
                                             base_type: torch.dtype)->str:
        msg = f"""\
        An issue has been encountered during construction. The tensor feature of name
        {violating_name} had dtype {violating_type}. However, the tensor feature of
        name {base_name} had dtype {base_type}. Differing dtypes are not allowed. 
        """
        msg = Core.dedent(msg)
        return msg

    @staticmethod
    def constructor_bad_tensor_device_message(violating_name: str,
                                              violating_device: torch.device,
                                              base_name: str,
                                              base_device: torch.device)->str:
        msg = f"""\
        An issue has been encountered during construction. The tensor feature of name
        {violating_name} had device {violating_device}. However, the tensor feature of
        name {base_name} had device {base_device}. Differing dtypes are not allowed. 
        """
        msg = Core.dedent(msg)
        return msg

    @staticmethod
    def constructor_bad_tensor_batch_shape_message(violating_name: str,
                                                   total_shape: List[int],
                                                   violating_shape: List[int],
                                                   expected_name: str,
                                                   expected_shape: List[int],
                                                   )->str:
        msg = f"""\
        An issue has been encountered while making a new state tensor. The tensor feature
        of name {violating_name} had shape {total_shape}.
        
        Its batch shape was {violating_shape}. However, the expected batch shape, based 
        on {expected_name}, was {expected_shape}.
        
        Since these are not the same, the state tensor failed to syncronize and could
        not be made.
        """
        msg = Core.dedent(msg)
        return msg

    @staticmethod
    def constructor_tensors_violate_constraint(
            violated_constraint: str,
            tensor_a_name: str,
            tensor_a_dim_num: int,
            tensor_a_dim_len: int,
            tensor_b_name: str,
            tensor_b_dim_num: int,
            tensor_b_dim_len: int
            )->str:
        msg = f"""\
        An issue has been encountered while making a new state tensor. The
        constraints established in tensor {tensor_a_name} do not match those
        in {tensor_b_name} for constraint {violated_constraint}.
        
        In particular, a constraint was placed on {tensor_a_name} for dimension
        {tensor_a_dim_num} which indicated the length should be {tensor_a_dim_len}.
        
        However, tensor {tensor_b_name}'s dimension {tensor_b_dim_num} had length
        of {tensor_b_dim_len}.
        
        Since these are not the same, a state tensor could not be made.
        """
        msg = Core.dedent(msg)
        return msg

    @staticmethod
    def arithmetic_no_state_tensor():
        msg = """\
        An issue occurred when performing bioperand state arithmetic. It was 
        expected that at least one of the two operands was a state tensor,
        but neither was found to be one.
        """
        msg = Core.dedent(msg)
        return msg

    @staticmethod
    def arithmetic_bad_batch_dims(preoperand_batch_rank: int,
                                  postoperand_batch_rank: int):
        msg = f"""\
        An issue occurred when performing bioperand state arithmetic. It was 
        expected that when doing operations between two state tensors the 
        state tensors will have the same batch rank. However, the first operand
        has batch rank {preoperand_batch_rank} while the second has batch rank 
        {postoperand_batch_rank}
        """
        msg = Core.dedent(msg)
        return msg

    @staticmethod
    def arithmetic_bad_batch_shape(preoperand_shape: List[int],
                                   post_operand_shape: List[int]):
        msg = f"""\
        An issue occurred when performing bioperand state arithmetic. It was 
        expected that when doing operations between two state tensors the 
        state tensors will have the same batch shape. However, the preoperand
        has batch shape {preoperand_shape} while the postoperand has batch shape 
        {post_operand_shape}
        """
        msg = Core.dedent(msg)
        return msg

    @staticmethod
    def arithmetic_bad_state_keys()->str:
        msg = f"""\
        An issue occurred when performing bioperand state arithmetic. It was 
        expected that arithmetic between two states will involve the states
        sharing keys. However, the keys in the preoperand did not match the
        keys in the post operand.
        """
        msg = Core.dedent(msg)
        return msg

    @staticmethod
    def arithmetic_cannot_reverse_broadcast(key_name: str,
                                            preoperand_shape: List[int],
                                            postoperand_shape: List[int],
                                            location: int):
        msg = f"""\
        An issue occurred when performing bioperand state arithmetic. It was not possible
        to reverse broadcast across key {key_name} at dimension {location}. The preoperand
        shape was shape {preoperand_shape} and had value {preoperand_shape[location]} at
        the location of interest. However, the postoperand shape was {postoperand_shape}
        and had value of {postoperand_shape[location]} at the same place.
        """
        msg = Core.dedent(msg)
        return msg

        pass


class BatchStateTensor:
    """
    The native state tensor. This contains within it information on synchronized
    tensors and their associated current shapes. It also allows easy execution
    of arithmetic operations

    It is strongly focused on properly asserting restrictions are respected. It also
    has a strong focus on making arithmetic work.


    --- construction ---

    A state tensor is constructed using a group of tensors which need to be linked together
    as part of a broader batch. The tensors may be named tensors, in which case they will
    have named dimensions. Named dimensions must be the same length, and regardless the
    batch dims must be the same as well. The dtype and device must also all be the same.

    Entries are constructed by specifying a number of batch dimensions
    and passing in a corrolated dictionary of name:tensors and name:dimension_labels.
    The various tensor entries must have the same batch shape, starting from the leftmost
    tensor dimension

    As an example, if you have batch dims of 1, or 2, the following would work. However,
    3 would not

    tensors = {}
    tensors["weights"] = torch.randn([3, 4, 6, 5])
    tensors["defaults"] = torch.randn([3, 4, 2, 5])

    Observe the first two dimensions are syncronized, but after that
    they are not. Now, in addition to this, we can also specify
    names for dimensions. These must then be the same length in all
    subsequent operations. For example, lets suppose the last dimension
    of the above were a common embedding dim. We could handle that as:

    names = {}
    names["weights"] = ["embeddings"]
    names["defaults"] = ["embeddings"]

    This can then be used as

    BatchStateTensor(batch_dims = 2,
                    tensors,
                    names)

    --- getting, setting, and modications ---

    The class, once setup, can be modified but will always return a new
    instance. It is, to as great a degree as possible, functional.

    Getting is fairly simple. Just use __getitem__, as in StateTensor["weights"] to fetch
    what you are looking for.

    Setting is another story. There are two primary setters to worry about. These are .set and .replace

    .set: Sets a single feature, returns a new tensor
    .replace: Replaces multiple things at once.

    Note that you can only set a tensor to a new shape if mutability is allowed, and you
    still have to update all tensors which have the same changed dimension at once

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

    """


    # Define dictionary like helpers


    def __getitem__(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def __len__(self) -> int:
        return len(self.tensors)

    def __repr__(self) -> str:
        return str(self.tensors)

    def __str__(self) -> str:
        return str(self.tensors)

    def __eq__(self, other: 'BatchStateTensor') -> bool:
        if not isinstance(other, BatchStateTensor):
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

    def __ne__(self, other: 'BatchStateTensor') -> bool:
        return not self.__eq__(other)

    def __iter__(self):
        return self.tensors.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.tensors.keys()

    def set(self, key: str, tensor: torch.Tensor, dim_names: List[str]) -> 'BatchStateTensor':
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
        new_tensor_data = dict(self.tensors)
        new_dim_data = dict(self.constraints_spec)

        new_tensor_data[key] = tensor
        new_dim_data[key] = dim_names

        return BatchStateTensor(
            self.batch_dim,
            new_tensor_data,
            new_dim_data,
            self.validate)

    def replace_all(self, tensors: Dict[str, torch.Tensor])->'BatchStateTensor':
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
        return BatchStateTensor(self.batch_dim,
                                new_data,
                                self.constraints_spec,
                                self.validate)


    def remove(self, key: str)->'BatchStateTensor':
        """
        Removes a tensor entry completely.

        :param key: The key to remove
        :return: A state tensor with the key removed.
        """

        new_tensors = self.tensors.copy()
        new_constraints = self.constraints_spec.copy()

        new_tensors.pop(key)
        new_constraints.pop(key)

        return BatchStateTensor(self.batch_dim,
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

    def update(self, other: 'BatchStateTensor') -> 'BatchStateTensor':

        assert other.batch_dim == self.batch_dim
        new_tensors = self.tensors.copy()
        new_constraints = self.constraints_spec.copy()

        new_tensors.update(other.tensors)
        new_constraints.update(other.constraints_spec)

        return BatchStateTensor(
            self.batch_dim,
            new_tensors,
            new_constraints,
            self.validate
        )

    ## Constructor validation helpers ##
    def _constructor_validate_batch_details(self,
                               batch_dims: int,
                               tensors: Dict[str, torch.Tensor]):
        """
        Validates that batch shape, dtypes, device are all sane.
        """

        # Verify that we actually were handled tensors to make
        if len(tensors) == 0:
            raise ValueError(ErrorMessageBuilder.constructor_no_tensors_message())

        # Verify the rank is satisfactory for all entries
        for key, tensor in tensors.items():
            if batch_dims > tensor.dim():
                raise ValueError(ErrorMessageBuilder.constructor_bad_rank_message(key, batch_dims, tensor.dim()))

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
                raise ValueError(ErrorMessageBuilder.constructor_bad_tensor_dtype_message(key,
                                                                                          tensor.dtype,
                                                                                          base_key,
                                                                                          base_dtype))
            # Check device is sane
            if tensor.device != base_device:
                raise ValueError(ErrorMessageBuilder.constructor_bad_tensor_device_message(key,
                                                                                           tensor.device,
                                                                                           base_key,
                                                                                           base_device))
            # Check shape is common
            if tensor.shape[:batch_dims] != batch_shape:
                msg = ErrorMessageBuilder.constructor_bad_tensor_batch_shape_message(key,
                                                                                     list(tensor.shape),
                                                                                     list(tensor.shape[:batch_dims]),
                                                                                     base_key,
                                                                                     batch_shape)
                raise ValueError(msg)

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
                        msg = ErrorMessageBuilder.constructor_tensors_violate_constraint(name,
                                                                                         source_name,
                                                                                         source_loc,
                                                                                         expected_len,
                                                                                         key,
                                                                                         loc,
                                                                                         dim
                                                                                         )
                        raise ValueError(msg)

    def _constructor_get_constraint_lengths(self,
                                            tensors: Dict[str, torch.Tensor],
                                            dim_names: Dict[str, List[str]]

                                            )->Dict[str, int]:

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
                                               )->Dict[str, Dict[str, int]]:
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
        self.batch_shape = tensors[key_zero].shape[:batch_dims]
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
                raise ValueError(ErrorMessageBuilder.arithmetic_cannot_reverse_broadcast(feature_name,
                                                                                         preoperand.shape,
                                                                                         postoperand.shape,
                                                                                         i))
            # Expand dimensions
            while preoperand.dim() < postoperand.dim():
                preoperand = preoperand.unsqueeze(-1)
            while postoperand.dim() < preoperand.dim():
                postoperand = postoperand.unsqueeze(-1)

        # Torchscript's type refinement system is really rather
        # silly in some ways. In order for the operator code to work,
        # torchscript MUST know what the type is by that point. The
        # following code provides this type refinement, then does
        # the actual math. We refine catagories of
        #
        # float, float
        # float, tensor
        # tensor, float
        # int, tensor,
        # tensor, int
        # tensor, tensor


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
                              ):
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
                           operand_one: Union[int, float, torch.Tensor, 'BatchStateTensor'],
                           operator: str,
                           operand_two: Union[int, float, torch.Tensor, 'BatchStateTensor'])\
                                ->'BatchStateTensor':
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
            if not isinstance(operand_two, BatchStateTensor):
                raise ArithmeticError(ErrorMessageBuilder.arithmetic_no_state_tensor())

            # These if statements perform torchscript type refinement. They are needed.
            if isinstance(operand_one, int):
                auxilary_tensor = torch.tensor(operand_one)
            elif isinstance(operand_one, float):
                auxilary_tensor = torch.tensor(operand_one)
            elif isinstance(operand_one, torch.Tensor):
                auxilary_tensor = operand_one
            else:
                raise ValueError() # Really just here to make torchscript happy

            root = operand_two
            operand_one_stack = {key : auxilary_tensor for key in operand_two.keys()}
            operand_two_stack = operand_two.tensors.copy()

        elif isinstance(operand_two, (int, float, torch.Tensor)):
            if not isinstance(operand_one, BatchStateTensor):
                raise ArithmeticError(ErrorMessageBuilder.arithmetic_no_state_tensor())

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

        elif isinstance(operand_one, BatchStateTensor) and isinstance(operand_two, BatchStateTensor):
            if operand_one.batch_dim != operand_two.batch_dim:
                raise ArithmeticError(ErrorMessageBuilder.arithmetic_bad_batch_dims(operand_one.batch_dim,
                                                                                    operand_two.batch_dim))
            if operand_one.batch_shape != operand_two.batch_shape:
                raise ArithmeticError(ErrorMessageBuilder.arithmetic_bad_batch_shape(operand_one.batch_shape,
                                                                                      operand_two.batch_shape))
            if operand_one.keys() != operand_two.keys():
                raise ArithmeticError(ErrorMessageBuilder.arithmetic_bad_state_keys())

            root = operand_one
            operand_one_stack = operand_one.tensors.copy()
            operand_two_stack = operand_two.tensors.copy()


        else:
            raise ValueError()

        updates = self.process_operand_stack(root.keys(),
                                             operand_one_stack,
                                             operator,
                                             operand_two_stack)

        return BatchStateTensor(root.batch_dim,
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

torch.jit.script(BatchStateTensor)


