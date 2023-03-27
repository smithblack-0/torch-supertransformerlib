from typing import Dict, Tuple, Iterator, List, Union, Optional
from supertransformerlib import Core
import torch

import torch
from typing import Dict


@torch.jit.script
class ImmutableDict:
    """
    This is a small helper class that presents an immutable dictionary
    for usage elsewhere. Attempts to modify this dictionary
    """
    def __init__(self, data: Dict[str, torch.Tensor]):
        self._data = data

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return str(self._data)

    def __str__(self) -> str:
        return str(self._data)

    def __eq__(self, other: 'ImmutableDict') -> bool:
        if not isinstance(other, ImmutableDict):
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

    def __ne__(self, other: 'ImmutableDict') -> bool:
        return not self.__eq__(other)

    def __iter__(self):
        return self._data.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._data.keys()

    def set(self, key: str, value: torch.Tensor) -> 'ImmutableDict':
        new_data = dict(self._data)
        new_data[key] = value
        return ImmutableDict(new_data)

    def remove(self, key: str)->'ImmutableDict':
        new_data = self._data.copy()
        new_data.pop(key)
        return ImmutableDict(new_data)

    def __setitem__(self, key, value):
        msg = "Cannot modify an immutable dictionary\n"
        msg += "You should instead use the set methods provided on the main class"
        raise TypeError("Cannot modify an immutable dictionary.")

    def __hash__(self) -> int:
        h = 0
        for k, v in self._data.items():
            h = hash(31 * h + hash(k) + hash(v))
        return h

    def items(self) -> List[Tuple[str, torch.Tensor]]:
        return list(self._data.items())

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def values(self) -> List[torch.Tensor]:
        return list(self._data.values())

    def update(self, other: 'ImmutableDict') -> 'ImmutableDict':
        new_data = dict(self._data)
        new_data.update(other._data)
        return ImmutableDict(new_data)

@torch.jit.script
class Storage:
    """
    A small data helper class which contains all
    the important information for the state tensor
    in a single location and also contains the validation
    mechanisms.

    This will be modified by it's parent state tensor,
    but exists to allow easy passing and initialization
    """

    def _check_keys_common(self,
                           feature_name: str,
                           weights_dict: ImmutableDict,
                           defaults_dict: ImmutableDict):
        """ Check all keys are properly shared """
        # We cannot use set here because torch.jit.script cannot
        # compile it
        weights_keys = weights_dict.keys()
        defaults_keys = defaults_dict.keys()
        if len(weights_keys) != len(defaults_keys):
            msg = f"""\
            Issue during validation: Dictionaries have different numbers of keys.

            In the {feature_name} dictionaries the weights dictionary had {len(weights_keys)} keys,
            while the defaults dictionary had {len(defaults_keys)} keys."""
            msg = Core.dedent(msg)
            raise ValueError(msg)
        for key in weights_keys:
            if key not in defaults_dict:
                msg = f"""\
                Issue during validation: Keys in {feature_name} weights dictionary not found in defaults dictionary.

                The key '{key}' was found in the {feature_name} weights dictionary, but was not found in the
                defaults dictionary."""
                msg = Core.dedent(msg)
                raise ValueError(msg)
        for key in defaults_keys:
            if key not in weights_dict:
                msg = f"""\
                Issue during validation: Keys in {feature_name} defaults dictionary not found in weights dictionary.

                The key '{key}' was found in the {feature_name} defaults dictionary, but was not found in the
                weights dictionary."""
                msg = Core.dedent(msg)
                raise ValueError(msg)

    def _check_dictionary_routine(self,
                                  dictionary_name: str,
                                  dtype: torch.dtype,
                                  device: torch.device,
                                  dicts: ImmutableDict
                                  ):
        """ Checks for matching dtype and device"""
        for name, value in dicts.items():
            if value.dtype != dtype:
                raise ValueError(f"{dictionary_name} of name {name} was not found to have dtype {dtype}")
            if value.device != device:
                raise ValueError(f"{dictionary_name} of name {name} was not found to have device {device}")
    def _check_weights_shape_sane(self,
                             dictionary_name: str,
                             memory: torch.Tensor,
                             weights_dict: ImmutableDict
                             ):
        """ check that weight tensors are mostly sane"""
        batch_shape = memory.shape[:-2]
        memory_size = memory.shape[-2]

        for name, value in weights_dict.items():
            if value.shape[:-2] != torch.Size(batch_shape):
                msg = f"""\
                {dictionary_name} tensor of name {name} is not correctly shaped. Since memory
                has shape {memory.shape} it was expect to see weights with batch and ensembe shape 
                {batch_shape}. However, got {value.shape}
                """
                msg = Core.dedent(msg)
                raise ValueError(msg)
            if value.shape[-1] != memory_size:
                msg = f"""\
                {dictionary_name} tensor of name {name} is not correctly shaped. Memory
                had size {memory_size}, but dimension -1 of {name} was {value.shape[-1]}
                """
                msg = Core.dedent(msg)
                raise ValueError(msg)

    def _check_weights_defaults_shape_corrolated(self,
                                                 type: str,
                                                 weights: ImmutableDict,
                                                 defaults: ImmutableDict):
        for name in weights.keys():
            read_weight = weights[name]
            read_default = defaults[name]
            if read_default.shape != read_weight.shape[-read_default.dim():]:
                msg = f"""\
                Default shape and weights are mismatched. For the {type}, 
                the read weights had final dimensions of {read_weight.shape[-read_default.dim():]}
                while the read_defaults had shape {read_default.shape}
                """
                msg = Core.dedent(msg)
                raise ValueError(msg)


    def _perform_validation(self,
                            memory: torch.Tensor,
                            read_weights: ImmutableDict,
                            write_weights: ImmutableDict,
                            read_defaults: ImmutableDict,
                            write_defaults: ImmutableDict
                            ):
        """ Performs basic validation actions"""

        # Validate dtype and device. This is done by verifying all
        # tensors match the dtype and device of the memory tensor. Raise
        # descriptive error messages when problems occur

        dtype = memory.dtype
        device = memory.device

        self._check_dictionary_routine("read weight", dtype, device, read_weights)
        self._check_dictionary_routine("write weight", dtype, device, write_weights)
        self._check_dictionary_routine("read default", dtype, device, read_defaults)
        self._check_dictionary_routine("write default", dtype, device, write_defaults)

        # Validate that corrolations are maintained between the weights and defaults
        # dictionary. Something found in the weights dictionary better be in the
        # defaults dictionary, and vice versa

        self._check_keys_common("reader", read_weights, read_defaults)
        self._check_keys_common("writer", write_weights, write_defaults)

        # Begin validation of shapes. Ensure they are sane.
        #
        # It is important the memory and weight tensors all share batch
        # dimensions properly. It is also important that the memory
        # size be common

        self._check_weights_shape_sane("read weight", memory, read_weights)
        self._check_weights_shape_sane("write weight", memory, write_weights)

        # Finally, we need matching default shapes up to the degree that exists

        self._check_weights_defaults_shape_corrolated("read features", read_weights, read_defaults)
        self._check_weights_defaults_shape_corrolated("write features", write_weights, write_defaults)

    def __init__(self,
                 memory: torch.Tensor,
                 read_weights: ImmutableDict,
                 write_weights: ImmutableDict,
                 read_defaults: ImmutableDict,
                 write_defaults: ImmutableDict,
                 validate: bool = True,
                 ):
        """
        The initialization method for the NTM system must provide
        a variety of important features. All tensors should have the same
        dtype and same device

        :param memory: The NTM memory as currently existing. This will be a tensor of
                some sort of dtype on some device of shape
                ...batch_shape x (...ensemble_shape) x memory_size x memory_width,
                where parenthesis indicates optional, memory_size indicates number
                of memory elements, and mem_width the embedding width
        :param read_weights:
                The NTM read weights utilized by the various reader layers, and
                used where appropriate by other layers of the model. It has shape
                ...batch_shape x (...ensemble_shape) x num_heads x memory_size.
                The keys, meanwhile, are the name of that reader.

        :param write_weights:
                The NTM write weights for the various writer layers incoroporated
                in the model. The keys are the name of the writer layer, the
                values are the write weights and have shape
                ...batch_shape x (...ensemble_shape) x num_heads x memory_size.
        :param read_defaults:
                The NTM read weight defaults. The model can choose, under some
                circumstances, to reset the read weights back to their default
                values. In order to keep the parameters centralized, the read
                defaults are loaded from one layer and passed to others. This
                has shape (...ensemble_shape) x num_heads x memory_size
        :param write_defaults:
                The NTM write weight defaults. The model can choose, under
                some circumstances, to reset the write weights back to their
                default values. In order to keep the parameters centralized, the read
                defaults are loaded from one layer and passed to others. This
                has shape (...ensemble_shape) x num_heads x memory_size
        """

        if validate:
            self._perform_validation(memory,
                                     read_weights,
                                     write_weights,
                                     read_defaults,
                                     write_defaults)

        self.memory = memory
        self.read_weights = read_weights
        self.write_weights = write_weights
        self.read_defaults = read_defaults
        self.write_defaults = write_defaults

class ArithmeticBuilder:
    """
    This is a small helper class utilized to help
    easily handle arithmetic operations. It is designed
    to act as a sort of interface in which the only decision
    that a feature need to make is how to make the updated version
    """

    def get_arithmetic_ids(self, storage: Storage)->List[Tuple[str, str]]:
        """
        Creates al is
        :param storage: The storage to draw from
        :return: The walk ids for the given instance
        """
        output = [("memory", "none")]
        output = output + [("read_weights", name) for name in storage.read_weights.keys()]
        output = output + [("write_weights", name) for name in storage.write_weights.keys()]
        return output

    def get_arithmetic_tensors(self, storage: Storage)->List[torch.Tensor]:
        """
        Gets the tensors which can be modified by an arithmetic operation,
        in concert with the appropriate ids
        """
        output = [storage.memory]
        output = output + list(storage.read_weights.values())
        output = output + list(storage.write_weights.values())
        return output

    def push(self, ident: Tuple[str, str], value: torch.Tensor):
        """
        Pushes the given updated tensor to the given ident slot. Designed
        to be used with an arithmetic walker.
        :param ident: The identifier
        :param value: The tensor value
        """
        field, key = ident
        if field == "memory":
            self.memory = value
        elif field == "read_weights":
            assert value.dtype == self.storage.read_weights[key].dtype
            assert value.device == self.storage.read_weights[key].device
            assert value.shape == self.storage.read_weights[key].shape
            self.read_weights[key] = value
        elif field == "write_weights":
            assert value.dtype == self.storage.write_weights[key].dtype
            assert value.device == self.storage.write_weights[key].device
            assert value.shape == self.storage.write_weights[key].shape
            self.write_weights[key] = value
        else:
            raise ValueError("Illegal identifier specified")


    def __init__(self, storage: Storage):
        """
        Creates the setup for a arithmetic builder. This
        will start making the dictionary fields we will be utilizing
        :param storage: The storage device
        """

        self.storage = storage
        self.memory = torch.empty([0])
        self.read_weights: Dict[str, torch.Tensor] = {}
        self.write_weights: Dict[str, torch.Tensor] = {}



class StateTensor:
    """
    A small state tensor dataclass which is responsible for holding
    the NTM state information and allowing easy manipulation of it's
    contents using basic arithmetic. The arithmetic can be performed
    with respect to a numeric such as an int or float, a tensor, or
    even another state tensor of the same shape.

    --- properties and setters immutable dicts ---

    Note that the properties of this object return immutable
    dicts, which are unchangable dictionaries containing information
    of nature Dict[str, torch.Tensor]

    The properties of the StateTensor are

     * memory: The memory tensor
    * read_weights: The read weights ImmutableDict.
    * write_weights: The write weights ImmutableDict.
    * read_defaults: The read defaults ImmutableDict.
    * write_defaults: The write defaults ImmutableDict.

    The setters of imporance are

    .set_memory
    .set_weight
    .set_default
    .set_all_defaults

    --- arithmetic: introduction  ---

    The items which can be manipulated in the state tensor by basic
    arithmetic are the memory tensor, the read weights tensors, and the write
    weights tensors. These tensors can be affected by broadcast like operations

    * add
    * subtract
    * multiply
    * divide
    * modulo
    * exponentiation

    This has several usages. for example, one could do

    ```
    state_output = 0.1*state_tensor1 + 0.9*state_tensor2
    ```

    --- arithmetic between state tensors ----

    Importantly, it is the case that when manipulating state tensors the
    shapes have to match between all entries - no broadcasting is allowed there.
    Additionally, it is required that the default dictionaries and tensors be
    identical between the two features. If you need to perform arithmetic
    between state tensors with non-matching defaults, you should refresh them
    to match first.


    --- arithmetic between state tensors and tensors ---

    Arithmetic can be cleanly performed between the state tensors and the tensors,
    so long as it is the case the right justified broadcast format is followed.

    Broadcasting for state tensors occurs from left to right,  not right to left like
    with standard tensors. This means that if you are to multiply a state tensor possessing
    batch shape [5, 4] with a tensor of shape [5] or [5, 4] it would happily apply the multiplication
    against the leftmost dimensions.

    --- arithmetic between state tensors and scalars ---

    Scalars are broadcast across the entire tensor

    ---- properties ---

    A number of data properties exist. These return either
    dictionaries of tensors, or tensors themselves. Remember to
    use .clone when appropriate to avoid editing the dictionaries
    when


    """

    # Define the getters and setters for the class. The getters
    # will just be parameters, but for the setters certain functions
    # must be invoked instead.

    @property
    def device(self)->torch.device:
        return self.memory.device

    @property
    def dtype(self)->torch.dtype:
        return self.memory.dtype

    @property
    def memory(self)->torch.Tensor:
        return self.storage.memory

    @property
    def read_weights(self)->ImmutableDict:
        return self.storage.read_weights

    @property
    def write_weights(self)->ImmutableDict:
        return self.storage.write_weights

    @property
    def read_defaults(self)->ImmutableDict:
        return self.storage.read_defaults

    @property
    def write_defaults(self)->ImmutableDict:
        return self.storage.write_defaults

    def set_memory(self, value: torch.Tensor)->'StateTensor':
        """
        Sets the memory to the indicated tensor, and returns the new tensor
        """
        assert value.dtype == self.storage.memory.dtype, "new and original memory dtypes were different"
        assert value.device == self.storage.memory.device, "new and original memory device was different"
        assert value.shape == self.storage.memory.shape, "New and original memory shape were different"
        return StateTensor(
            value,
            self.storage.read_weights,
            self.storage.write_weights,
            self.storage.read_defaults,
            self.storage.write_defaults,
            validate=False
        )

    def set_weight(self, key: str, value: torch.Tensor) -> 'StateTensor':
        """
        Sets the value of the read or write weight corresponding to the given key
        to the specified value and returns a new StateTensor object with the updated
        weight. If the specified key is not found in the read or write weights,
        a KeyError is raised.

        :param key: str, the key corresponding to the read or write weight to be updated
        :param value: torch.Tensor, the new value to be set for the specified weight
        :return: StateTensor, a new StateTensor object with the updated weight
        :raises KeyError: if the specified key is not found in the read or write weights
        """
        assert value.dtype == self.storage.memory.dtype, "new and original read weights dtypes were different"
        assert value.device == self.storage.memory.device, "new and original read weight device was different"

        if key in self.storage.read_weights:
            assert value.shape == self.storage.read_weights[key].shape, "New and original read weight shape were different"
            return StateTensor(
                self.storage.memory,
                self.storage.read_weights.set(key, value),
                self.storage.write_weights,
                self.storage.read_defaults,
                self.storage.write_defaults,
                False,
            )
        elif key in self.storage.write_weights:
            assert value.shape == self.storage.write_weights[key].shape, "New and original write weight shape were different"
            return StateTensor(
                self.storage.memory,
                self.storage.read_weights,
                self.storage.write_weights.set(key, value),
                self.storage.read_defaults,
                self.storage.write_defaults,
                False
            )
        else:
            raise KeyError(f"Weight key of name {key} was found in neither the read nor write weights")

    def set_default(self, key: str, value: torch.Tensor) -> 'StateTensor':
        """
        Sets the indicated read or write defaults to the indicated value,
        then returns the new StateTensor

        :param key: The key of the default to set. Should be a string.
        :param value: The new value of the default. Should be a torch.Tensor
        :return: A new StateTensor with the indicated default updated
        """
        assert value.dtype == self.dtype, "New and original default dtypes were different"
        assert value.device == self.device, "New and original default devices were different"

        if key in self.storage.read_defaults:
            assert value.shape == self.storage.read_defaults[
                key].shape, "New and original read default shapes were different"
            return StateTensor(
                self.storage.memory,
                self.storage.read_weights,
                self.storage.write_weights,
                self.storage.read_defaults.set(key, value),
                self.storage.write_defaults,
                False
            )
        elif key in self.storage.write_defaults:
            assert value.shape == self.storage.write_defaults[
                key].shape, "New and original write default shapes were different"
            return StateTensor(
                self.storage.memory,
                self.storage.read_weights,
                self.storage.write_weights,
                self.storage.read_defaults,
                self.storage.write_defaults.set(key, value),
                False,
            )
        else:
            raise KeyError(f"Default key of name {key} was not found in either the read or write defaults")
    def _set_all_defaults_helper(self):
        pass

    def set_all_defaults(self,
                         read_defaults: Union[Dict[str, torch.Tensor], ImmutableDict],
                         write_defaults: Union[Dict[str, torch.Tensor], ImmutableDict]) -> 'StateTensor':
        """
        Sets all the read and write weight defaults to the indicated tensors,
        then returns the new StateTensor instance.
        """
        # Ignore the IDE yelling at you about the dict having parameters here.
        if not torch.jit.isinstance(read_defaults, ImmutableDict):
            read_defaults = ImmutableDict(read_defaults)
        if not torch.jit.isinstance(write_defaults,ImmutableDict):
            write_defaults = ImmutableDict(write_defaults)


        assert read_defaults.keys() == self.storage.read_defaults.keys(), "New and original read default keys were different"
        assert write_defaults.keys() == self.storage.write_defaults.keys(), "New and original write default keys were different"
        for key in read_defaults.keys():
            assert read_defaults[
                       key].dtype == self.storage.memory.dtype, "New and original read defaults dtypes were different"
            assert read_defaults[
                       key].device == self.storage.memory.device, "New and original read defaults device was different"
            assert read_defaults[key].shape == self.storage.read_defaults[
                key].shape, "New and original read default shapes were different"
        for key in write_defaults.keys():
            assert write_defaults[
                       key].dtype == self.storage.memory.dtype, "New and original write defaults dtypes were different"
            assert write_defaults[
                       key].device == self.storage.memory.device, "New and original write defaults device was different"
            assert write_defaults[key].shape == self.storage.write_defaults[
                key].shape, "New and original write default shapes were different"

        if read_defaults == self.storage.read_defaults and write_defaults == self.storage.write_defaults:
            # no need to create a new StateTensor instance if the read_defaults
            # and write_defaults are the same as the original ones
            return self

        return StateTensor(
            self.storage.memory,
            self.storage.read_weights,
            self.storage.write_weights,
            read_defaults,
            write_defaults,
            validate=False
        )
    def to(self,
           dtype: Optional[torch.dtype] = None,
           device: Optional[torch.device] = None)->'StateTensor':
        """
        Transfers the kernel dtype or device to something
        else
        :param dtype: The dtype to turn the kernels into
        :param device: The device to turn it int
        :return: A new statetenosr
        """
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device

        memory = self.memory.to(dtype=dtype, device=device)
        read_weights = {}
        write_weights =  {}
        read_defaults = {}
        write_defaults = {}

        for key, tensor in self.read_weights.items():
            read_weights[key] = tensor.to(dtype=dtype, device=device)
        for key, tensor in self.write_weights.items():
            write_weights[key] = tensor.to(dtype=dtype, device=device)
        for key, tensor in self.read_defaults.items():
            read_defaults[key] = tensor.to(dtype=dtype, device=device)
        for key, tensor in self.write_defaults.items():
            write_defaults[key] = tensor.to(dtype=dtype, device=device)
        return StateTensor(
            memory,
            read_weights,
            write_weights,
            read_defaults,
            write_defaults,
            validate=False
        )




    def __init__(self,
                 memory: torch.Tensor,
                 read_weights: Union[Dict[str, torch.Tensor], ImmutableDict],
                 write_weights: Union[Dict[str, torch.Tensor], ImmutableDict],
                 read_defaults: Union[Dict[str, torch.Tensor], ImmutableDict],
                 write_defaults: Union[Dict[str, torch.Tensor], ImmutableDict],
                 validate: Optional[bool] = None,
                 ):
        """
        The initialization method for the NTM system must provide
        a variety of important features. All tensors should have the same
        dtype and same device

        :param memory: The NTM memory as currently existing. This will be a tensor of
                some sort of dtype on some device of shape
                ...batch_shape x (...ensemble_shape) x memory_size x memory_width,
                where parenthesis indicates optional, memory_size indicates number
                of memory elements, and mem_width the embedding width
        :param read_weights:
                The NTM read weights utilized by the various reader layers, and
                used where appropriate by other layers of the model. It has shape
                ...batch_shape x (...ensemble_shape) x num_heads x memory_size.
                The keys, meanwhile, are the name of that reader.

        :param write_weights:
                The NTM write weights for the various writer layers incoroporated
                in the model. The keys are the name of the writer layer, the
                values are the write weights and have shape
                ...batch_shape x (...ensemble_shape) x num_heads x memory_size.
        :param read_defaults:
                The NTM read weight defaults. The model can choose, under some
                circumstances, to reset the read weights back to their default
                values. In order to keep the parameters centralized, the read
                defaults are loaded from one layer and passed to others. This
                has shape (...ensemble_shape) x num_heads x memory_size
        :param write_defaults:
                The NTM write weight defaults. The model can choose, under
                some circumstances, to reset the write weights back to their
                default values. In order to keep the parameters centralized, the read
                defaults are loaded from one layer and passed to others. This
                has shape (...ensemble_shape) x num_heads x memory_size
        :param validate:
                Whether or not to do validation when running the constructor. Default
                is True
        """

        if torch.jit.isinstance(read_weights, Dict[str, torch.Tensor]):
            read_weights = ImmutableDict(read_weights)
        if torch.jit.isinstance(write_weights, Dict[str, torch.Tensor]):
            write_weights = ImmutableDict(write_weights)
        if torch.jit.isinstance(read_defaults, Dict[str, torch.Tensor]):
            read_defaults = ImmutableDict(read_defaults)
        if torch.jit.isinstance(write_defaults, Dict[str, torch.Tensor]):
            write_defaults = ImmutableDict(write_defaults)
        if validate is None:
            validate = True

        self.storage = Storage(
            memory,
            read_weights,
            write_weights,
            read_defaults,
            write_defaults,
            validate
        )
    def __eq__(self, other):
        if not isinstance(other, StateTensor):
            msg = f"""\
            A problem occurred. Equality cannot be tested between state tensors
            and other objects. Please only attempt equality between state tensors
            """
        else:
            if hash(self.memory) != hash(other.memory):
                return False
            if hash(self.read_weights) != hash(other.read_weights):
                return False
            if hash(self.write_weights) != hash(other.write_weights):
                return False
            if hash(self.read_defaults) != hash(other.read_defaults):
                return False
            if hash(self.write_defaults) != hash(other.write_defaults):
                return False
            return True

    # Arithmetic validation and helper routines start here.

    def _is_reverse_broadcastable(self, tensor: torch.Tensor, shape: List[int])->bool:
        # Checks if a tensor is reverse broadcastable with a particular shape.
        #
        # A reverse broadcastable tensor has common shape or ones along the initial,
        # not final dimensions. For instance, a tensor of shape [3, 4] would be revers
        # broadcastable with another of shape 3, 4, 6, 7.
        if tensor.dim() > len(shape):
            return False
        for i, dim in enumerate(tensor.shape):
            if dim != 1 and dim != shape[i]:
                return False
        return True

    def _setup_reverse_broadcast(self, tensor: torch.Tensor, target: torch.Tensor)->torch.Tensor:
        # Sets up a tensor to be directly arithmetic capable using reverse broadcast logic.
        #
        # This occurs by simply unsqueezing the rightmost dimension of tensor until
        # it is the same rank as the target
        while tensor.dim() < target.dim():
            tensor = tensor.unsqueeze(-1)
        return tensor


    def _validate_tensor_operand(self, other: torch.Tensor):
        if other.device != self.device:
            msg = f"""\
            An issue occurred. The tensor operand was on device {other.device}.
            However, the state tensor was on device {self.device}. This
            is not allowed: Make sure they are on the same device
            """
            msg = Core.dedent(msg)
            raise ValueError(msg)
        if other.dtype != self.dtype:
            msg = f"""\
            An issue occurred. The tensor operand possessed dtype {other.dtype}.
            However, the state tensor had dtype {self.dtype}. Please ensure that
            both tensors have the same dtype.
            """
            msg = Core.dedent(msg)
            raise ValueError(msg)
        composite_batch_shape: List[int] = list(self.memory.shape[:-2])
        if not self._is_reverse_broadcastable(other, composite_batch_shape):
            if other.dim() > len(composite_batch_shape):
                msg = f"""\
                There was an issue with the tensor operand. The tensor's 
                rank was {other.dim()}, but broadcasting is only supported
                among the batch + ensemble rank, which is {len(composite_batch_shape)}
                """
                msg = Core.dedent(msg)
                raise ValueError(msg)
            msg = f"""\
            There was an issue with the tensor operand's shape. It did not successfully
            reverse broadcast with the batch and ensemble shape. The tensor's shape was
            {other.shape}, but the expected shape should broadcast with {composite_batch_shape[:other.dim()]}
            """
            msg = Core.dedent(msg)
            raise ValueError(msg)

    def _validate_state_tensor_operand(self, other: 'StateTensor'):

        if other.device != self.device:
            # State tensors must lie on the same device
            msg = f"""\
            An issue occurred. One state_tensor operand was on device {self.device}.
            However, the other state tensor was on device {other.device}. This
            is not allowed: Make sure they are on the same device
            """
            msg = Core.dedent(msg)
            raise ValueError(msg)
        if other.dtype != self.dtype:
            # State tensors must be the same datatype
            msg = f"""\
            An issue occurred. One state tensor operand possessed dtype {self.dtype}.
            However, the other state tensor had dtype {other.dtype}. Please ensure that
            both tensors have the same dtype.
            """
            msg = Core.dedent(msg)
            raise ValueError(msg)
        if self.read_weights.keys() != other.read_weights.keys():
            msg = f"""\
            An issue occurred. An attempt was made to perform arithmetic with two 
            state tensors possessing differing read keys. This is not allowed as 
            it makes no sense.
            """
            msg = Core.dedent(msg)
            raise ValueError(msg)
        if self.write_weights.keys() != other.write_weights.keys():
            msg = f"""\
            An issue occurred. An attempt was made to perform arithmetic with 
            two state tensors possessing differing write keys. This is not allowed
            as it makes no sense.
            """
            msg = Core.dedent(msg)
            raise ValueError(msg)
        # NOTE: There is no defaults key validation because they should have been
        # confirmed to be the same keys during storage setup.

        if hash(self.read_defaults) != hash(other.read_defaults):
            # Read defaults must be the same between both tensors
            msg = f"""\
            An arithmetic issue occurred. The two state tensor operands do not share
            the same read defaults values. This is not allowed when performing arithmetic.
            Use set_all_defaults to sync them to each other
            """
            msg = Core.dedent(msg)
            raise ValueError(msg)
        if hash(self.write_defaults) != hash(other.write_defaults):
            # Write defaults must be the same between both tensors
            msg = f"""\
            An arithmetic issue occurred. The two state tensor operands do not share
            the same write defaults values. This is not allowed when performing arithmetic.
            Use set_all_defaults to sync them to each other
            """
            msg = Core.dedent(msg)
            raise ValueError(msg)
        if self.memory.shape != other.memory.shape:
            # Memory shapes must be the same between the tensors. No broadcasting.
            # I might change this later, it depends on if it is useful
            msg = f"""\
            An arithmetic issue occurred. The two state tensor operands do not share
            the same memory shapes. The left hand operand had memory shape of 
            {self.memory.shape}. However, the right hand operand had shape
            of {other.memory.shape}
            """
            msg = Core.dedent(msg)
            raise ValueError(msg)
        for key_name, my_value, their_value in zip(self.read_weights.keys(),
                                                   self.read_weights.values(),
                                                   other.read_weights.values()):
            if my_value.shape != their_value.shape:
                # read weight shapes must be the same between both tensors
                msg = f"""\
                An arithmetic issue occurred. The two state tensor operands do not have a common
                shape on read key {key_name}. On the left hand operand the shape was {my_value.shape}.
                But the right hand operand had shape {their_value.shape}. This is not allowed
                """
                msg = Core.dedent(msg)
                raise ValueError(msg)
        for key_name, my_value, their_value in zip(self.write_defaults.keys(),
                                                   self.write_defaults.values(),
                                                   other.write_defaults.values()):
            # write weight shapes must be the same between both tensors
            if my_value.shape != their_value.shape:
                msg = f"""\
                An arithmetic issue occurred. The two state tensor operands do not have a common
                shape on write key {key_name}. On the left hand operand the shape was {my_value.shape}.
                But the right hand operand had shape {their_value.shape}. This is not allowed
                """
                msg = Core.dedent(msg)
                raise ValueError(msg)

    def _validate_arithmetic_operand(self, other: Union[float, int, torch.Tensor, 'StateTensor']):
        # Arithmetic operands can come in three different
        # flavors. These are scalars, tensors, or StateTensors.

        # This function validates these cases to a basic degree.

        if isinstance(other, (int, float)):
            # No additional action is required, as scalars can always broadcast onto a
            # tensor.
            return None
        elif isinstance(other, torch.Tensor):
            # Tensors must be reverse broadcastable with the batch shape
            # to be acceptable as means of modifications. They must also
            # be of similar dtype and device
            self._validate_tensor_operand(other)
        elif isinstance(other, StateTensor):
            # State tensors must have exactly the same keys and exactly the same shape. Otherwise,
            # they can not interact with each other. Obviously, they must have the same dtype and
            # device too
            self._validate_state_tensor_operand(other)
        else:
            raise ValueError("Illegal operand: Must have been StateTensor, float, int, or tensor")



    # Actual arithmetic drive mechanisms

    def __sub__(self, other):
        # Subtracting x from y is equivalent to adding -x to y
        return self.__add__(-other)

    def __rsub__(self, other):
        # Subtracting x from y is equivalent to adding -x to y
        return (-self).__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self * -1

    # Note that in following pages there is a ridiculous degree
    # of duplication. This is intentional. torchscript will
    # not allow passing around functions which makes
    # the magic methods kind of verbose.
    def __add__(self, other):

        # Validate

        self._validate_arithmetic_operand(other)

        # Make the arithmetic builder
        builder = ArithmeticBuilder(self.storage)
        idents = builder.get_arithmetic_ids(self.storage)
        current_tensors = builder.get_arithmetic_tensors(self.storage)
        if isinstance(other, (float, int)):

            # Handle making the scalar updates if that is the case
            for ident, tensor in zip(idents, current_tensors):
                update = tensor + other
                builder.push(ident, update)

        elif isinstance(other, torch.Tensor):
            # Handle doing a reverse broadcast across the cases
            # if needed here. A reverse broadcast would mean adding
            # [3, 5, 4] to [3, 5] works, and operates by unsqueezing the
            # latter to [3, 5, 1]
            for ident, tensor, in zip(idents, current_tensors):
                update = tensor + self._setup_reverse_broadcast(other, tensor)
                builder.push(ident, update)
        elif isinstance(other, StateTensor):
            # Handle doing arithmetic in the case of comparing state tensors
            #
            # Due to their meaning, state tensors can only be added together
            # if their keys and shapes are the same, and if their defaults
            # tensors are the same.

            other_tensors = builder.get_arithmetic_tensors(other.storage)
            for ident, my_tensor, their_tensor in zip(idents, current_tensors, other_tensors):
                update = my_tensor + their_tensor
                builder.push(ident, update)
        else:
            raise ValueError("Not adding a tensor, a StateTensor, or a scalar")

        # Finish up

        return StateTensor(
            builder.memory,
            builder.read_weights,
            builder.write_weights,
            self.read_defaults,
            self.write_defaults,
            validate=False
        )
    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other: Union[float, int, torch.Tensor, 'StateTensor']):

        # Validate

        self._validate_arithmetic_operand(other)

        # Make the arithmetic builder
        builder = ArithmeticBuilder(self.storage)
        idents = builder.get_arithmetic_ids(self.storage)
        current_tensors = builder.get_arithmetic_tensors(self.storage)
        if isinstance(other, (float, int)):
            # Handle making the scalar updates if that is the case
            for ident, tensor in zip(idents, current_tensors):
                update = tensor * other
                builder.push(ident, update)
        elif isinstance(other, torch.Tensor):
            # Handle doing a reverse broadcast across the cases
            # if needed here. A reverse broadcast would mean multiplying
            # [3, 5, 4] with [3, 5] works, and operates by unsqueezing the
            # latter to [3, 5, 1]
            for ident, tensor, in zip(idents, current_tensors):
                update = tensor * self._setup_reverse_broadcast(other, tensor)
                builder.push(ident, update)
        elif isinstance(other, StateTensor):
            # Handle doing arithmetic in the case of comparing state tensors
            #
            # Due to their meaning, state tensors can only be added together
            # if their keys and shapes are the same, and if their defaults
            # tensors are the same.

            other_tensors = builder.get_arithmetic_tensors(other.storage)
            for ident, my_tensor, their_tensor in zip(idents, current_tensors, other_tensors):
                update = my_tensor * their_tensor
                builder.push(ident, update)
        else:
            raise ValueError("Not multiplying a tensor, a StateTensor, or a scalar")

        # Finish up
        return StateTensor(
            builder.memory,
            builder.read_weights,
            builder.write_weights,
            self.read_defaults,
            self.write_defaults,
            validate=False
        )

    def __rtruediv__(self, other):

        # Validate

        self._validate_arithmetic_operand(other)

        # Make the arithmetic builder
        builder = ArithmeticBuilder(self.storage)
        idents = builder.get_arithmetic_ids(self.storage)
        current_tensors = builder.get_arithmetic_tensors(self.storage)

        if isinstance(other, (float, int)):
            # Handle making the scalar updates if that is the case
            for ident, tensor in zip(idents, current_tensors):
                update =  other / tensor
                builder.push(ident, update)
        elif isinstance(other, torch.Tensor):
            # Handle doing a reverse broadcast across the cases
            # if needed here. A reverse broadcast would mean dividing
            # [3, 5, 4] by [3, 5] works, and operates by unsqueezing the
            # latter to [3, 5, 1]
            for ident, tensor, in zip(idents, current_tensors):
                update = self._setup_reverse_broadcast(other, tensor) / tensor
                builder.push(ident, update)
        elif isinstance(other, StateTensor):
            # Handle doing arithmetic in the case of comparing state tensors
            #
            # Due to their meaning, state tensors can only be divided together
            # if their keys and shapes are the same, and if their defaults
            # tensors are the same.
            other_tensors = builder.get_arithmetic_tensors(other.storage)
            for ident, my_tensor, their_tensor in zip(idents, current_tensors, other_tensors):
                update = their_tensor / my_tensor
                builder.push(ident, update)
        else:
            raise ValueError("Not dividing a tensor, a StateTensor, or a scalar")

        # Finish up
        return StateTensor(
            builder.memory,
            builder.read_weights,
            builder.write_weights,
            self.read_defaults,
            self.write_defaults,
            validate=False
        )

    def __truediv__(self, other):

        # Validate

        self._validate_arithmetic_operand(other)

        # Make the arithmetic builder
        builder = ArithmeticBuilder(self.storage)
        idents = builder.get_arithmetic_ids(self.storage)
        current_tensors = builder.get_arithmetic_tensors(self.storage)

        if isinstance(other, (float, int)):
            # Handle making the scalar updates if that is the case
            for ident, tensor in zip(idents, current_tensors):
                update = tensor / other
                builder.push(ident, update)
        elif isinstance(other, torch.Tensor):
            # Handle doing a reverse broadcast across the cases
            # if needed here. A reverse broadcast would mean dividing
            # [3, 5, 4] by [3, 5] works, and operates by unsqueezing the
            # latter to [3, 5, 1]
            for ident, tensor, in zip(idents, current_tensors):
                update = tensor / self._setup_reverse_broadcast(other, tensor)
                builder.push(ident, update)
        elif isinstance(other, StateTensor):
            # Handle doing arithmetic in the case of comparing state tensors
            #
            # Due to their meaning, state tensors can only be divided together
            # if their keys and shapes are the same, and if their defaults
            # tensors are the same.
            other_tensors = builder.get_arithmetic_tensors(other.storage)
            for ident, my_tensor, their_tensor in zip(idents, current_tensors, other_tensors):
                update = my_tensor / their_tensor
                builder.push(ident, update)
        else:
            raise ValueError("Not dividing a tensor, a StateTensor, or a scalar")

        # Finish up
        return StateTensor(
            builder.memory,
            builder.read_weights,
            builder.write_weights,
            self.read_defaults,
            self.write_defaults,
            validate=False
        )
    def __pow__(self, other, modulo=None):

        # Validate

        self._validate_arithmetic_operand(other)

        # Make the arithmetic builder
        builder = ArithmeticBuilder(self.storage)
        idents = builder.get_arithmetic_ids(self.storage)
        current_tensors = builder.get_arithmetic_tensors(self.storage)

        if isinstance(other, (float, int)):
            # Handle making the scalar updates if that is the case
            for ident, tensor in zip(idents, current_tensors):
                update = tensor ** other
                builder.push(ident, update)
        elif isinstance(other, torch.Tensor):
            # Handle doing a reverse broadcast across the cases
            # if needed here. A reverse broadcast would mean dividing
            # [3, 5, 4] by [3, 5] works, and operates by unsqueezing the
            # latter to [3, 5, 1]
            for ident, tensor, in zip(idents, current_tensors):
                update = tensor ** self._setup_reverse_broadcast(other, tensor)
                builder.push(ident, update)
        elif isinstance(other, StateTensor):
            # Handle doing arithmetic in the case of comparing state tensors
            #
            # Due to their meaning, state tensors can only be divided together
            # if their keys and shapes are the same, and if their defaults
            # tensors are the same.
            other_tensors = builder.get_arithmetic_tensors(other.storage)
            for ident, my_tensor, their_tensor in zip(idents, current_tensors, other_tensors):
                update = my_tensor / their_tensor
                builder.push(ident, update)
        else:
            raise ValueError("Not dividing a tensor, a StateTensor, or a scalar")

        # Finish up
        return StateTensor(
            builder.memory,
            builder.read_weights,
            builder.write_weights,
            self.read_defaults,
            self.write_defaults,
            validate=False
        )

torch.jit.script(StateTensor)


