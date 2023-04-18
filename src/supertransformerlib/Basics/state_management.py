import torch
from torch import nn
from supertransformerlib import Core
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


class SetupSublayer(nn.Module, ABC):
    """
    The SetupSublayer is an interface for layers responsible for setting up a
    usable tensor. It inherits from nn.Module.

    This layer is an interface without logic but with prebuilt type requirements
    and output requirements. The forward parameters are prebuilt to consist of a section
    "batch_shape" and "tensors" which can be used when developing a model.

    "batch_shape" : A List of ints representing the required shape of the first few output dimensions.
                    Broadcasting is expected to be used to match this.
    "tensors": Dict[str, torch.Tensor]. This may contain extra information, such as content
                inputs used to build a correct tensor.

    """
    @abstractmethod
    def forward(self,
                batch_shape: List[int],
                tensors: Dict[str, torch.Tensor]
                ) -> torch.Tensor:
        msg = "SetupSublayer is an interface and should be subclassed with a specific implementation."
        raise NotImplementedError(msg)


@dataclass
class _SetupFrame:
    """
    The SetupFrame is a dataclass object designed to contain the information
    needed to integrate a particular SetupSublayer into a broader setup mechanism. It is an
    internal helper mechanism.

    In particular, it is an object containing:
    * layer: The SetupSublayer layer
    * required_tensors: A List[str] of names of required tensors. These will be fed in as a dict
    * constraints: A List[str] of the constraints to attach to the constructed result

    This is just used to contain information within an object conveniently.
    """
    layer: SetupSublayer
    required_tensors: List[str]
    constraints: List[str]

@dataclass
class SetupConfig:
    """
    A SetupConfig object should be instanced once before beginning to build any utility layers.
    It is a mutable object whose purpose is to make and hold SetupFrames inside an internal dictionary.

    This class provides the method .add which accepts the name of a state to ultimately generate,
    the layer to make that state, the required tensors as a List[str], and the ultimate constraints
    as a List[str].
    """
    batch_dims: int
    frames: Dict[str, _SetupFrame] = field(default_factory=dict)

    def add(self, name: str, layer: SetupSublayer, required_tensors: List[str], constraints: List[str]) -> None:
        """
        Adds a new SetupFrame to the internal dictionary of frames.

        Args:
            name (str): The name of the state to ultimately generate.
            layer (nn.Module): The layer to make that state.
            required_tensors (List[str]): The list of required tensor names.
            constraints (List[str]): The list of constraints for the generated state.

        Raises:
            ValueError: If an entry with the same name already exists.
        """
        if name in self.frames:
            raise ValueError(f"An entry with the name '{name}' already exists.")

        self.frames[name] = _SetupFrame(layer=layer, required_tensors=required_tensors, constraints=constraints)

@dataclass
class _StringListHolder:
    """
    This is a small helper class. It exists explictly as a result of
    stupid torchscript behavior.

    It is the case that torchscript will not properly handle the situation
    of having a Dict[str, List[str]] entry placed as a field in the constructor
    of a nn.Module and used during forward. Instead, it only allows one level
    of data storage: Dict[str, !!Something!!].

    To get around this, we define StringListHolder to hold string lists, then
    store on our fields by Dict[str, StringListHolder].

     It is required due to torchscript
    not explictly supporting Dict[str, List[str]], that is, supporting
    nesting. Instead, we make a small dataclass and store that.
    """
    string_list: List[str]

torch.jit.script(_StringListHolder)

class Setup(nn.Module):
    """
    The Setup layer is expected to be provided a SetupConfig object on construction,
    and whatever additional tensors are required.

    The forward method, when called, will require the batch shape and optionally
    a dictionary of additional entries which may contain extra tensor information.

    It will use the stored configuration in order to setup the state's default BundleTensor
    """
    def __init__(self, config: SetupConfig):
        """
        :param config: The config option
        """
        super().__init__()

        names: List[str] = []
        layers: Dict[str, SetupSublayer] = {}
        required_tensors: Dict[str, _StringListHolder] = {}
        constraints: Dict[str, _StringListHolder] = {}

        for key in config.frames.keys():
            names.append(key)

            frame = config.frames[key]
            layers[key] = frame.layer
            required_tensors[key] = _StringListHolder(frame.required_tensors)
            constraints[key] = _StringListHolder(frame.constraints)

        self.batch_dims = config.batch_dims
        self.names = names
        self.layers = nn.ModuleDict(layers)
        self.requirements = required_tensors
        self.constraints = constraints

    def forward(self,
                batch_shape: Core.StandardShapeType,
                tensors: Optional[Dict[str, torch.Tensor]] = None)->Core.BundleTensor:
        """
        :param batch_shape: The expected shape of the batch. Required for broadcasting purposes. Must
                            match length of batch config.
        :param tensors: A dictionary containing the external tensor information needed for setup. Optional
        :return: A BundleTensor setup to handle state operations.
        """

        if tensors is None:
            tensors = {}


        bundle_tensors: Dict[str, torch.Tensor] = {}
        bundle_constraints: Dict[str, List[str]] = {}
        batch_shape_list: List[int] = Core.standardize_shape(batch_shape, "batch_shape").tolist()
        assert len(batch_shape_list) == self.batch_dims

        for key, layer in self.layers.items():

            # Go make the new tensor, whatever it might be
            needed_tensor_names = self.requirements[key]
            needed_tensors = {name : tensors[name] for name in needed_tensor_names.string_list}
            initialized_tensor = layer(batch_shape_list, needed_tensors)

            # Go store away the relevant info
            bundle_tensors[key] = initialized_tensor
            bundle_constraints[key] = self.constraints[key].string_list

        output = Core.BundleTensor(self.batch_dims,
                                   bundle_tensors,
                                   bundle_constraints)
        return output

class StateUsageLayer(nn.Module, ABC):
    """
    The state usage layer is an interface layer designed to
    be subclassed which defines portions of the interface
    you are expected to use when using the state management
    setup manager
    """
    @abstractmethod
    def __init__(self, config: SetupConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
