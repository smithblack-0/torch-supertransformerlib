from typing import Optional, Tuple, List, Dict

import torch
from torch import nn

from supertransformerlib import Basics
from supertransformerlib import Core
from supertransformerlib.NTM.indexer import Indexer
from supertransformerlib.NTM.state_utilities import StateTensor

class Reader(nn.Module):
    """
    A collection of read heads designed to fetch information out of a
    NTM datastructure. This is performed by using the prior read
    weights, the control state, and the current memory.

    Multiple read heads are contained within the structure and all
    work alongside the input to handle the particular problem being
    worked upon.

    The output of the reader will be the read result and the
    updated reader_weights
    """

    def __init__(self,
                 reader_name: str,
                 memory_size: int,
                 memory_width: int,
                 num_heads: int,
                 control_width: int,
                 shift_kernel_width: int,
                 head_creation_mode: Optional[str] = "project",
                 head_merge_mode: Optional[str] = "weighted_sum",
                 allow_reset_weights: Optional[bool] = True,
                 ensemble_shape: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        :param reader_name: A string indicating what reader this is.
        :param memory_size: The size of the auxilary memory in terms of memory units
        :param memory_width: The width of the memory embeddings
        :param num_heads: The number of heads to make
        :param control_width: The width of the control tensor embedding
        :param shift_kernel_width: How wide to make the shift kernel. Notably,
                                 shifts are defined to be symmetric around 0
        :param ensemble_shape: The shape of any ensemble action which is occuring
        :param head_creation_mode: The control tensor can have heads placed on it by two
                                    means. We could either do a linear projection, or a reshape
                                    action. This decides which to use. The two modes are
                                    "reshape" or "project".
        :param head_merge_mode: The read tensors will consist of a weighted sum of embeddings with
                                one per head. These have to be merged together before the results
                                are returned. Two options exist for this. They are "project" which
                                uses a linear projection, "weight" which makes a weighted
                                sum between each head, and "sum" which simply sums across the
                                appropriate dimension. These each require progressively decreasing
                                numbers of parameters, and the default is weight
        :param dtype: The dtype of the kernels
        :param device: The device of the kernels
        """

        # Store natural parameters.

        self.name = reader_name
        self.memory_size = memory_size
        self.memory_width = memory_width
        self.num_heads = num_heads
        self.control_width = control_width
        self.shift_kernel_width = shift_kernel_width
        self.ensemble_shape = ensemble_shape
        self.head_creation_mode = head_creation_mode

        super().__init__()



        # We need to make the layers responsible for
        # creating and merging heads. Heads will need to
        # be created on the control stte tensor, and when
        # merging we will be merging the memory tensor stack

        self.create_heads = Basics.MakeHead(control_width, num_heads,
                                            mode=head_creation_mode,
                                            parallel=ensemble_shape,
                                            dtype=dtype,
                                            device=device)
        self.merge_heads = Basics.ReductiveMergeHeads(memory_width,
                                                      num_heads,
                                                      mode=head_merge_mode,
                                                      parallel=ensemble_shape,
                                                      dtype=dtype,
                                                      device=device)

        # Define the NTM indexing mechanism. Notice how we are expecting to
        # recieve control tensors of head width, not control width. Also, note
        # that the ensemble content must have the head width added to it to
        # create the proper parallel kernels. This ensures the index generation
        # is entirely indepedent between heads: There is no reuse of parameters as in
        # a convolution.

        if ensemble_shape is not None:
            ensemble_shape = Core.standardize_shape(ensemble_shape, "ensemble_shape")
            ensemble_shape = torch.concat([ensemble_shape, torch.tensor([num_heads])], dim=-1)
        else:
            ensemble_shape = torch.tensor([num_heads])

        self.make_read_weights = Indexer(
                                    memory_size,
                                    memory_width,
                                    self.create_heads.head_width,
                                    shift_kernel_width,
                                    ensemble_shape,
                                    dtype=dtype,
                                    device=device)

        # If the head resetting mechanism is active, we will also
        # need to get a projection to handle that.

        if allow_reset_weights:
            self.make_reset_logit = Basics.Linear(self.create_heads.head_width,
                                                  1,
                                                  ensemble_shape
                                                  )
        else:
            self.make_reset_probabilities = None



    def forward(self,
                control_state: torch.Tensor,
                state_tensor: StateTensor)->Tuple[torch.Tensor, StateTensor]:

        """
        Performs the read operation

        :param control_state: A ... x control_width tensor used to control actions
        :param state_tensor: The state tensor object for the current NTM action.
        :return:
            A ...  x mem_width tensor of memory read output
            A StateTensor that has had one of it's weights entries updated.
        """

        # Fetch the relevant information out of the state tensor

        memory = state_tensor.memory
        weights = state_tensor.read_weights[self.name]
        defaults = state_tensor.read_defaults[self.name]

        # Create head states on the control tensor, and on the memory tensor

        control_state = self.create_heads(control_state.unsqueeze(-2)).squeeze(-2)
        memory = state_tensor.memory
        memory = memory.unsqueeze(-3)

        # Reset the read weights, per head, if considered appropriate

        if self.make_reset_probabilities is not None:
            reset_probabilities = torch.sigmoid(self.make_reset_logit(control_state))
            weights = weights *(1 - reset_probabilities) + defaults * reset_probabilities


        # Make the new read weights
        weights = self.make_read_weights(control_state,
                                         memory,
                                         weights
                                         )
        # Do the read, merge the heads
        headed_output = torch.matmul(weights.unsqueeze(-2), memory)
        output = self.merge_heads(headed_output).squeeze(-2)

        # Commit state then return results
        state_tensor = state_tensor.set_weight(self.name, weights)
        return output, state_tensor

class Writer(nn.Module):
    """
    A collection of write heads designed to insert information back
    into an NTM memory datastructure, while recursively keeping
    in mind the prior actions.
    NTM datastructure. This is performed by using the prior read
    weights, the control state, and the current memory.

    Multiple read heads are contained within the structure and all
    work alongside the input to handle the particular problem being
    worked upon.

    The output of the reader will be the read result and the
    updated reader_weights
    """

class NTM_Builder:
    """
    The NTM builder is responsible, perhaps unsuprisingly, for building a
    working NTM stetup. It is capable of building a variable number of read,
    write, and reset heads and acting appropriately when such a head is
    encountered.

    It can be utilized to build the required nn.Module layers, and then
    the .finalize method can be used to finish the process and return the
    final layer.

    The controller is not built by this mechanism.

    --- methods ---

    .make_reader
    .make_writer
    .finalize()
    """
    # The challenges that need to be met in order for the NTM
    # layer to function consistently are primarily challenges
    # involving where the default parameters may lie.

    # Our goal in the builder are as follows
    #   * Create the default parameters to be loaded into the master NTM layer later
    #   * Make reader layers, and whatever defaults are needed
    #   * Make writer layers, and whatever defaults are needed
    #
    # Finally, at the end
    #   * Make a master layer which holds the default parameters, loads them into
    #   a dictionary, and otherwise is really useful for managing state.

    # The whole point of all this is to end up with a single dictionary of state information
    # which all the layers agree on regarding what corrolates with what.

    def make_reader(self,
                    num_heads: int,
                    shift_kernel_width: int,
                    head_creation_mode: Optional[str] = "project",
                    head_merge_mode: Optional[str] = "weighted_sum",
                    reset_mode: Optional[str] = "None"
                    )->Reader:
        """
        This will create a reader, and stash away the defaults for
        later usage. See layer "Reader" for more information

        :param num_heads: The number of heads the reader should use
        :param shift_kernel_width: The width of the shift kernel. Wider will allow more drastic jumps
        :param head_creation_mode: How to convert the control state into control head. The modes are
                                   "reshape" and "project"
        :param head_merge_mode: How to merge the read heads bock together. Whatever the mode, the
                                return will be of memory width. The three modes are "sum", "weighted_sum",
                                and "project".
        :param reset_mode: Whether and to what degree
        :return: A reader layer, which can be utilized with a control state and a state dictionary to perform
                a NTM read.
        """



    def __init__(self,
                 memory_size: int,
                 memory_width: int,
                 control_width: int,
                 ensemble_shape: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        :param memory_size: The number of memory elements we will have
        :param memory_width: The width of the memory embeddings
        :param control_width: The width of the control state embeddings
        :param ensemble_shape: The ensemble dimensions if they exist
        :param dtype: The dtype if they exist
        :param device: The device if they exist.
        """

        self.memory_size = memory_size
        self.memory_width = memory_width
        self.control_width = control_width
        self.ensemble_shape = ensemble_shape
        self.dtype = dtype
        self.device = device

        # Create memory parameter
        memory_shape = [memory_size, memory_width]
        if ensemble_shape is not None:
            ensemble_shape: List[int] = Core.standardize_shape(ensemble_shape, "ensemble_shape").tolist()
            memory_shape = ensemble_shape + memory_shape

        memory_parameter = torch.zeros(memory_shape, dtype=dtype, device=device)
        torch.nn.init.kaiming_uniform_(memory_parameter)
        memory_parameter = nn.Parameter(memory_parameter)
        self.memory_default = memory_parameter

        # Create storage locations for read and write parameters. Also, create the
        # read and write head counts.

        self.reader_count = 0
        self.writer_count = 0

        self.reader_defaults: Dict[str, nn.Parameter] = {}
        self.writer_defaults: Dict[str, nn.Parameter] = {}

class NTM(nn.Module):
    """
    The NTM layer is responsible for two distinct actions.

    --- setup ----

    The NTM layer is responsible for creating demanded read, write, and reset layers
    when so demanded. Read and write layers may be created straightforwardly enough
    by calling the appropriate methods.

    .make_reader
    .make_writer

    Meanwhile, a resetter layer can be made in one of several manners

    .make_syncronous_resetter
    .make_weight_resetter

    Once all construction is

    --- defaults_creation ---

    The layer is capable of setting up a default block of state parameters
    compatible with a batch. This is done with the

    .make_batch function

    """