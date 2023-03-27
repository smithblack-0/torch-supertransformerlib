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
    NTM state_tensor datastructure. This is performed by using the prior read
    weights, the control state, and the current memory. Optionally,
    the read head may also decide to reset read weights back to their
    default values per head.

    Multiple read heads are contained within the structure and all
    work alongside the input to handle the particular problem being
    worked upon.

    The output of the reader will be the read result and the updated
    state tensor.
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
                 ensemble_shape: Optional[Core.StandardShapeType] = None,
                 allow_resetting_heads: Optional[bool] = False,
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
        :param allow_resetting_heads: Whether or not to allow resetting read head read weights to
                                     their default values.
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

        # We use the ensemble mode of linear layers to ensure each head is independent.
        #
        # This is done by adding an extra ensemble dimension.
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

        if allow_resetting_heads:
            self.make_reset_logit = Basics.Linear(self.create_heads.head_width,
                                                  1,
                                                  ensemble_shape
                                                  )
        else:
            self.make_reset_logit = None



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
        memory = memory.unsqueeze(-3)

        # Reset the read weights, per head, if considered appropriate

        if self.make_reset_logit is not None:
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
    A collection of write heads designed to put information into an
    NTM state_tensor datastructure. This is performed by using the prior write
    weights, the control state, and the current memory. Optionally,
    the write head may also decide to reset write weights back to their
    default values per head.

    Multiple write heads are contained within the structure and all
    work alongside the input to handle the particular problem being
    worked upon.

    The output of the writer will be the updated state tensor
    """
    def __init__(self,
                 writer_name: str,
                 memory_size: int,
                 memory_width: int,
                 num_heads: int,
                 control_width: int,
                 data_width: int,
                 shift_kernel_width: int,
                 head_creation_mode: Optional[str] = "project",
                 allow_resetting_heads: Optional[bool] = False,
                 ensemble_shape: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        :param writer_name: A string indicating what writer this is.
        :param memory_size: The size of the auxilary memory in terms of memory units
        :param memory_width: The width of the memory embeddings
        :param num_heads: The number of heads to make
        :param control_width: The width of the control tensor embedding. This must contain both
                              the control information and the memory info to store, so usually
                              should be bigger than memory width
        :param data_widht: The width of the data tensor embedding. This must contain the
                            data we wish to insert into the memory.
        :param shift_kernel_width: How wide to make the shift kernel. Notably,
                                 shifts are defined to be symmetric around 0
        :param ensemble_shape: The shape of any ensemble action which is occuring
        :param head_creation_mode: The control tensor can have heads placed on it by two
                                    means. We could either do a linear projection, or a reshape
                                    action. This decides which to use. The two modes are
                                    "reshape" or "project".
        :param allow_resetting_heads: Whether or not to allow resetting read head write weights to
                                     their default values.
        :param dtype: The dtype of the kernels
        :param device: The device of the kernels
        """

        # Store natural parameters.

        self.name = writer_name
        self.memory_size = memory_size
        self.memory_width = memory_width
        self.num_heads = num_heads
        self.control_width = control_width
        self.data_width = data_width
        self.shift_kernel_width = shift_kernel_width
        self.ensemble_shape = ensemble_shape
        self.head_creation_mode = head_creation_mode

        super().__init__()



        # We need to make the layers responsible for
        # creating heads. Heads will need to
        # be created on the control state tensor.

        self.create_control_heads = Basics.MakeHead(control_width,
                                            num_heads,
                                            mode=head_creation_mode,
                                            parallel=ensemble_shape,
                                            dtype=dtype,
                                            device=device)
        self.create_data_heads = Basics.MakeHead(data_width,
                                                 num_heads,
                                                 mode=head_creation_mode,
                                                 parallel=ensemble_shape,
                                                 dtype=dtype,
                                                 device=device)



        # Define the NTM indexing mechanism. Notice how we are expecting to
        # recieve control tensors of head width, not control width. Also, note
        # that the ensemble content must have the head width added to it to
        # create the proper parallel kernels. This ensures the index generation
        # is entirely indepedent between heads: There is no reuse of parameters as in
        # a convolution.

        # We use the ensemble mode of linear layers to ensure each head is independent.
        #
        # This is done by adding an extra ensemble dimension, which is performed
        # by concatenating that dimension into the ensemble shape
        if ensemble_shape is not None:
            ensemble_shape = Core.standardize_shape(ensemble_shape, "ensemble_shape")
            ensemble_shape = torch.concat([ensemble_shape, torch.tensor([num_heads])], dim=-1)
        else:
            ensemble_shape = torch.tensor([num_heads])

        # Notice how each head is independently evaluated now using their
        # own parameters

        self.make_write_weights = Indexer(
                                    memory_size,
                                    memory_width,
                                    self.create_control_heads.head_width,
                                    shift_kernel_width,
                                    ensemble_shape,
                                    dtype=dtype,
                                    device=device)

        # Writing requires more parameters than reading. In particular,
        # we require the erase tensor and the update tensor as well.
        #
        # Those will be made directly from the unheaded control state
        # tensors and will possess the required heads themselves

        self.make_erase_logit = Basics.Linear(self.create_control_heads.head_width,
                                              [1, memory_width],
                                              ensemble_shape,
                                              dtype=dtype,
                                              device=device)
        self.make_write_feature = Basics.Linear(self.create_data_heads.head_width,
                                                [1, memory_width],
                                                ensemble_shape,
                                                dtype=dtype,
                                                device=device)


        # If the head resetting mechanism is active, we will also
        # need to get a projection to handle that.

        if allow_resetting_heads:
            self.make_reset_logit = Basics.Linear(self.create_heads.head_width,
                                                  1,
                                                  ensemble_shape,
                                                  dtype=dtype,
                                                  device=device
                                                  )
        else:
            self.make_reset_logit = None



    def forward(self,
                control_state: torch.Tensor,
                data: torch.Tensor,
                state_tensor: StateTensor)->StateTensor:

        """
        Performs the write operation

        :param control_state: A ... x control_width tensor used to control actions
        :param control_state: A ... x data_width tensor containing the data to store
        :param state_tensor: The state tensor object for the current NTM action.
        :return:
            A ...  x mem_width tensor of memory read output
            A StateTensor that has had one of it's weights entries updated.
        """

        # Fetch the relevant information out of the state tensor

        memory = state_tensor.memory
        weights = state_tensor.write_weights[self.name]
        defaults = state_tensor.write_defaults[self.name]

        # Create important write parameters from the control state


        # Create head states on the control tensor, and on the memory tensor
        #
        # This is needed to allow indexer to work independently across the heads.

        control_state = self.create_control_heads(control_state.unsqueeze(-2)).squeeze(-2)
        data = self.create_data_heads(data.unsqueeze(-2)).squeeze(-2)
        memory = memory.unsqueeze(-3)

        # Create the erase and write parameters out of the headed control state

        erase_weights = torch.sigmoid(self.make_erase_logit(control_state))
        write_features = self.make_write_feature(data)

        # Reset the read weights, per head, if considered appropriate

        if self.make_reset_logit is not None:
            reset_probabilities = torch.sigmoid(self.make_reset_logit(control_state))
            weights = weights *(1 - reset_probabilities) + defaults * reset_probabilities


        # Make the new write weights
        weights = self.make_write_weights(control_state,
                                         memory,
                                         weights
                                         )
        # Update each component of the memory heads. Reduce the heads down. Combine

        retained_memory = memory*(1-erase_weights*weights.unsqueeze(-1))
        retained_memory = retained_memory.prod(dim=-3)

        added_memory = write_features*weights.unsqueeze(-1)
        added_memory = added_memory.sum(dim=-3)

        memory = retained_memory + added_memory

        # Store results and return

        state_tensor = state_tensor.set_weight(self.name, weights)
        state_tensor = state_tensor.set_memory(memory)
        return state_tensor

class Refresher(nn.Module):
    """
    The NTM refresher is the location where parameters which may need to be
    syncronized together, such as default parameters and default memory, live. The NTM instance
    should call the refresher with the state tensor at least once per batch in order to ensure
    the parameters are fresh within the graph.

    It also contains mechanisms for creating an empty batch, resetting a batch
    back towards defaults, and even allowing the model to decide when to reset
    a batch.
    """
class NTM_Builder:
    """
    The NTM builder is responsible, perhaps unsurprisingly, for building a
    working NTM stetup. It is capable of building a variable number of read,
    write, and reset heads and acting appropriately when such a head is
    encountered.

    It can be utilized to build the required nn.Module layers, and then
    the .finalize method can be used to finish the process and return the
    final layer.

    The controller is not built by this mechanism.

    --- methods ---

    .make_controller
    .make_reader
    .make_writer
    .finalize()
    """


    def make_reader(self,
                    num_heads: int,
                    shift_kernel_width: int,
                    head_creation_mode: Optional[str] = "project",
                    head_merge_mode: Optional[str] = "weighted_sum",
                    reset_heads: Optional[bool] = False,
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
        :param reset_heads: Whether or not to allow the reader to reset it's own read weights when
                            so desired.
        :return: A reader layer, which can be utilized with a control state and a state dictionary to perform
                a NTM read.
        """

        reader_name = "reader_" + str(self.reader_count)
        self.reader_names.append(reader_name)
        self.reader_count += 1
        return Reader(
            reader_name,
            self.memory_size,
            self.memory_width,
            num_heads,
            self.control_width,
            shift_kernel_width,
            head_creation_mode,
            head_merge_mode,
            self.ensemble_shape,
            reset_heads,
            self.dtype,
            self.device
        )

    def make_writer(self,
                    shift_kernel_width: int,
                    data_width: int,


                    ):
        """
        Creates a writer layer which can be utilized in the NTM process.

        :param num_heads: The number of heads to make
        :param control_width: The width of the control tensor embedding
        :param shift_kernel_width: How wide to make the shift kernel. Notably,
                                 shifts are defined to be symmetric around 0
        :param head_creation_mode: The control tensor can have heads placed on it by two
                                    means. We could either do a linear projection, or a reshape
                                    action. This decides which to use. The two modes are
                                    "reshape" or "project".
        :param allow_resetting_heads: Whether or not to allow resetting read head write weights to
                                     their default values.
        """
        writer_name = "writer_" + str(self.writer_count)
        self.writer_names.append(writer_name)
        self.writer_count += 1
        return Writer(
            writer_name,


        )

    def __init__(self,
                 memory_size: int,
                 memory_width: int,
                 control_width: int,
                 data_width: Optional[int] = None,
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

        self.reader_names = []
        self.writer_names = []

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