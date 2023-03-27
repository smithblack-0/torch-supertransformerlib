import unittest
import torch
import itertools
from supertransformerlib.NTM.ntm import Reader, Writer
from supertransformerlib.NTM.state_utilities import StateTensor


class TestReader(unittest.TestCase):
    def setUp(self):
        self.memory_size = 10
        self.memory_width = 20
        self.num_heads = 4
        self.control_width = 5
        self.shift_kernel_width = 3
        self.batch_shape = (7, 8)
        self.ensemble_shape = (3,)
        self.dtype = torch.float32
        self.device = torch.device('cpu')

    def test_modes_basic(self):
        """
         Test all the various head modes in a simple
        batched configuration

        """
        # Define test parameters
        name = "reader1"
        batch_size = 24
        memory_size = 10
        memory_width = 21
        num_heads = 4
        control_width = 20
        shift_kernel_width = 3
        creation_modes = ["reshape", "project"]
        merge_mode = ["sum", "weighted_sum", "project"]

        # Define the test inputs
        control_state = torch.randn(batch_size, control_width)

        memory = torch.randn(batch_size, memory_size, memory_width)
        weights = torch.softmax(torch.randn(batch_size, num_heads, memory_size), dim=-1)
        defaults = torch.softmax(torch.randn(num_heads, memory_size), dim=-1)

        read_weights = {"reader1": weights}
        write_weights = {}
        read_defaults = {"reader1": defaults}
        write_defaults = {}
        state_tensor = StateTensor(memory,
                                   read_weights,
                                   write_weights,
                                   read_defaults,
                                   write_defaults)

        # perform test cases
        for creation_mode, merge_mode in itertools.product(creation_modes, merge_mode):

            # Define reader. Then implement read

            reader = Reader(name,
                            memory_size,
                            memory_width,
                            num_heads,
                            control_width,
                            shift_kernel_width,
                            creation_mode,
                            merge_mode)

            read_results, new_state = reader(control_state, state_tensor)

            self.assertTrue(read_results.shape == (batch_size, memory_width))
            self.assertTrue(new_state.read_weights[name].shape == weights.shape)
            self.assertTrue(read_results.dtype == torch.float32)

    def test_in_ensemble(self):
        """ Test the various modes still work on ensembles of data"""
        # Define test parameters
        name = "reader1"
        batch_size = 24
        ensemble_size = 17
        memory_size = 10
        memory_width = 21
        num_heads = 4
        control_width = 20
        shift_kernel_width = 3
        creation_modes = ["reshape", "project"]
        merge_mode = ["sum", "weighted_sum", "project"]

        # Define the test inputs
        control_state = torch.randn(batch_size, ensemble_size, control_width)

        memory = torch.randn(batch_size,ensemble_size, memory_size, memory_width)
        weights = torch.softmax(torch.randn(batch_size, ensemble_size,  num_heads, memory_size), dim=-1)
        defaults = torch.softmax(torch.randn(ensemble_size, num_heads, memory_size), dim=-1)

        read_weights = {"reader1" : weights}
        write_weights = {}
        read_defaults = {"reader1" : defaults}
        write_defaults = {}
        state_tensor = StateTensor(memory,
                                   read_weights,
                                   write_weights,
                                   read_defaults,
                                   write_defaults)


        # perform test cases
        for creation_mode, merge_mode in itertools.product(creation_modes, merge_mode):

            # Define reader. Then implement read

            reader = Reader(name,
                            memory_size,
                            memory_width,
                            num_heads,
                            control_width,
                            shift_kernel_width,
                            creation_mode,
                            merge_mode,
                            ensemble_shape=ensemble_size)

            read_results, new_state = reader(control_state, state_tensor)

            self.assertTrue(read_results.shape == (batch_size, ensemble_size, memory_width))
            self.assertTrue(new_state.read_weights[name].shape == weights.shape)
            self.assertTrue(read_results.dtype == torch.float32)

    def test_torchscript_compiles(self):
        """ Test that torchscript can compile the code"""

        # Define test parameters
        name = "reader1"
        batch_size = 24
        ensemble_size = 17
        memory_size = 10
        memory_width = 21
        num_heads = 4
        control_width = 20
        shift_kernel_width = 3
        creation_modes = ["reshape", "project"]
        merge_mode = ["sum", "weighted_sum", "project"]

        # Define the test inputs
        control_state = torch.randn(batch_size, ensemble_size, control_width)

        memory = torch.randn(batch_size,ensemble_size, memory_size, memory_width)
        weights = torch.softmax(torch.randn(batch_size, ensemble_size,  num_heads, memory_size), dim=-1)
        defaults = torch.softmax(torch.randn(ensemble_size, num_heads, memory_size), dim=-1)

        read_weights = {"reader1" : weights}
        write_weights = {}
        read_defaults = {"reader1" : defaults}
        write_defaults = {}
        state_tensor = StateTensor(memory,
                                   read_weights,
                                   write_weights,
                                   read_defaults,
                                   write_defaults)


        # perform test cases
        for creation_mode, merge_mode in itertools.product(creation_modes, merge_mode):

            # Define reader. Then implement read

            reader = Reader(name,
                            memory_size,
                            memory_width,
                            num_heads,
                            control_width,
                            shift_kernel_width,
                            creation_mode,
                            merge_mode,
                            ensemble_shape=ensemble_size)
            reader = torch.jit.script(reader)

            read_results, new_state = reader(control_state, state_tensor)

            self.assertTrue(read_results.shape == (batch_size, ensemble_size, memory_width))
            self.assertTrue(new_state.read_weights[name].shape == weights.shape)
            self.assertTrue(read_results.dtype == torch.float32)

class TestWriter(unittest.TestCase):
    """
    A test suite for the ntm writer.

    The writer is responsible for transferring information
    into the ntm state tensor.
    """
    def test_modes_basic(self):
        """
         Test all the various head modes in a simple
        batched configuration

        """
        # Define test parameters
        name = "writer1"
        batch_size = 24
        memory_size = 10
        memory_width = 21
        num_heads = 4
        control_width = 20
        data_width = 28
        shift_kernel_width = 3
        creation_modes = ["reshape", "project"]

        # Define the test inputs
        control_state = torch.randn(batch_size, control_width)
        data = torch.randn(batch_size, data_width)

        memory = torch.randn(batch_size, memory_size, memory_width)
        weights = torch.softmax(torch.randn(batch_size, num_heads, memory_size), dim=-1)
        defaults = torch.softmax(torch.randn(num_heads, memory_size), dim=-1)

        read_weights = {}
        write_weights = {"writer1": weights}
        read_defaults = {}
        write_defaults = {"writer1": defaults}
        state_tensor = StateTensor(memory,
                                   read_weights,
                                   write_weights,
                                   read_defaults,
                                   write_defaults)
        # perform test cases
        for creation_mode in creation_modes:

            # Define reader. Then implement read

            writer = Writer(name,
                            memory_size,
                            memory_width,
                            num_heads,
                            control_width,
                            data_width,
                            shift_kernel_width,
                            creation_mode,
                            False,
                            )

            new_state = writer(control_state, data, state_tensor)

            self.assertTrue(torch.any(new_state.memory != state_tensor.memory))
    def test_in_ensemble(self):
        """
         Test layer in a ensemble configuration

        """
        # Define test parameters
        name = "writer1"
        batch_size = 24
        memory_size = 10
        memory_width = 21
        num_heads = 4
        control_width = 20
        shift_kernel_width = 3
        ensemble_width = 27
        creation_modes = ["reshape", "project"]

        # Define the test inputs
        control_state = torch.randn(batch_size,ensemble_width, control_width)

        memory = torch.randn(batch_size, ensemble_width, memory_size, memory_width)
        weights = torch.softmax(torch.randn(batch_size, ensemble_width, num_heads, memory_size), dim=-1)
        defaults = torch.softmax(torch.randn(ensemble_width, num_heads,  memory_size), dim=-1)

        read_weights = {}
        write_weights = {"writer1": weights}
        read_defaults = {}
        write_defaults = {"writer1": defaults}
        state_tensor = StateTensor(memory,
                                   read_weights,
                                   write_weights,
                                   read_defaults,
                                   write_defaults)
        # perform test cases
        for creation_mode in creation_modes:

            # Define reader. Then implement read

            writer = Writer(name,
                            memory_size,
                            memory_width,
                            num_heads,
                            control_width,
                            shift_kernel_width,
                            creation_mode,
                            True,
                            ensemble_width)

            new_state = writer(control_state, state_tensor)

            self.assertTrue(torch.any(new_state.memory != state_tensor.memory))

    def test_jit_script(self):
        """
         Test torch can script the layer

        """
        # Define test parameters
        name = "writer1"
        batch_size = 24
        memory_size = 10
        memory_width = 21
        num_heads = 4
        control_width = 20
        shift_kernel_width = 3
        ensemble_width = 27
        creation_modes = ["reshape", "project"]

        # Define the test inputs
        control_state = torch.randn(batch_size,ensemble_width, control_width)

        memory = torch.randn(batch_size, ensemble_width, memory_size, memory_width)
        weights = torch.softmax(torch.randn(batch_size, ensemble_width, num_heads, memory_size), dim=-1)
        defaults = torch.softmax(torch.randn(ensemble_width, num_heads,  memory_size), dim=-1)

        read_weights = {}
        write_weights = {"writer1": weights}
        read_defaults = {}
        write_defaults = {"writer1": defaults}
        state_tensor = StateTensor(memory,
                                   read_weights,
                                   write_weights,
                                   read_defaults,
                                   write_defaults)
        # perform test cases
        for creation_mode in creation_modes:

            # Define reader. Then implement read

            writer = Writer(name,
                            memory_size,
                            memory_width,
                            num_heads,
                            control_width,
                            shift_kernel_width,
                            creation_mode,
                            True,
                            ensemble_width)
            writer = torch.jit.script(writer)

            new_state = writer(control_state, state_tensor)

            self.assertTrue(torch.any(new_state.memory != state_tensor.memory))