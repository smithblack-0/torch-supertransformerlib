import unittest
import torch
import itertools
from supertransformerlib.NTM.ntm import Reader



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
        batch_size = 24
        memory_size = 10
        memory_width = 21
        num_heads = 4
        control_width = 20
        shift_kernel_width = 3
        creation_modes = ["reshape", "project"]
        merge_mode = ["sum", "weighted_sum", "project"]

        # Define the test inputs

        memory = torch.randn(batch_size, memory_size, memory_width)
        weights = torch.softmax(torch.randn(batch_size, num_heads, memory_size), dim=-2)
        control_state = torch.randn(batch_size, control_width)

        # perform test cases
        for creation_mode, merge_mode in itertools.product(creation_modes, merge_mode):

            # Define reader. Then implement read

            reader = Reader(memory_size,
                            memory_width,
                            num_heads,
                            control_width,
                            shift_kernel_width,
                            creation_mode,
                            merge_mode)

            read_results, new_weights = reader(control_state, memory, weights)

            self.assertTrue(read_results.shape == (batch_size, memory_width))
            self.assertTrue(new_weights.shape == weights.shape)
            self.assertTrue(read_results.dtype == torch.float32)

    def test_in_ensemble(self):
        """ Test the various modes still work on ensembles of data"""
        # Define test parameters
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

        memory = torch.randn(batch_size,ensemble_size, memory_size, memory_width)
        weights = torch.softmax(torch.randn(batch_size, ensemble_size,  num_heads, memory_size), dim=-2)
        control_state = torch.randn(batch_size, ensemble_size, control_width)

        # perform test cases
        for creation_mode, merge_mode in itertools.product(creation_modes, merge_mode):

            # Define reader. Then implement read

            reader = Reader(memory_size,
                            memory_width,
                            num_heads,
                            control_width,
                            shift_kernel_width,
                            creation_mode,
                            merge_mode,
                            ensemble_shape=ensemble_size)

            read_results, new_weights = reader(control_state, memory, weights)

            self.assertTrue(read_results.shape == (batch_size, ensemble_size, memory_width))
            self.assertTrue(new_weights.shape == weights.shape)
            self.assertTrue(read_results.dtype == torch.float32)

    def test_torchscript_compiles(self):
        """ Test that torchscript can compile the code"""

        # Define test parameters
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

        memory = torch.randn(batch_size, ensemble_size, memory_size, memory_width)
        weights = torch.softmax(torch.randn(batch_size, ensemble_size, num_heads, memory_size), dim=-2)
        control_state = torch.randn(batch_size, ensemble_size, control_width)

        # perform test cases
        for creation_mode, merge_mode in itertools.product(creation_modes, merge_mode):
            # Define reader. Then implement read

            reader = Reader(memory_size,
                            memory_width,
                            num_heads,
                            control_width,
                            shift_kernel_width,
                            creation_mode,
                            merge_mode,
                            ensemble_shape=ensemble_size)
            reader = torch.jit.script(reader)

            read_results, new_weights = reader(control_state, memory, weights)

            self.assertTrue(read_results.shape == (batch_size, ensemble_size, memory_width))
            self.assertTrue(new_weights.shape == weights.shape)
            self.assertTrue(read_results.dtype == torch.float32)
