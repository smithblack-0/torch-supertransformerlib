import unittest
import torch
from torch import nn
from typing import List, Optional
from supertransformerlib.NTM import defaults

# Assuming the MemDefault class is defined here or imported




class TestInterpolateTools(unittest.TestCase):
    """
    Test the functions floating around which are
    utilized for various purposes.
    """
    def test_interpolate_reset_mem_elements(self):
        """ Test that reset memory functions as is appropriate"""
        mem_size = 10
        mem_width = 5
        ensemble_shape = [2, 3]
        batch_shape = [4, 2]

        mem_default = torch.zeros(ensemble_shape + [mem_size, mem_width])
        memory = 2 * torch.ones(batch_shape + ensemble_shape + [mem_size, mem_width])
        reset_probabilities = 0.3 * torch.ones(batch_shape + ensemble_shape + [mem_size])

        updated_memory = defaults.interpolate_tensor_defaults(reset_probabilities, memory, mem_default)
        expected_memory = memory * (1 - reset_probabilities.unsqueeze(-1)) + \
                          reset_probabilities.unsqueeze(-1) * mem_default


        self.assertEqual(updated_memory.shape, tuple(batch_shape + ensemble_shape + [mem_size, mem_width]))
        self.assertTrue(torch.allclose(updated_memory, expected_memory))

    def test_interpolate_reset_ensemble_bank(self):
        """ Test all is well when resetting an entire memory unit"""
        mem_size = 10
        mem_width = 5
        ensemble_shape = [2, 3]
        batch_shape = [4, 2]

        mem_default = torch.zeros(ensemble_shape + [mem_size, mem_width])
        memory = 2 * torch.ones(batch_shape + ensemble_shape + [mem_size, mem_width])
        reset_probabilities = 0.3 * torch.ones(batch_shape + ensemble_shape)

        updated_memory = defaults.interpolate_tensor_defaults(reset_probabilities, memory, mem_default)
        reset_probabilities = reset_probabilities.unsqueeze(-1).unsqueeze(-1)
        expected_memory = memory * (1 - reset_probabilities) + \
                          reset_probabilities * mem_default


        self.assertEqual(updated_memory.shape, tuple(batch_shape + ensemble_shape + [mem_size, mem_width]))
        self.assertTrue(torch.allclose(updated_memory, expected_memory))

    def test_interpolate_reset_batches(self):
        """ Test all is well when resetting an entire batches memory"""
        mem_size = 10
        mem_width = 5
        ensemble_shape = [2, 3]
        batch_shape = [4, 2]

        mem_default = torch.zeros(ensemble_shape + [mem_size, mem_width])
        memory = 2 * torch.ones(batch_shape + ensemble_shape + [mem_size, mem_width])
        reset_probabilities = 0.3 * torch.ones(batch_shape)

        updated_memory = defaults.interpolate_tensor_defaults(reset_probabilities, memory, mem_default)
        reset_probabilities = reset_probabilities.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        expected_memory = memory * (1 - reset_probabilities) + \
                          reset_probabilities * mem_default


        self.assertEqual(updated_memory.shape, tuple(batch_shape + ensemble_shape + [mem_size, mem_width]))
        self.assertTrue(torch.allclose(updated_memory, expected_memory))

    def test_jit_compiles(self):
        """ Test that jit compiles and runs"""
        mem_size = 10
        mem_width = 5
        ensemble_shape = [2, 3]
        batch_shape = [4, 2]

        mem_default = torch.zeros(ensemble_shape + [mem_size, mem_width])
        memory = 2 * torch.ones(batch_shape + ensemble_shape + [mem_size, mem_width])
        reset_probabilities = 0.3 * torch.ones(batch_shape)

        func = torch.jit.script(defaults.interpolate_tensor_defaults)
        updated_memory = func(reset_probabilities, memory, mem_default)
        reset_probabilities = reset_probabilities.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        expected_memory = memory * (1 - reset_probabilities) + \
                          reset_probabilities * mem_default


        self.assertEqual(updated_memory.shape, tuple(batch_shape + ensemble_shape + [mem_size, mem_width]))
        self.assertTrue(torch.allclose(updated_memory, expected_memory))

class TestMemDefault(unittest.TestCase):

    def test_constructor(self):
        """ Test the ability of the constructor to resolve in both modes"""
        memory_size = 10
        memory_width = 5
        ensemble_shape = [2, 3]

        # Test without ensemble_shape
        mem_default = defaults.DefaultMemoryParameters(memory_size, memory_width)
        self.assertIsInstance(mem_default.memory_default, nn.Parameter)
        self.assertEqual(mem_default.memory_default.shape, (memory_size, memory_width))
        self.assertFalse(torch.equal(mem_default.memory_default, torch.zeros(memory_size, memory_width)))

        # Test with ensemble_shape
        mem_default = defaults.DefaultMemoryParameters(memory_size, memory_width, ensemble_shape)
        self.assertIsInstance(mem_default.memory_default, nn.Parameter)
        self.assertTrue(mem_default.memory_default.shape == torch.Size(ensemble_shape + [memory_size, memory_width]))
        self.assertFalse(torch.equal(mem_default.memory_default, torch.zeros(ensemble_shape + [memory_size, memory_width])))

    def test_make_memory(self):
        """ Tes the ability of the class to make memory units"""
        memory_size = 10
        memory_width = 5
        ensemble_shape = [2, 3]
        batch_shape = [4, 2]

        mem_default = defaults.DefaultMemoryParameters(memory_size, memory_width, ensemble_shape)
        mem_default.memory_default = nn.Parameter(torch.ones(ensemble_shape + [memory_size, memory_width]))

        memory = mem_default.make_memory(batch_shape)
        expected_shape = batch_shape + ensemble_shape + [memory_size, memory_width]

        self.assertEqual(memory.shape, tuple(expected_shape))
        self.assertTrue(torch.allclose(memory, torch.ones(expected_shape)))

    def test_reset_memory(self):
        """ Test that reset memory functions as is appropriate"""
        memory_size = 10
        memory_width = 5
        ensemble_shape = [2, 3]
        batch_shape = [4, 2]

        mem_default = defaults.DefaultMemoryParameters(memory_size, memory_width, ensemble_shape)
        true_default = nn.Parameter(torch.zeros(ensemble_shape + [memory_size, memory_width]))
        mem_default.memory_default = true_default

        memory = 2 * torch.ones(batch_shape + ensemble_shape + [memory_size, memory_width])
        reset_probabilities = 0.3 * torch.ones(batch_shape + ensemble_shape + [memory_size])

        updated_memory = mem_default.reset_memory(memory, reset_probabilities)
        expected_memory = memory * (1 - reset_probabilities.unsqueeze(-1)) + \
                          reset_probabilities.unsqueeze(-1) * true_default


        self.assertEqual(updated_memory.shape, tuple(batch_shape + ensemble_shape + [memory_size, memory_width]))
        self.assertTrue(torch.allclose(updated_memory, expected_memory))

    def test_reset_to_defaults(self):
        """ Test that reset to defaults functions as appropriate"""
        memory_size = 10
        memory_width = 5
        ensemble_shape = [2, 3]
        batch_shape = [4, 2]

        mem_default = defaults.DefaultMemoryParameters(memory_size, memory_width, ensemble_shape)
        true_default = nn.Parameter(torch.zeros(ensemble_shape + [memory_size, memory_width]))
        mem_default.memory_default = true_default

        memory = 2 * torch.ones(batch_shape + ensemble_shape + [memory_size, memory_width])
        reset_mask = 0.3 * torch.ones(batch_shape + ensemble_shape + [memory_size]) > 1

        updated_memory = mem_default.force_reset_memory(memory, reset_mask)
        expected_memory = torch.where(reset_mask.unsqueeze(-1), true_default, memory)


        self.assertEqual(updated_memory.shape, tuple(batch_shape + ensemble_shape + [memory_size, memory_width]))
        self.assertTrue(torch.allclose(updated_memory, expected_memory))

    def test_jit_compiles(self):
        """ Test that the unit can jit compile"""
        memory_size = 10
        memory_width = 5
        ensemble_shape = [2, 3]
        batch_shape = [4, 2]

        mem_default = defaults.DefaultMemoryParameters(memory_size, memory_width, ensemble_shape)
        true_default = nn.Parameter(torch.zeros(ensemble_shape + [memory_size, memory_width]))
        mem_default.memory_default = true_default
        mem_default = torch.jit.script(mem_default)

        memory = 2 * torch.ones(batch_shape + ensemble_shape + [memory_size, memory_width])
        reset_mask = 0.3 * torch.ones(batch_shape + ensemble_shape + [memory_size]) > 1

        updated_memory = mem_default.force_reset_memory(memory, reset_mask)
        expected_memory = torch.where(reset_mask.unsqueeze(-1), true_default, memory)


        self.assertEqual(updated_memory.shape, tuple(batch_shape + ensemble_shape + [memory_size, memory_width]))
        self.assertTrue(torch.allclose(updated_memory, expected_memory))