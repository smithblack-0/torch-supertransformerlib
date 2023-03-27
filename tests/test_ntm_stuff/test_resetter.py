import unittest
import torch
from supertransformerlib.NTM import reset

class TestMemoryResetter(unittest.TestCase):
    """ Test the reset memory capabilities"""
    def test_reset_memory(self):
        """Test the reset_memory feature in reset."""

        # Define test parameters
        batch_shape = 14
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24

        # Create an instance of the resetter class
        resetter = reset.MemoryResetter(control_width,
                                        memory_size,
                                        memory_embeddings)


        # Create input tensors
        control_state = torch.rand([batch_shape, word_shape, control_width])
        memory = torch.randn([batch_shape, word_shape, memory_size, memory_embeddings])

        # Call the reset_memory method
        new_memory= resetter(control_state, memory)

        # Check output shapes
        self.assertEqual(new_memory.shape, memory.shape)

    def test_force_reset(self):
        """ test that the force reset option does change the indicated dimensions"""

        # Define test parameters
        batch_shape = 2
        memory_size = 24
        memory_embeddings = 17
        control_width = 24

        # Create an instance of the resetter class
        resetter = reset.MemoryResetter(control_width,
                                        memory_size,
                                        memory_embeddings)


        # Create input tensors
        batch_mask = torch.tensor([True, False])
        memory = torch.randn([batch_shape, memory_size, memory_embeddings])

        # Call the reset_memory method
        new_memory= resetter.force_reset(batch_mask, memory)

        # Check assertions
        self.assertEqual(new_memory.shape, memory.shape)
        self.assertTrue(torch.any(memory[0] != new_memory[0]))
        self.assertTrue(torch.all(memory[1] == new_memory[1]))
    def test_torchscript_compiles(self):
        """ Test the class torchscript compiles"""
        # Define test parameters
        batch_shape = 14
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24

        # Create an instance of the resetter class
        resetter = reset.MemoryResetter(control_width,
                                        memory_size,
                                        memory_embeddings)
        resetter = torch.jit.script(resetter)


        # Create input tensors
        control_state = torch.rand([batch_shape, word_shape, control_width])
        memory = torch.randn([batch_shape, word_shape, memory_size, memory_embeddings])

        # Call the reset_memory method
        new_memory= resetter(control_state, memory)

        # Check output shapes
        self.assertEqual(new_memory.shape, memory.shape)
class TestWeightsResetter(unittest.TestCase):


    def test_reset_weights(self):
        """ Test the reset weights class in reset"""

        # Define test parameters
        batch_shape = 14
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24

        # Create an instance of the resetter class
        resetter = reset.WeightsResetter(control_width,
                                        memory_size,
                                        memory_embeddings)


        # Create input tensors
        control_state = torch.rand([batch_shape, word_shape, control_width])
        weights = torch.randn([batch_shape, word_shape, memory_size])

        # Call the reset_memory method
        new_weights = resetter(control_state, weights)

        # Check output shapes
        self.assertEqual(new_weights.shape, weights.shape)

    def test_force_reset(self):
        """ test that the force reset option does change the indicated dimensions"""

        # Define test parameters
        batch_shape = 2
        memory_size = 24
        memory_embeddings = 17
        control_width = 24

        # Create an instance of the resetter class
        resetter = reset.WeightsResetter(control_width,
                                        memory_size,
                                        memory_embeddings)


        # Create input tensors
        batch_mask = torch.tensor([True, False])
        weights = torch.randn([batch_shape, memory_size])

        # Call the reset_memory method
        new_weights= resetter.force_reset(batch_mask, weights)

        # Check assertions
        self.assertEqual(new_weights.shape, weights.shape)
        self.assertTrue(torch.any(weights[0] != new_weights[0]))
        self.assertTrue(torch.all(weights[1] == new_weights[1]))
    def test_torchscript(self):
        """ Test that the class compiles correctly under torchscript"""

        # Define test parameters
        batch_shape = 14
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24

        # Create an instance of the resetter class
        resetter = reset.WeightsResetter(control_width,
                                        memory_size,
                                        memory_embeddings)
        resetter = torch.jit.script(resetter)


        # Create input tensors
        control_state = torch.rand([batch_shape, word_shape, control_width])
        weights = torch.randn([batch_shape, word_shape, memory_size])

        # Call the reset_memory method
        new_weights = resetter(control_state, weights)

        # Check output shapes
        self.assertEqual(new_weights.shape, weights.shape)