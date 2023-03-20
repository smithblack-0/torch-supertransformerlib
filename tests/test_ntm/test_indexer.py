import torch
import unittest
from torch.nn import functional as F
from src.supertransformerlib.NTM import indexer as index_module

class TestIndexer(unittest.TestCase):

    def test_create_control_tensors(self):
        """ Test the create control tensor helper class"""

        # Define test parameters

        batch_shape = 14
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24
        shift_kernel_width = 4

        # Make test indexer and input tensors
        indexer = index_module.Indexer(memory_size,
                                          memory_embeddings,
                                          control_width,
                                          shift_kernel_width)
        control_state = torch.randn([batch_shape, word_shape, control_width])

        # Create control tensors
        control_tensors = indexer.create_control_tensors(control_state)

        # Test existance
        self.assertTrue("content_key" in control_tensors)
        self.assertTrue("content_strength" in control_tensors)
        self.assertTrue("interpolation_probability" in control_tensors)
        self.assertTrue("shift_probabilities" in control_tensors)
        self.assertTrue("sharpening_logit" in control_tensors)

        # Test restrictions

        content_constraints_satisfied = control_tensors["content_strength"] >= 0
        self.assertTrue(torch.all(content_constraints_satisfied))

        interpolation_constraints_satisfied = torch.logical_and( 0 <= control_tensors["interpolation_probability"],
                                                                 1 >= control_tensors["interpolation_probability"])
        self.assertTrue(torch.all(interpolation_constraints_satisfied))

        shift_constraints_satisfied =  0 <= control_tensors["shift_probabilities"]
        shift_constraints_satisfied = torch.logical_and(shift_constraints_satisfied,
                                                        1 >= control_tensors["shift_probabilities"])
        shift_sum_satisfied = 1e-4 > torch.abs(control_tensors["shift_probabilities"].sum(dim=-1)) - 1
        self.assertTrue(torch.all(shift_constraints_satisfied))
        self.assertTrue(torch.all(shift_sum_satisfied))

        sharpening_constraint_satisfied = 1 <= control_tensors["sharpening_logit"]
        self.assertTrue(torch.all(sharpening_constraint_satisfied))



    def test_content_addressing(self):
        """Test the content_addressing method in the Indexer class."""

        # Define test parameters
        batch_shape = 14
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24
        shift_kernel_width = 4

        # Create an instance of the Indexer class
        indexer = index_module.Indexer(memory_size,
                                          memory_embeddings,
                                          control_width,
                                          shift_kernel_width)

        # Create input tensors
        memory = torch.randn([batch_shape, word_shape, memory_size, memory_embeddings])
        keys = torch.randn([batch_shape, word_shape, memory_embeddings])
        strength = torch.rand([batch_shape, word_shape, 1])

        # Call the content_addressing method
        content_weights = indexer.content_addressing(memory, keys, strength)

        # Check output shape
        self.assertEqual(content_weights.shape, (batch_shape, word_shape, memory_size))

        # Calculate the expected content_weights using cosine similarity and softmax
        similarity = F.cosine_similarity(memory, keys.unsqueeze(-2), dim=-1)
        expected_content_weights = F.softmax(similarity * strength, dim=-2)

        # Compare the expected and actual content_weights
        content_weights_diff = torch.abs(content_weights - expected_content_weights)

        # Set a tolerance level for the differences
        tolerance = 1e-6

        # Assert that the differences are within the tolerance level
        self.assertTrue(torch.all(content_weights_diff < tolerance))

    def test_interpolate(self):
        """Test the interpolate method in the Indexer class."""

        # Define test parameters
        batch_shape = 14
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24
        shift_kernel_width = 4

        # Create an instance of the Indexer class
        indexer = index_module.Indexer(memory_size,
                                          memory_embeddings,
                                          control_width,
                                          shift_kernel_width)

        # Create input tensors
        interp_prob = torch.rand([batch_shape, word_shape, 1])
        content_weights = torch.rand([batch_shape, word_shape, memory_size])
        prior_weights = torch.rand([batch_shape, word_shape, memory_size])

        # Call the interpolate method
        new_weights = indexer.interpolate(interp_prob, content_weights, prior_weights)

        # Check output shape
        self.assertEqual(new_weights.shape, (batch_shape, word_shape, memory_size))

        # Calculate the expected new_weights based on interpolation
        expected_new_weights = interp_prob * content_weights + (1 - interp_prob) * prior_weights

        # Compare the expected and actual new_weights
        new_weights_diff = torch.abs(new_weights - expected_new_weights)

        # Set a tolerance level for the differences
        tolerance = 1e-6

        # Assert that the differences are within the tolerance level
        self.assertTrue(torch.all(new_weights_diff < tolerance))

    def test_shift(self):
        """Test the shift method in the Indexer class."""

        # Define test parameters
        batch_shape = 14
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24
        shift_kernel_width = 4

        # Create an instance of the Indexer class
        indexer = index_module.Indexer(memory_size,
                                          memory_embeddings,
                                          control_width,
                                          shift_kernel_width)

        # Create input tensors
        shift_prob = F.softmax(torch.randn([batch_shape, word_shape, shift_kernel_width]), dim=-1)
        weights = torch.rand([batch_shape, word_shape, memory_size])

        # Call the shift method
        shifted_weights = indexer.shift(shift_prob, weights)

        # Check output shape
        self.assertEqual(shifted_weights.shape, (batch_shape, word_shape, memory_size))

        # Verify that the shifted weights are computed correctly
        shift_kernel_size = shift_prob.shape[-1]
        roll_values = torch.arange(-(shift_kernel_size // 2), shift_kernel_size // 2 + 1, device=shift_prob.device)

        shift_accumulator = []
        for roll_value, shift_prob_case in zip(roll_values, shift_prob.unbind(-1)):
            rolled_case = torch.roll(weights, int(roll_value), dims=-1)
            weighted_case = rolled_case * shift_prob_case.unsqueeze(-1)
            shift_accumulator.append(weighted_case)

        expected_shifted_weights = torch.stack(shift_accumulator, dim=-1).sum(dim=-1)

        # Compare the expected and actual shifted_weights
        shifted_weights_diff = torch.abs(shifted_weights - expected_shifted_weights)

        # Set a tolerance level for the differences
        tolerance = 1e-6

        # Assert that the differences are within the tolerance level
        self.assertTrue(torch.all(shifted_weights_diff < tolerance))

    def test_sharpen(self):
        """Test the sharpen method in the Indexer class."""

        # Define test parameters
        batch_shape = 14
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24
        shift_kernel_width = 4

        # Create an instance of the Indexer class
        indexer = index_module.Indexer(memory_size,
                                          memory_embeddings,
                                          control_width,
                                          shift_kernel_width)

        # Create input tensors
        sharpening = torch.rand([batch_shape, word_shape, 1])  # Values between 0 and 1
        weights = F.softmax(torch.randn([batch_shape, word_shape, memory_size]), dim=-1)

        # Call the sharpen method
        sharp_weights = indexer.sharpen(sharpening, weights)

        # Check output shape
        self.assertEqual(sharp_weights.shape, (batch_shape, word_shape, memory_size))

        # Verify that the sharpened weights are computed correctly
        expected_sharp_weights = weights ** (sharpening + 1)
        expected_sharp_weights = expected_sharp_weights / torch.sum(expected_sharp_weights, dim=-1, keepdim=True)

        # Compare the expected and actual sharp_weights
        sharp_weights_diff = torch.abs(sharp_weights - expected_sharp_weights)

        # Set a tolerance level for the differences
        tolerance = 1e-6

        # Assert that the differences are within the tolerance level
        self.assertTrue(torch.all(sharp_weights_diff < tolerance))

        # Check that the sharpened weights still sum to 1 (or very close to 1)
        sharp_weights_sum = sharp_weights.sum(dim=-1)
        self.assertTrue(torch.all(torch.abs(sharp_weights_sum - 1.0) < tolerance))

    def test_end_to_end(self):
        """Test the end-to-end functionality of the Indexer class."""

        # Define test parameters
        batch_shape = 14
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24
        shift_kernel_width = 4

        # Create an instance of the Indexer class
        indexer = index_module.Indexer(memory_size,
                                       memory_embeddings,
                                       control_width,
                                       shift_kernel_width)

        # Create input tensors
        memory = torch.randn([batch_shape, word_shape, memory_size, memory_embeddings])
        prior_weights = F.softmax(torch.randn([batch_shape, word_shape, memory_size]), dim=-1)
        control_state = torch.randn([batch_shape, word_shape, control_width])

        # Call the Indexer layer with the input tensors
        final_weights = indexer(control_state, memory, prior_weights)

        # Check output shape
        self.assertEqual(final_weights.shape, (batch_shape, word_shape, memory_size))

        # Check that the final weights still sum to 1 (or very close to 1)
        tolerance = 1e-6
        final_weights_sum = final_weights.sum(dim=-1)
        self.assertTrue(torch.all(torch.abs(final_weights_sum - 1.0) < tolerance))

        # Assert that the final_weights values are within the valid range [0, 1]
        weights_range_satisfied = torch.logical_and(0 <= final_weights, 1 >= final_weights)
        self.assertTrue(torch.all(weights_range_satisfied))
    def test_end_to_end(self):
        """Test the end-to-end functionality of the Indexer class."""

        # Define test parameters
        batch_shape = 14
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24
        shift_kernel_width = 4

        # Create an instance of the Indexer class
        indexer = index_module.Indexer(memory_size,
                                       memory_embeddings,
                                       control_width,
                                       shift_kernel_width)

        # Create input tensors
        memory = torch.randn([batch_shape, word_shape, memory_size, memory_embeddings])
        prior_weights = F.softmax(torch.randn([batch_shape, word_shape, memory_size]), dim=-1)
        control_state = torch.randn([batch_shape, word_shape, control_width])

        # Call the Indexer layer with the input tensors
        final_weights = indexer(control_state, memory, prior_weights)

        # Check output shape
        self.assertEqual(final_weights.shape, (batch_shape, word_shape, memory_size))

        # Check that the final weights still sum to 1 (or very close to 1)
        tolerance = 1e-6
        final_weights_sum = final_weights.sum(dim=-1)
        self.assertTrue(torch.all(torch.abs(final_weights_sum - 1.0) < tolerance))

        # Assert that the final_weights values are within the valid range [0, 1]
        weights_range_satisfied = torch.logical_and(0 <= final_weights, 1 >= final_weights)
        self.assertTrue(torch.all(weights_range_satisfied))

    def test_end_to_end_with_ensemble(self):
        """Test the end-to-end functionality of the Indexer class with ensemble dimensions."""

        # Define test parameters
        batch_shape = 14
        ensemble_shape = 10
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24
        shift_kernel_width = 4

        # Create an instance of the Indexer class
        indexer = index_module.Indexer(memory_size,
                                       memory_embeddings,
                                       control_width,
                                       shift_kernel_width)

        # Create input tensors with ensemble dimensions
        control_state = torch.randn([batch_shape, ensemble_shape, control_width])
        memory = torch.randn([batch_shape, ensemble_shape, memory_size, memory_embeddings])
        prior_weights = F.softmax(torch.randn([batch_shape,ensemble_shape, memory_size]), dim=-1)

        # Call the forward method of the Indexer class with the input tensors
        updated_weights = indexer(control_state, memory, prior_weights)

        # Check output shape
        self.assertEqual(updated_weights.shape, (batch_shape, ensemble_shape, memory_size))

        # Check that the updated weights still sum to 1 (or very close to 1)
        updated_weights_sum = updated_weights.sum(dim=-1)
        tolerance = 1e-6
        self.assertTrue(torch.all(torch.abs(updated_weights_sum - 1.0) < tolerance))

    def test_end_to_end_with_multidimensional_ensemble(self):
        """Test the end-to-end functionality of the Indexer class with multi-dimensional ensemble dimensions."""

        # Define test parameters
        batch_shape = 14
        ensemble_shape = [10, 16]
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24
        shift_kernel_width = 4

        # Create an instance of the Indexer class
        indexer = index_module.Indexer(memory_size,
                                       memory_embeddings,
                                       control_width,
                                       shift_kernel_width)

        # Create input tensors with multi-dimensional ensemble dimensions
        control_state = torch.randn([batch_shape, *ensemble_shape, control_width])
        memory = torch.randn([batch_shape, *ensemble_shape, memory_size, memory_embeddings])
        prior_weights = F.softmax(torch.randn([batch_shape, *ensemble_shape, memory_size]), dim=-1)

        # Call the forward method of the Indexer class with the input tensors
        updated_weights = indexer(control_state, memory, prior_weights)

        # Check output shape
        self.assertEqual(updated_weights.shape, (batch_shape, *ensemble_shape, memory_size))

        # Check that the updated weights still sum to 1 (or very close to 1)
        updated_weights_sum = updated_weights.sum(dim=-1)
        tolerance = 1e-6
        self.assertTrue(torch.all(torch.abs(updated_weights_sum - 1.0) < tolerance))

    def test_end_to_end_with_torchscript(self):
        """Test the end-to-end functionality of the Indexer class with TorchScript."""

        # Define test parameters
        batch_shape = 14
        word_shape = 15
        memory_size = 24
        memory_embeddings = 17
        control_width = 24
        shift_kernel_width = 4

        # Create an instance of the Indexer class
        indexer = index_module.Indexer(memory_size,
                                       memory_embeddings,
                                       control_width,
                                       shift_kernel_width)

        # Script the indexer instance
        scripted_indexer = torch.jit.script(indexer)

        # Create input tensors
        control_state = torch.randn([batch_shape, word_shape, control_width])
        memory = torch.randn([batch_shape, word_shape, memory_size, memory_embeddings])
        prior_weights = F.softmax(torch.randn([batch_shape, word_shape, memory_size]), dim=-1)

        # Call the forward method of the scripted Indexer class with the input tensors
        updated_weights = scripted_indexer(control_state, memory, prior_weights)

        # Check output shape
        self.assertEqual(updated_weights.shape, (batch_shape, word_shape, memory_size))

        # Check that the updated weights still sum to 1 (or very close to 1)
        updated_weights_sum = updated_weights.sum(dim=-1)
        tolerance = 1e-6
        self.assertTrue(torch.all(torch.abs(updated_weights_sum - 1.0) < tolerance))