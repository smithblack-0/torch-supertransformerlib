import torch
import unittest

import NTM.indexer
from NTM import heads

class TestIndexer(unittest.TestCase):

    def test_create_control_tensors(self):

        # create the test instances
        batch_shape = 14


        indexer = NTM.indexer.Indexer




    def test_shift(self):
        # Compute expected result using for loop
        expected_result = []
        for i in range(self.shift_kernel_size):
            shifted_weights = torch.roll(self.weights, shifts=i-1, dims=-1)
            expected_result.append(shifted_weights * self.shift_prob[..., i:i+1])
        expected_result = torch.sum(torch.stack(expected_result, dim=-1), dim=-1)

        # Compute actual result using vectorized implementation
        actual_result = self.indexer.shift(self.shift_prob, self.weights)

        # Check that expected and actual results match
        self.assertTrue(torch.allclose(expected_result, actual_result, rtol=1e-05, atol=1e-05))