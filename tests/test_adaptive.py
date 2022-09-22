import torch
import unittest
from src.supertransformerlib import Adaptive


class test_Adaptive_Attention(unittest.TestCase):
    def test_constructor(self):
        """Test the constructor functions reasonably well"""
        Adaptive.Adaptive_Attention(10, 20, 30, 5, 5, 5)
    def test_startup(self):
        """Test that the startup feature of the layer is able to make correct accumulators"""

        query = torch.zeros([10, 3, 5, 32])
        key = torch.zeros([10, 3, 5, 64])
        value = torch.zeros([10, 3, 5, 48])
        layer = Adaptive.Adaptive_Attention(32, 64, 48, 5, 5, 5)

        accumulator = layer.start_accumulator(query, key, value)
        self.assertTrue(accumulator.Halting_Probabilities.shape == query.shape[:-1])
        self.assertTrue(accumulator.Residuals.shape == query.shape[:-1])
        self.assertTrue(accumulator.Output.shape == torch.Size([*query.shape[:-1], value.shape[-1]]))

    def test_

