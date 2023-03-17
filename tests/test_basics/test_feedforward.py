import unittest

import torch

from src.supertransformerlib import Basics


class TestFeedforward(unittest.TestCase):
    def test_straightforward(self):
        """Test feedforward works without any tricks"""
        test_tensor = torch.randn([10, 20, 4, 16])


        layer = Basics.Feedforward(16)
        layer = torch.jit.script(layer)

        output = layer(test_tensor)

        self.assertTrue(output.shape == torch.Size([10, 20, 4, 16]))
        self.assertTrue(torch.any(output != test_tensor))

    def test_parallel(self):
        """ Test the parallel processing system is engaging"""
        test_tensor = torch.randn([10, 20, 4, 16])

        layer = Basics.Feedforward(16, parallel=[10, 20])
        layer = torch.jit.script(layer)

        output = layer(test_tensor)
        self.assertTrue(output.shape == torch.Size([10, 20, 4, 16]))
        self.assertTrue(torch.any(output != test_tensor))