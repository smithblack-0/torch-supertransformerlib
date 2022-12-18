"""
Test module for the parameter block attention mechanism.

"""

import torch
import unittest
from src.supertransformerlib.Attention import parameter_block_attention


class TestParameterBlockAttention(unittest.TestCase):
    """
    A test fixture for testing parameter block
    attention. This attention mechanism is designed
    to inject a large amount of related memories or materials
    into the tensor stream when the appropriate key is triggered.
    """

    def test_base_case(self):
        test_tensor = torch.randn([10, 30, 64])

    def test_attention_modes(self):


