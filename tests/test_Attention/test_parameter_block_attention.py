"""
Test module for the PI attention mechanism.

"""

import torch
import unittest
from src.supertransformerlib.Attention import parameter_block_attention

class TestParameterBlockAttn(unittest.TestCase):
    """
    A test fixture for parameter block attention
    """
    def test_softmax_mode(self):
        """ test system in softmax (default) mode"""

    def test_sigmoid_mode(self):
        """ test system in """