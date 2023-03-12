import unittest
import torch
import textwrap
from src.supertransformerlib.Basics import syncronized_tensor

class test_syncronous_logic(unittest.TestCase):

    """Tests that generation will work properly"""
    def test_generate_class(self):
        """ Test that the generate class portion is working properly"""

        # Function to append
        def double(x: torch.Tensor) -> torch.Tensor:
            return 2 * x

        # Fields which are available
        info = [("name", str), ("age", int)]
        fields = ["x", "y"]
        transformations = {"double_x": double, "double_y": double}

        # Make call
        class_code = syncronized_tensor.generate_parallel_container("MyClass", info, fields, transformations)

        # Run assert
        with open("test_basics/example.txt") as f:
            expected = f.read()
        self.assertTrue(class_code == expected)