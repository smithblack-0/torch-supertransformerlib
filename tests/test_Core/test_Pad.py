import itertools
import unittest
from typing import List

import numpy as np
import torch
from torch.nn import functional as F

import src.supertransformerlib.Core as Core

print_errors = True


def example_circular_padding(tensor: torch.Tensor, paddings: List[int]):
    """
    A straightforward, but not torchscript compatible, version of
    circular padding
    """

    # Numpy has a nicely working circular wrap function
    # but this does not work with torchscript. Use
    # it for validation.

    paddings = torch.tensor(paddings, dtype=torch.int64)
    pairings = paddings.view(paddings.shape[0]//2, 2)
    ordered_pairings = torch.flip(pairings, dims=[0])

    interesting_length = ordered_pairings.shape[0]
    ordered_pairings = F.pad(ordered_pairings, (0, 0, tensor.dim() - interesting_length, 0))
    numpy_pairings = [(start, end) for start, end in ordered_pairings]

    array = tensor.numpy()
    array = np.pad(array, numpy_pairings, mode="wrap")
    output = torch.from_numpy(array)

    return output


class test_circular_padding(unittest.TestCase):
    """
    Test that the circular padding mechanism is working correctly.
    """
    def test_tools(self):
        """Test that my tool and torch's tool behave the same way"""

        tensor = torch.arange(4).view(2,2)
        padinstruction = [2, 2, 2, 2]

        expected = tensor.repeat(3, 3)
        got = example_circular_padding(tensor, padinstruction)
        self.assertTrue(torch.all(expected == got))

    def test_cases(self):

        tensor = torch.arange(1000).view(10, 10, 10)
        ndim = 2
        paddings = [[0, 1, 2]]*2*ndim

        options = itertools.product(*paddings)
        for option in options:
            try:
                expected = example_circular_padding(tensor, option)
                got = Core.Pad.pad_circular(tensor, option)
                self.assertTrue(torch.all(expected == got))
            except Exception as err:
                expected = example_circular_padding(tensor, option)
                got = Core.Pad.pad_circular(tensor, option)
                self.assertTrue(torch.all(expected == got))


class test_circular_padding_errors(unittest.TestCase):
    """
    Test that the circular padding mechanism is making sane errors.
    """
    def test_padding_too_long(self):
        tensor = torch.randn([10])
        padding = [1, 1, 1, 1]
        try:
            output = Core.Pad.pad_circular(tensor, padding)
            raise RuntimeError("Did not throw")
        except Core.Pad.PaddingException as err:
            if print_errors:
                print(err)


    def test_padding_not_symmetric(self):
        tensor = torch.randn([10])
        padding = [1]

        try:
            output = Core.Pad.pad_circular(tensor, padding)
            raise RuntimeError("Did not throw")
        except Core.Pad.PaddingException as err:
            if print_errors:
                print(err)

    def test_padding_negative(self):
        tensor = torch.randn([10])
        padding = [-1, 10]

        try:
            output = Core.Pad.pad_circular(tensor, padding)
            raise RuntimeError("Did not throw")
        except Core.Pad.PaddingException as err:
            if print_errors:
                print(err)

    def test_tensor_dead_dim(self):
        tensor = torch.randn([1, 10, 0])
        padding = [1, 0]

        try:
            output = Core.Pad.pad_circular(tensor, padding)
            raise RuntimeError("Did not throw")
        except Core.Pad.PaddingException as err:
            if print_errors:
                print(err)