"""

A simple test collection to test
the functions subsection of Core.

"""



from typing import List

import itertools
import unittest
import numpy as np
import torch


from torch.nn import functional as F
from supertransformerlib import Core

PRINT_ERRORS = True


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

    array = tensor.numpy()
    array = np.pad(array, ordered_pairings, mode="wrap")
    output = torch.from_numpy(array)

    return output


class TestCirculuarPadding(unittest.TestCase):
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
        """ Test a variety of pad permutations"""

        tensor = torch.arange(1000).view(10, 10, 10)
        ndim = 2
        paddings = [[0, 1, 2]]*2*ndim

        options = itertools.product(*paddings)
        for option in options:
            expected = example_circular_padding(tensor, option)
            got = Core.pad_circular(tensor, option)
            self.assertTrue(torch.all(expected == got))
    def test_torchscript(self):
        """ Test the function torchscript compiles"""
        tensor = torch.arange(1000).view(10, 10, 10)
        ndim = 2
        paddings = [[0, 1, 2]]*2*ndim

        options = itertools.product(*paddings)
        padder = torch.jit.script(Core.pad_circular)
        for option in options:
            expected = example_circular_padding(tensor, option)
            got = padder(tensor, option)
            self.assertTrue(torch.all(expected == got))
class TestCircularPaddingErrors(unittest.TestCase):
    """
    Test that the circular padding mechanism is making sane errors.
    """
    def test_padding_wrong_rank(self):
        """ test that the padding throws on bad rank"""
        tensor = torch.randn([10])
        padding = [1, 1, 1, 1]
        try:
            Core.pad_circular(tensor, padding)
            raise RuntimeError("Did not throw")
        except Core.PaddingException as err:
            if PRINT_ERRORS:
                print(err)


    def test_padding_not_symmetric(self):
        """test we throw when we do not get padding before AND after"""
        tensor = torch.randn([10])
        padding = [1]

        try:
            Core.pad_circular(tensor, padding)
            raise RuntimeError("Did not throw")
        except Core.PaddingException as err:
            if PRINT_ERRORS:
                print(err)

    def test_padding_negative(self):
        """TEst we throw when a negative pad length is passed"""
        tensor = torch.randn([10])
        padding = [-1, 10]

        try:
            Core.pad_circular(tensor, padding)
            raise RuntimeError("Did not throw")
        except Core.PaddingException as err:
            if PRINT_ERRORS:
                print(err)

    def test_tensor_dead_dim(self):
        """ Test we throw when there is nothing to pad with"""
        tensor = torch.randn([1, 10, 0])
        padding = [1, 0]

        try:
            Core.pad_circular(tensor, padding)
            raise RuntimeError("Did not throw")
        except Core.PaddingException as err:
            if PRINT_ERRORS:
                print(err)
