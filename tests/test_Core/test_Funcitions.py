import unittest
import torch
from torch.nn import functional as F
from typing import List
from src.supertransformerlib.Core import Functions


def vectorized_circular_padding(tensor: torch.Tensor, paddings: List[int]):
    """
    A straightforward, if inefficient, version of circular padding
    """

    # Get a tensor in the correct order with
    # [start, end] portions.

    paddings = torch.tensor(paddings, dtype=torch.int64)
    pairings = paddings.view(paddings.shape[0]//2, 2)
    ordered_pairings = torch.flip(pairings, dims=[0])

    # Create a mesh out of the pairings. Then
    # use modular arithmetic to wrap the indices

    interesting_length = ordered_pairings.shape[0]
    dim_lengths = tensor.shape[-interesting_length:]
    indices = torch.meshgrid([torch.arange(-start, end + dim) for (start, end), dim
                              in zip(ordered_pairings, dim_lengths)], indexing="ij")
    indices = list(indices)
    for i, index in enumerate(indices):
        dim_length = dim_lengths[i]
        indices[i] = torch.remainder(index, dim_length)
    indices = tuple(indices)

    # Use the mesh to generate the padding. Move
    # the interesting dimensions to the front of the
    # tensor, dereference them, and then move the results back.

    output = tensor

    for _ in range(len(dim_lengths)):
        output = output.movedim(-1, 0)

    output = output[indices]

    for _ in range(len(dim_lengths)):
        output = output.movedim(0, -1)

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
        got = vectorized_circular_padding(tensor, padinstruction)
        self.assertTrue(torch.all(expected == got))
    def test_cases(self):


        #Todo: Do5444444445


class test_circular_padding_errors(unittest.TestCase):
    """
    Test that the circular padding mechanism is making sane errors.
    """