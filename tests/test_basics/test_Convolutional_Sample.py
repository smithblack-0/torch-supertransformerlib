"""

Test features for the convolutional
kernel sampling functions and features

"""
import unittest
import torch
from src.supertransformerlib.Basics import ConvolutionalSample

class test_simple_convolutional(unittest.TestCase):
    """
    Test that the function which runs sampling correctly
    handles the simple cases involving the three different
    modes available.

    simple cases are cases which are one dimensional with
    no batching.
    """
    def test_simple_sampling_case(self):
        """Test that a straightforward sampling proceeds as expected"""
        tensor = torch.arange(5)
        start = torch.tensor(-1)
        end = torch.tensor(1)
        dilation = torch.tensor(1)
        stride = torch.tensor(1)
        offset = torch.tensor(0)
        mode = "native"

        expected_shape = torch.tensor([3, 3])
        expected_tensor = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4]])

        output = ...
        self.assertTrue(output.shape == torch.Size(expected_shape))
        self.assertTrue(output == expected_tensor)

    def test_simple_padded_case(self):
        """Test that a straightforward sampling with padding proceeds as expected """

        tensor = torch.arange(5)
        start = torch.tensor(-1)
        end = torch.tensor(1)
        dilation = torch.tensor(1)
        stride = torch.tensor(1)
        offset = torch.tensor(0)
        mode = "pad"

        expected_shape = torch.tensor([5, 3])
        expected_tensor = torch.tensor([[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0]])

        output = ...
        self.assertTrue(output.shape == torch.Size(expected_shape))
        self.assertTrue(output == expected_tensor)

    def test_simple_rollover_case(self):
        """Test that a straightforward sampling with padding proceeds as expected """

        tensor = torch.arange(5)
        start = torch.tensor(-1)
        end = torch.tensor(1)
        dilation = torch.tensor(1)
        stride = torch.tensor(1)
        mode = "rollover"

        expected_shape = torch.tensor([5, 3])
        expected_tensor = torch.tensor([[4, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0]])

        output = ...
        self.assertTrue(output.shape == torch.Size(expected_shape))
        self.assertTrue(output == expected_tensor)

    def test_simple_dilated_case_continous(self):
        """Test the behavior with a simple dilation is proper"""
        tensor = torch.arange(5)
        start = torch.tensor(-1)
        end = torch.tensor(1)
        dilation = torch.tensor(2)
        stride = torch.tensor(1)
        offset = torch.tensor(0)
        mode = "rollover"

        expected_shape = torch.tensor([5, 3])
        expected_tensor = torch.tensor([[3, 0, 2], [4, 1, 3], [0, 2, 4], [1, 3,0], [2, 4,1]])

        output = ...
        self.assertTrue(output.shape == torch.Size(expected_shape))
        self.assertTrue(output == expected_tensor)

    def test_simple_dilated_case_native(self):
        """Test the behavior with a simple dilation is proper"""
        tensor = torch.arange(5)
        start = torch.tensor(-1)
        end = torch.tensor(1)
        dilation = torch.tensor(2)
        stride = torch.tensor(1)
        offset = torch.tensor(0)
        mode = "native"

        expected_shape = torch.tensor([1, 3])
        expected_tensor = torch.tensor([[0, 2, 4]])

        output = ...
        self.assertTrue(output.shape == torch.Size(expected_shape))
        self.assertTrue(output == expected_tensor)

    def test_simple_strided_case(self):
        """Test the behavior with a simple stride is proper"""
        tensor = torch.arange(5)
        start = torch.tensor(-1)
        end = torch.tensor(1)
        dilation = torch.tensor(1)
        stride = torch.tensor(2)
        offset = torch.tensor(0)
        mode = "rollover"

        expected_shape = torch.tensor([3, 3])
        expected_tensor = torch.tensor([[4, 0, 1], [2, 3, 4], [4, 0, 1]])

        output = ...
        self.assertTrue(output.shape == torch.Size(expected_shape))
        self.assertTrue(output == expected_tensor)

    def test_simple_strided_case_native(self):
        """Test the behavior with a simple stride is proper"""
        tensor = torch.arange(5)
        start = torch.tensor(-1)
        end = torch.tensor(1)
        dilation = torch.tensor(1)
        stride = torch.tensor(2)
        offset = torch.tensor(0)
        mode = "native"

        expected_shape = torch.tensor([3, 3])
        expected_tensor = torch.tensor([[0, 1, 2], [2, 3,4]])

        output = ...
        self.assertTrue(output.shape == torch.Size(expected_shape))
        self.assertTrue(output == expected_tensor)

    def test_simple_offset_continous(self):
        """ Test the behavior is sane when using a simple offset in a continous manner """
        tensor = torch.arange(5)
        start = torch.tensor(-1)
        end = torch.tensor(1)
        dilation = torch.tensor(1)
        stride = torch.tensor(1)
        offset = torch.tensor(1)
        mode = "rollover"

        expected_shape = torch.tensor([3, 3])
        expected_tensor = torch.tensor([[0, 1, 2], [1, 2,3], [2, 3, 4], [3, 4, 0], [4, 0, 1]])

        output = ...
        self.assertTrue(output.shape == torch.Size(expected_shape))
        self.assertTrue(output == expected_tensor)

    def test_simple_offset_native(self):
        """ Test the behavior is sane when using a simple offset in a continous manner """
        tensor = torch.arange(5)
        start = torch.tensor(-1)
        end = torch.tensor(1)
        dilation = torch.tensor(1)
        stride = torch.tensor(1)
        offset = torch.tensor(1)
        mode = "native"

        expected_shape = torch.tensor([2, 3])
        expected_tensor = torch.tensor([[1,2,3], [2, 3, 4]])

        output = ...
        self.assertTrue(output.shape == torch.Size(expected_shape))
        self.assertTrue(output == expected_tensor)



class test_convolutional_sample_