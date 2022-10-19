"""

Test features for the convolutional
kernel sampling functions and features

"""
import unittest
import torch
import itertools
from src.supertransformerlib.Basics import ConvolutionalSample
from typing import List

def native_example_generator(test_list: torch.Tensor, start: int, end: int,  stride: int,  dilation: int, offset: int):
    """
    Manually generates a native view equivalent, in a not-very-efficient loop.
    Used for testing and for giving an idea as to how the damn view shoud
    work.

    Works in 1d, on the last dimension.
    """

    output: List[torch.Tensor] = []
    iterater = range(0, test_list.shape[-1], stride)
    for i in iterater:
        options = torch.range(start, end).to(dtype=torch.int64)*dilation + i + offset
        if torch.all(0 <= options) and torch.all(options < test_list.shape[-1]):
            sample = test_list[..., options]
            output.append(sample)
    if len(output) > 0:
        return torch.stack(output, -2)
    else:
        shape = [*test_list.shape[:-1], 0, end-start + 1]
        return torch.empty(shape, dtype = test_list.dtype)




class test_native_cases(unittest.TestCase):
    """
    Uses the native test case generator to test a variety of different cases with
    different specifications
    """
    def test_particular(self):

        tensor = torch.arange(10).view(1, 10)

        expected = native_example_generator(tensor, -3, 0, 2, 1, 0)
        output = ConvolutionalSample.convolutional_sample(tensor, -3, 0, 2, 1, 0, mode="native")

        print(expected)
        print(output)
        print(output.shape)

    def test_run(self):

        tensor = torch.arange(200).view(10, 20)
        starts = [-3, -2, -1, 0]
        ends = [0, 1, 2, 3, 4]
        strides = [1, 2, 3, 4]
        dilations = [1, 2, 3, 4]
        offsets = [-1, 0, 1]

        options = itertools.product(starts, ends, strides, dilations, offsets)

        for option in options:
            expected = native_example_generator(tensor, *option)
            got = ConvolutionalSample.convolutional_sample(tensor, *option, mode="native")
            try:
                self.assertTrue(torch.all(expected == got))
            except Exception as err:
                print("failure with combo")
                print(option)
                raise err



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
        start = torch.tensor([-1])
        end = torch.tensor([1])
        dilation = torch.tensor([1])
        stride = torch.tensor([1])
        offset = torch.tensor([0])
        mode = "native"

        expected_shape = torch.tensor([3, 3])
        expected_tensor = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4]])

        output = ConvolutionalSample.convolutional_sample(tensor, start, end,
                                                          dilation, stride, offset,
                                                          mode)
        self.assertTrue(output.shape == torch.Size(expected_shape))
        self.assertTrue(torch.all(output == expected_tensor))

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



class test_native_errors(unittest.TestCase):
    """
    Test that native convolution throws appropriate
    errors when called.
    """
    def test_tensor_rank_insufficient(self):
        pass
    def test_tensor_dim_not_large_enough_simple(self):
        pass
    def test_tensor_dim_not_large_enough_dilated(self):
        pass
    def test_tensor_dim_not_large_enough_2d(self):
        pass