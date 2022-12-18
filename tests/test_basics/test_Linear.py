import unittest

import torch
from src.supertransformerlib.Basics import linear
PRINT_ERRORS = True

class testForwardWithBias(unittest.TestCase):
    """
    Test that the linear function call works properly.
    """
    def test_basic_linear(self):
        """Tests the basic linear function operates"""
        tensor = torch.randn([10])
        kernel = torch.randn([10, 5])
        bias = torch.randn([5])
        expected_shape = torch.Size([5])

        output = linear._linear_forward(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)
    def test_batched_linear(self):
        """Test the linear operator works when handling batches"""
        tensor = torch.randn([15, 10])
        kernel = torch.randn([10, 5])
        bias = torch.randn([5])
        expected_shape = torch.Size([15, 5])

        output = linear._linear_forward(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)

    def test_multibatched_linear(self):
        """Test the linear operator works when handling arbitrary batches"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([10, 5])
        bias = torch.randn([5])
        expected_shape = torch.Size([40, 20, 15, 5])

        output = linear._linear_forward(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)
    def test_parallel_kernel_linear(self):
        """Test the linear operators broadcasts across parallel kernels properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        output = linear._linear_forward(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)
    def test_torchscript_functional_linear(self):
        """Test the linear operators torchscript compile properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        function = torch.jit.script(linear._linear_forward)
        output = function(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)

class testForwardWithoutBias(unittest.TestCase):
    """
    Test that the forward mechanism operates correctly
    even without bias.
    """
    def test_basic_linear(self):
        """Tests the basic linear function operates"""
        tensor = torch.randn([10])
        kernel = torch.randn([10, 5])
        expected_shape = torch.Size([5])

        output = linear._linear_forward(tensor, kernel)
        self.assertTrue(output.shape == expected_shape)

    def test_batched_linear(self):
        """Test the linear operator works when handling batches"""
        tensor = torch.randn([15, 10])
        kernel = torch.randn([10, 5])
        expected_shape = torch.Size([15, 5])

        output = linear._linear_forward(tensor, kernel)
        self.assertTrue(output.shape == expected_shape)

    def test_multibatched_linear(self):
        """Test the linear operator works when handling arbitrary batches"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([10, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        output = linear._linear_forward(tensor, kernel)
        self.assertTrue(output.shape == expected_shape)
    def test_parallel_kernel_linear(self):
        """Test the linear operators broadcasts across parallel kernels properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        output = linear._linear_forward(tensor, kernel)
        self.assertTrue(output.shape == expected_shape)

class test_linear_exceptions(unittest.TestCase):
    """
    Test the validation mechanisms descently well
    """
    def test_wrong_linear_dim(self):
        """
        Test a reasonable error occurs when the last dimension is the
        wrong dynamic_shape for the kernel
        """
        tensor = torch.randn([40, 20, 15, 7])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_error = linear.LinearForwardException

        try:
            closure = linear.linear_forward(tensor, kernel, bias)
            closure(tensor)
            raise RuntimeError("No error when there should be")
        except expected_error as err:
            if PRINT_ERRORS:
                print(err)

    def test_wrong_dtype(self):
        """
        Test a reasonable error occurs when the tensor
        and kernel do not match.
        """
        tensor = torch.randn([40, 20, 15, 7]).to(torch.complex64)
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_error = linear.LinearForwardException

        try:
            linear.linear_forward(tensor, kernel, bias)
            raise RuntimeError("No error when there should be")
        except expected_error as err:
            if PRINT_ERRORS:
                print(err)

    def test_wrong_parallel_dim(self):
        """
        Test a reasonable error occurs when the parallelization
        dimensions do not match
        """
        tensor = torch.randn([40, 20, 4, 10])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_error = linear.LinearForwardException

        try:
            linear.linear_forward(tensor, kernel, bias)
            raise RuntimeError("No error when there should be")
        except expected_error as err:
            if PRINT_ERRORS:
                print(err)

class testKernelConstruct(unittest.TestCase):
    """Test the ability to construct the required linear kernels"""
    def test_basic_sane(self):
        """Test kernel maker is making something sane"""
        input_shape = torch.tensor(10)
        output_shape = torch.tensor(5)
        parallel = None
        expected_shape = torch.Size([10, 5])

        parameter = linear.Linear.make_kernel(input_shape, output_shape, parallel, None, None)
        self.assertTrue(parameter.shape == expected_shape)

    def test_complicated(self):
        """Test that initialization works when far more complicated."""
        input_shape = torch.tensor([30, 20])
        output_shape = torch.tensor(5)
        parallel = torch.tensor([3, 4])
        expected_shape = torch.Size([3, 4, 600, 5])

        parameter = linear.Linear.make_kernel(input_shape, output_shape, parallel, None, None)
        self.assertTrue(parameter.shape == expected_shape)

class testBiasConstruct(unittest.TestCase):
    """Test the ability to construct the required linear kernels"""
    def test_basic_sane(self):
        """Test kernel maker is making something sane"""
        output_shape = torch.tensor(5)
        parallel = None
        expected_shape = torch.Size([5])

        parameter = linear.Linear.make_bias(output_shape, parallel, None, None)
        self.assertTrue(parameter.shape == expected_shape)

    def test_complicated(self):
        """Test that initialization works when far more complicated."""
        output_shape = torch.tensor(5)
        parallel = torch.tensor([3, 4])
        expected_shape = torch.Size([3, 4, 5])

        parameter = linear.Linear.make_bias(output_shape, parallel, None, None)
        self.assertTrue(parameter.shape == expected_shape)


class testLinear(unittest.TestCase):
    """
    Tests the linear layer.
    """

    def test_basic(self):
        """Tests a basic linear layer"""
        tensor = torch.randn([10])
        expected_shape = torch.Size([5])

        layer = linear.Linear(10, 5)
        layer = torch.jit.script(layer)
        output = layer(tensor)

        self.assertTrue(output.shape == expected_shape)

    def test_reshaping(self):
        """Test that the reshaping mechanism is working properly"""

        tensor = torch.randn([10, 3, 5, 6])
        input_shape = [3, 5, 6]
        output_shape = [3, 2]

        expected_shape = torch.Size([10, 3,2])
        layer = linear.Linear(input_shape, output_shape)
        layer = torch.jit.script(layer)

        output = layer(tensor)
        self.assertTrue(expected_shape == output.shape)


    def test_parallel_utilization(self):
        """Tests usage on parallel tensors"""
        tensor = torch.randn([7, 5, 4, 6, 10])
        keywords = {
            'input_shape' : 10,
            'output_shape' : 5,
            'parallel' : [5, 4, 6]
        }
        expected_shape = torch.Size([7, 5, 4, 6, 5])

        layer = linear.Linear(**keywords)
        layer = torch.jit.script(layer)

        output = layer(tensor)
        self.assertTrue(output.shape == expected_shape)



