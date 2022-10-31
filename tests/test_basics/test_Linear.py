import unittest

import torch
from torch import nn
from src.supertransformerlib.Basics import Linear
from src.supertransformerlib.Core import SparseUtils
from src.supertransformerlib import Core
from src.supertransformerlib.Core import Reshape
print_errors = True

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

        output = Linear._linear_forward(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)
    def test_batched_linear(self):
        """Test the linear operator works when handling batches"""
        tensor = torch.randn([15, 10])
        kernel = torch.randn([10, 5])
        bias = torch.randn([5])
        expected_shape = torch.Size([15, 5])

        output = Linear._linear_forward(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)

    def test_multibatched_linear(self):
        """Test the linear operator works when handling arbitrary batches"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([10, 5])
        bias = torch.randn([5])
        expected_shape = torch.Size([40, 20, 15, 5])

        output = Linear._linear_forward(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)
    def test_parallel_kernel_linear(self):
        """Test the linear operators broadcasts across parallel kernels properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        output = Linear._linear_forward(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)
    def test_torchscript_functional_linear(self):
        """Test the linear operators torchscript compile properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        function = torch.jit.script(Linear._linear_forward)
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

        output = Linear._linear_forward(tensor, kernel)
        self.assertTrue(output.shape == expected_shape)

    def test_batched_linear(self):
        """Test the linear operator works when handling batches"""
        tensor = torch.randn([15, 10])
        kernel = torch.randn([10, 5])
        expected_shape = torch.Size([15, 5])

        output = Linear._linear_forward(tensor, kernel)
        self.assertTrue(output.shape == expected_shape)

    def test_multibatched_linear(self):
        """Test the linear operator works when handling arbitrary batches"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([10, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        output = Linear._linear_forward(tensor, kernel)
        self.assertTrue(output.shape == expected_shape)
    def test_parallel_kernel_linear(self):
        """Test the linear operators broadcasts across parallel kernels properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        output = Linear._linear_forward(tensor, kernel)
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
        expected_error = Linear.LinearForwardException

        try:
            closure = Linear.linear_forward(tensor, kernel, bias)
            output = closure(tensor)
            raise RuntimeError("No error when there should be")
        except expected_error as err:
            if print_errors:
                print(err)

    def test_wrong_dtype(self):
        """
        Test a reasonable error occurs when the tensor
        and kernel do not match.
        """
        tensor = torch.randn([40, 20, 15, 7]).to(torch.complex64)
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_error = Linear.LinearForwardException

        try:
            output = Linear.linear_forward(tensor, kernel, bias)
            raise RuntimeError("No error when there should be")
        except expected_error as err:
            if print_errors:
                print(err)

    def test_wrong_parallel_dim(self):
        """
        Test a reasonable error occurs when the parallelization
        dimensions do not match
        """
        tensor = torch.randn([40, 20, 4, 10])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_error = Linear.LinearForwardException

        try:
            closure = Linear.linear_forward(tensor, kernel, bias)
            raise RuntimeError("No error when there should be")
        except expected_error as err:
            if print_errors:
                print(err)

class testKernelConstruct(unittest.TestCase):
    """Test the ability to construct the required linear kernels"""
    def test_basic_sane(self):
        """Test kernel maker is making something sane"""
        input_shape = torch.tensor(10)
        output_shape = torch.tensor(5)
        parallel = None
        dynamic = None
        expected_shape = torch.Size([10, 5])

        parameter = Linear.make_kernel(input_shape, output_shape, parallel, dynamic, None, None)
        self.assertTrue(torch.Size(parameter.Kernel_Shape) == expected_shape)
    def test_complicated(self):
        """Test that initialization works when far more complicated."""
        input_shape = torch.tensor([30, 20])
        output_shape = torch.tensor(5)
        parallel = torch.tensor([3, 4])
        dynamic = torch.tensor([6, 7])
        expected_shape = torch.Size([3, 4, 600, 5])

        parameter = Linear.make_kernel(input_shape, output_shape, parallel, dynamic, None, None)

        self.assertTrue(torch.Size(parameter.Kernel_Shape) == expected_shape)

class testBiasConstruct(unittest.TestCase):
    """Test the ability to construct the required linear kernels"""
    def test_basic_sane(self):
        """Test kernel maker is making something sane"""
        output_shape = torch.tensor(5)
        parallel = None
        dynamic = None
        expected_shape = torch.Size([5])

        Parameter = Linear.make_bias(output_shape, parallel, dynamic, None, None)
        self.assertTrue(torch.Size(Parameter.Kernel_Shape) == expected_shape)

    def test_complicated(self):
        """Test that initialization works when far more complicated."""
        output_shape = torch.tensor(5)
        parallel = torch.tensor([3, 4])
        dynamic = torch.tensor([6, 7])
        expected_shape = torch.Size([3, 4, 5])

        Parameter = Linear.make_bias(output_shape, parallel, dynamic, None, None)
        self.assertTrue(torch.Size(Parameter.Kernel_Shape) == expected_shape)


class testLinearErrors(unittest.TestCase):
    def test_missing_dynamic(self):
        """Test that when a dynamic tensor is not passed, and we need it, we throw"""
        tensor = torch.randn([10])
        layer = Linear.Linear(10, 10, dynamic=10)
        try:
            layer(tensor)
            raise RuntimeError("Did not throw")
        except Core.KernelSetupError as err:
            if print_errors:
                print(err)

    def test_extra_dynamic(self):
        """Test that when a dynamic tensor is passed, and it is not needed, we throw"""
        tensor = torch.randn([15, 10])
        layer = Linear.Linear(10, 10)
        try:
            layer(tensor, torch.randn([20]))
            raise RuntimeError("Did not throw")
        except Core.KernelSetupError as err:
            if print_errors:
                print(err)

    def test_insufficient_rank(self):
        """Test we throw when the dynamic tensor is too small"""
        tensor = torch.randn([30, 10])
        layer = Linear.Linear(10, 10, dynamic=[20, 20, 20])
        dynamic = torch.randn([10])

        try:
            layer(tensor, dynamic)
            raise RuntimeError("Did not throw")
        except Core.KernelSetupError as err:
            if print_errors:
                print(err)

    def test_dynamic_wrong_shape(self):
        """Test that we throw when the dynamic tensor is shaped incorrectly"""
        layer = Linear.Linear(10, 10, dynamic = 4)
        dynamic = torch.rand(3, 20, 10)
        tensor = torch.randn([20, 10])
        try:
            layer(tensor, dynamic)
            raise RuntimeError("Did not throw")
        except Core.KernelSetupError as err:
            if print_errors:
                print(err)

    def test_bad_sparse_dim_num(self):
        """
        Test that when we pass a sparse dimension, it is the case incorrectly
        configured hybrid tensors throw an error
        """

        layer = Linear.Linear(10, 10, dynamic=4)

        tensor = torch.randn([20, 10])
        dynamic = torch.randn([4, 20, 10])
        mask = dynamic > 0
        indices = SparseUtils.gen_indices_from_mask(mask)
        values = dynamic[mask]
        sparse_dynamic = torch.sparse_coo_tensor(indices, values)

        try:
            layer(tensor, sparse_dynamic)
            raise RuntimeError("Did not throw")
        except Core.ReshapeException as err:
            if print_errors:
                print(err)


class testLinear(unittest.TestCase):
    """
    Tests the linear factory layer. It emits
    linear closures when called.
    """

    def test_basic(self):
        """Tests a basic linear layer"""
        tensor = torch.randn([10])
        expected_shape = torch.Size([5])

        layer = Linear.Linear(10, 5)
        layer = torch.jit.script(layer)
        output = layer(tensor)

        self.assertTrue(output.shape == expected_shape)

    def test_reshaping(self):
        """Test that the reshaping mechanism is working properly"""

        tensor = torch.randn([10, 3, 5, 6])
        input_shape = [3, 5, 6]
        output_shape = [3, 2]

        expected_shape = torch.Size([10, 3,2])
        layer = Linear.Linear(input_shape, output_shape)
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

        layer = Linear.Linear(**keywords)
        layer = torch.jit.script(layer)
        output = layer(tensor)
        self.assertTrue(output.shape == expected_shape)

    def test_superposition_dense(self):
        """ Test that superposition works properly when doing dense superposition"""

        """Tests usage on parallel tensors"""
        tensor = torch.randn([7, 5, 4, 6, 10])
        keywords = {
            'input_shape': 10,
            'output_shape': 5,
            'parallel': [5, 4, 6],
            'dynamic' : 11
        }
        expected_shape = torch.Size([7, 5, 4, 6, 5])

        dynamic = torch.rand([11])
        layer = Linear.Linear(**keywords)
        layer = torch.jit.script(layer)
        output = layer(tensor, dynamic)
        self.assertTrue(output.shape == expected_shape)

    def test_superposition_extra_dims(self):
        """
        Test that increasing the number of dims when building a superposition for
        batch purposes works properly
        """
        tensor = torch.randn([7, 5, 4, 6, 10])
        keywords = {
            'input_shape': 10,
            'output_shape': 5,
            'parallel': [5, 4, 6],
            'dynamic' : 11
        }
        expected_shape = torch.Size([7, 5, 4, 6, 5])

        dynamic = torch.rand([11, 7])
        layer = Linear.Linear(**keywords)
        layer = torch.jit.script(layer)
        output = layer(tensor, dynamic)
        self.assertTrue(output.shape == expected_shape)

    def test_sparse_superposition(self):
        """Test a sparse configuration works properly."""
        tensor = torch.randn([7, 5, 4, 6, 10])
        keywords = {
            'input_shape': 10,
            'output_shape': 5,
            'parallel': [5, 4, 6],
            'dynamic' : [11, 7]
        }
        expected_shape = torch.Size([7, 5, 4, 6, 5])

        dynamic = torch.rand([11, 7])
        mask = torch.rand([11, 7]) > 0.5
        index = SparseUtils.gen_indices_from_mask(mask)
        values = dynamic[mask]
        superposition_weights = torch.sparse_coo_tensor(index, values)

        layer = Linear.Linear(**keywords)
        layer = torch.jit.script(layer)
        output = layer(tensor, superposition_weights)
        self.assertTrue(output.shape == expected_shape)

    def test_configured_dynamic_superposition(self):
        """ Test that configuration by another layer doesn't throw any errors"""
        batch_shape = [7, 7]
        input_dim = 5
        output_dim = 6
        dynamic = 12

        test_data = torch.randn([*batch_shape, input_dim])
        class Dynamic_Linear_Configuration(nn.Module):

            def __init__(self):
                super().__init__()
                self.configuration_layer = Linear.Linear(input_dim, dynamic)
                self.execution_layer = Linear.Linear(input_dim, output_dim, dynamic=dynamic)
            def forward(self, tensor: torch.Tensor)->torch.Tensor:
                configuration = self.configuration_layer(tensor)
                configuration = configuration.movedim(-1, 0) #Notice the move required to place the dynamic dim to the front.
                output = self.execution_layer(tensor, configuration)
                return output

        instance = Dynamic_Linear_Configuration()
        instance = torch.jit.script(instance) #Optional line. Scripts it
        output = instance(test_data)
