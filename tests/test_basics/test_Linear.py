import unittest

import torch
from torch import nn
from src.supertransformerlib import Glimpses
from src.supertransformerlib.Basics import Linear
from src.supertransformerlib.Core import SparseUtils

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

        output = Linear.linear_forward(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)
    def test_batched_linear(self):
        """Test the linear operator works when handling batches"""
        tensor = torch.randn([15, 10])
        kernel = torch.randn([10, 5])
        bias = torch.randn([5])
        expected_shape = torch.Size([15, 5])

        output = Linear.linear_forward(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)

    def test_multibatched_linear(self):
        """Test the linear operator works when handling arbitrary batches"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([10, 5])
        bias = torch.randn([5])
        expected_shape = torch.Size([40, 20, 15, 5])

        output = Linear.linear_forward(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)
    def test_parallel_kernel_linear(self):
        """Test the linear operators broadcasts across parallel kernels properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        output = Linear.linear_forward(tensor, kernel, bias)
        self.assertTrue(output.shape == expected_shape)
    def test_torchscript_functional_linear(self):
        """Test the linear operators torchscript compile properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        function = torch.jit.script(Linear.linear_forward)
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

        output = Linear.linear_forward(tensor, kernel)
        self.assertTrue(output.shape == expected_shape)

    def test_batched_linear(self):
        """Test the linear operator works when handling batches"""
        tensor = torch.randn([15, 10])
        kernel = torch.randn([10, 5])
        expected_shape = torch.Size([15, 5])

        output = Linear.linear_forward(tensor, kernel)
        self.assertTrue(output.shape == expected_shape)

    def test_multibatched_linear(self):
        """Test the linear operator works when handling arbitrary batches"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([10, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        output = Linear.linear_forward(tensor, kernel)
        self.assertTrue(output.shape == expected_shape)
    def test_parallel_kernel_linear(self):
        """Test the linear operators broadcasts across parallel kernels properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        output = Linear.linear_forward(tensor, kernel)
        self.assertTrue(output.shape == expected_shape)

class testLinearClosure(unittest.TestCase):
    """
    Test the linear closure when this closure is suppose
    to work fine.
    """
    def test_parallel_kernel_linear_nomap(self):
        """Test the linear operators broadcasts across parallel kernels properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        closure = Linear.LinearClosure(kernel, bias)
        output = closure(tensor)
        self.assertTrue(output.shape == expected_shape)

    def test_multi_to_single_remap(self):
        """Test that a remap occurs correctly when called."""

        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([150, 5])
        bias = torch.randn([5])
        input_map = Glimpses.ReshapeClosure([15, 10], 150)
        output_map = None
        expected_shape = torch.Size([40, 20, 5])

        closure = Linear.LinearClosure(kernel, bias, input_map, output_map)
        output = closure(tensor)
        self.assertTrue(output.shape == expected_shape)

    def test_script(self):
        func = torch.jit.script(Linear.LinearClosure)

class testClosureExceptions(unittest.TestCase):
    def test_wrong_linear_dim(self):
        """
        Test a reasonable error occurs when the last dimension is the
        wrong shape for the kernel
        """
        tensor = torch.randn([40, 20, 15, 7])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_error = Linear.LinearForwardException

        try:
            closure = Linear.LinearClosure(kernel, bias)
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
            closure = Linear.LinearClosure(kernel, bias)
            output = closure(tensor)
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
            closure = Linear.LinearClosure(kernel, bias)
            output = closure(tensor)
            raise RuntimeError("No error when there should be")
        except expected_error as err:
            if print_errors:
                print(err)





class testKernelConstruct(unittest.TestCase):
    """Test the ability to construct the required linear kernels"""
    def test_basic_sane(self):
        """Test kernel maker is making something sane"""
        tensor = torch.randn([10])
        input_shape = torch.tensor(10)
        output_shape = torch.tensor(5)
        parallel = torch.empty([0], dtype=torch.int64)
        dynamic = torch.empty([0],dtype=torch.int64)
        expected_shape = torch.Size([10, 5])

        kernel = Linear.make_kernel(input_shape, output_shape, parallel, dynamic, None, None)
        Linear.linear_forward(tensor, kernel)
        self.assertTrue(kernel.shape == expected_shape)
    def test_complicated(self):
        """Test that initialization works when far more complicated."""
        input_shape = torch.tensor([30, 20])
        output_shape = torch.tensor(5)
        parallel = torch.tensor([3, 4])
        dynamic = torch.tensor([6, 7])
        expected_shape = torch.Size([6, 7, 3, 4, 600, 5])

        kernel = Linear.make_kernel(input_shape, output_shape, parallel, dynamic, None, None)
        self.assertTrue(kernel.shape == expected_shape)

class testBiasConstruct(unittest.TestCase):
    """Test the ability to construct the required linear kernels"""
    def test_basic_sane(self):
        """Test kernel maker is making something sane"""
        output_shape = torch.tensor(5)
        parallel = torch.empty([0], dtype=torch.int64)
        dynamic = torch.empty([0],dtype=torch.int64)
        expected_shape = torch.Size([5])

        kernel = Linear.make_bias(output_shape, parallel, dynamic, None, None)
        self.assertTrue(kernel.shape == expected_shape)

    def test_complicated(self):
        """Test that initialization works when far more complicated."""
        output_shape = torch.tensor(5)
        parallel = torch.tensor([3, 4])
        dynamic = torch.tensor([6, 7])
        expected_shape = torch.Size([6, 7, 3, 4, 5])

        kernel = Linear.make_bias(output_shape, parallel, dynamic, None, None)
        self.assertTrue(kernel.shape == expected_shape)

class testKernelSuperposition(unittest.TestCase):
    """
    Test that the kernel superposition
    function will work properly no matter
    the circumstances.
    """

    def test_dense_superposition(self):

        kernel = torch.randn([10, 20, 30, 5, 4])
        dynamics = torch.randn([7, 8, 10, 20])
        dynamic_shape = torch.tensor([10, 20])
        expected_shape = torch.Size([7, 8, 30, 5, 4])

        output = Linear.make_dense_superposition(dynamics, kernel, dynamic_shape)
        self.assertTrue(output.shape == expected_shape)

    def test_sparse_superposition(self):
        """Test sparse superposition builds correctly., test autograd still works"""
        kernel = torch.randn([10, 20, 30, 5, 4], requires_grad=True)
        dynamics = torch.randn([7, 8, 10, 20], requires_grad=True)
        dynamic_shape = torch.tensor([10, 20])

        mask = torch.randn([10, 20]) > 0
        indices = SparseUtils.gen_indices_from_mask(mask)
        values = dynamics[..., mask].movedim(-1, 0)
        hybrid_shape = [10, 20, 7, 8]
        sparse_dynamics = torch.sparse_coo_tensor(indices, values, size=hybrid_shape).coalesce()

        expected_shape = torch.Size([7, 8, 30, 5, 4])
        output = Linear.make_sparse_superposition(sparse_dynamics, kernel, dynamic_shape)
        self.assertTrue(output.shape == expected_shape)
        self.assertTrue(output.is_sparse == False)

        output.sum().backward()
        self.assertTrue(kernel.grad != None)
        self.assertTrue(dynamics.grad != None)

class testLinearErrors(unittest.TestCase):
    def test_missing_dynamic(self):
        """Test that when a dynamic tensor is not passed, and we need it, we throw"""
        layer = Linear.Linear(10, 10, dynamic=10)
        try:
            layer()
            raise RuntimeError("Did not throw")
        except Linear.LinearFactoryException as err:
            if print_errors:
                print(err)

    def test_extra_dynamic(self):
        """Test that when a dynamic tensor is passed, and it is not needed, we throw"""
        layer = Linear.Linear(10, 10)
        try:
            layer(torch.randn([20]))
            raise RuntimeError("Did not throw")
        except Linear.LinearFactoryException as err:
            if print_errors:
                print(err)

    def test_insufficient_rank(self):
        """Test we throw when the dynamic tensor is too small"""
        layer = Linear.Linear(10, 10, dynamic=[20, 20, 20])
        dynamic = torch.randn([10])

        try:
            layer(dynamic)
            raise RuntimeError("Did not throw")
        except Linear.LinearFactoryException as err:
            if print_errors:
                print(err)

    def test_dynamic_wrong_shape(self):
        """Test that we throw when the dynamic tensor is shaped incorrectly"""
        layer = Linear.Linear(10, 10, dynamic = 4)
        dynamic = torch.rand(3, 20, 10)

        try:
            layer(dynamic)
            raise RuntimeError("Did not throw")
        except Linear.LinearFactoryException as err:
            if print_errors:
                print(err)

    def test_bad_sparse_dim_num(self):
        """
        Test that when we pass a sparse dimension, it is the case incorrectly
        configured hybrid tensors throw an error
        """

        layer = Linear.Linear(10, 10, dynamic=4)

        dynamic = torch.randn([4, 20, 10])
        mask = dynamic > 0
        indices = SparseUtils.gen_indices_from_mask(mask)
        values = dynamic[mask]
        sparse_dynamic = torch.sparse_coo_tensor(indices, values)

        try:
            layer(sparse_dynamic)
            raise RuntimeError("Did not throw")
        except Linear.LinearFactoryException as err:
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
        closure = layer()
        output = closure(tensor)

        self.assertTrue(output.shape == expected_shape)

    def test_basic_torchscript_passing(self):
        """Tests the ability to pass a closure around"""
        tensor = torch.randn([10])
        expected_shape = torch.Size([5])

        @torch.jit.script
        def do_linear(tensor: torch.Tensor, closure: Linear.LinearClosure):
            return closure(tensor)

        layer = Linear.Linear(10, 5)
        closure = layer()
        output = do_linear(tensor, closure)

        self.assertTrue(output.shape == expected_shape)
    def test_reshaping(self):
        """Test that the reshaping mechanism is working properly"""

        tensor = torch.randn([3, 5, 6])
        input_shape = [3, 5, 6]
        output_shape = [3, 2]

        layer = Linear.Linear(input_shape, output_shape)
        closure = layer()
        output = closure(tensor)


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
        closure = layer()
        output = closure(tensor)
        self.assertTrue(output.shape == expected_shape)

    def test_dynamic_superposition(self):
        """ Test that dynamic superposition works properly"""

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
        closure = layer(dynamic)
        output = closure(tensor)
        self.assertTrue(output.shape == expected_shape)
    def test_torchscript(self):
        """ Test that the linear system torchscript compiles"""
        data = torch.randn([15, 10])
        instance = Linear.Linear(10, 20)
        instance = torch.jit.script(instance)
        closure=  instance()
        closure(data)
    def test_configured_dynamic_superposition(self):
        """ Test that configuration by another layer doesn't throw any errors"""

        test_data = torch.randn([11, 10])

        class Dynamic_Linear_Configuration(nn.Module):

            def __init__(self):
                super().__init__()
                self.configuration_layer = Linear.Linear(10, 12)
                self.execution_layer = Linear.Linear(10, 10, dynamic=12)
            def forward(self, tensor: torch.Tensor)->torch.Tensor:
                configuration = self.configuration_layer()(tensor)
                output = self.execution_layer(configuration)(tensor)
                return output

        instance = Dynamic_Linear_Configuration()
        instance = torch.jit.script(instance) #Optional line. Scripts it
        result = instance(test_data)