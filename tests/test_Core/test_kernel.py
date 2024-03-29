"""
A simple test collection to test the
kernel subsection of core.
"""



import unittest

import torch
from torch.nn import init
from supertransformerlib import Core

PRINT_ERRORS = True

def print_the_error(error):
    """ Prints a caught error """
    print("--- printing a caught error for message inspection. See following ---")
    print(error)

class TestKernelSuperposition(unittest.TestCase):
    """
    Test that the kernel superposition
    function will work properly no matter
    the circumstances.
    """

    def test_dense_superposition(self):
        """Test that dense superpositions operate correctly"""

        kernel = torch.randn([10, 20, 30, 5, 4])
        dynamics = torch.randn([10, 20, 7, 8])
        dynamic_shape = torch.tensor([10, 20])
        expected_shape = torch.Size([7, 8, 30, 5, 4])


        #Standard

        output = Core.make_dense_superposition(dynamics, kernel, dynamic_shape)
        self.assertTrue(output.shape == expected_shape)

        #Torchscript

        function = torch.jit.script(Core.make_dense_superposition)
        output = function(dynamics, kernel, dynamic_shape)
        self.assertTrue(output.shape == expected_shape)

    def test_sparse_superposition(self):
        """Test sparse superposition builds correctly., test autograd still works"""
        kernel = torch.randn([10, 20, 30, 5, 4], requires_grad=True)
        dynamics = torch.randn([7, 8, 10, 20], requires_grad=True)
        dynamic_shape = torch.tensor([10, 20])

        mask = torch.randn([10, 20]) > 0
        indices = Core.gen_indices_from_mask(mask)
        values = dynamics[..., mask].movedim(-1, 0)
        hybrid_shape = [10, 20, 7, 8]
        sparse_dynamics = torch.sparse_coo_tensor(indices, values, size=hybrid_shape).coalesce()

        expected_shape = torch.Size([7, 8, 30, 5, 4])

        #Standard

        output = Core.make_sparse_superposition(sparse_dynamics, kernel, dynamic_shape, None)
        self.assertTrue(output.shape == expected_shape)
        self.assertTrue(output.is_sparse is False)

        output.sum().backward()
        self.assertTrue(kernel.grad is not None)
        self.assertTrue(dynamics.grad is not None)

        #Torchscript

        mask = torch.randn([10, 20]) > 0
        indices = Core.gen_indices_from_mask(mask)
        values = dynamics[..., mask].movedim(-1, 0)
        hybrid_shape = [10, 20, 7, 8]
        sparse_dynamics = torch.sparse_coo_tensor(indices, values, size=hybrid_shape).coalesce()

        func = torch.jit.script(Core.make_sparse_superposition)
        output = func(sparse_dynamics, kernel, dynamic_shape, None)
        self.assertTrue(output.shape == expected_shape)
        self.assertTrue(output.is_sparse is False)

        output.sum().backward()
        self.assertTrue(kernel.grad is not None)
        self.assertTrue(dynamics.grad is not None)

class TestParameter(unittest.TestCase):
    """
    Test the actual parameter manager itself.
    """
    def test_make_basic_parameter(self):
        """ Test that a parameter with no superposition functions like normal"""
        shape = [20, 30, 10]
        func = init.kaiming_uniform_

        layer = Core.Parameter(func, shape)
        layer = torch.jit.script(layer)
        output = layer()
        self.assertTrue(torch.Size(shape) == output.shape)

    def test_make_superposition_parameter_dense(self):
        """ test we can build a superposition using a dense spec"""
        shape = [20, 30, 10]
        superposition = [10, 7]
        weights = torch.rand([10, 7])
        func = init.kaiming_uniform_


        expected_shape = torch.Size([20, 30, 10])
        layer = Core.Parameter(func, shape, superposition)
        layer = torch.jit.script(layer)
        output = layer(weights)
        self.assertTrue(expected_shape == output.shape)

    def test_make_batched_dense_superposition(self):
        """ test we can build a batched superposition using a dense spec"""
        shape = [20, 30, 10]
        superposition = [10, 7]
        weights = torch.rand([10, 7, 5])
        func = init.kaiming_uniform_


        expected_shape = torch.Size([5, 20, 30, 10])
        layer = Core.Parameter(func, shape, superposition)
        layer = torch.jit.script(layer)
        output = layer(weights)
        self.assertTrue(expected_shape == output.shape)

    def test_make_superposition_parameter_sparse(self):
        """ Test we can build a batched superposition with a sparse unbatched spec."""
        shape = [20, 30, 10]
        superposition = [10, 10]
        func = init.kaiming_uniform_

        weights = torch.rand([10, 10])
        mask = weights > 0.5
        weights = weights*mask
        weights = weights.to_sparse()


        expected_shape = torch.Size([20, 30, 10])
        layer = Core.Parameter(func, shape, superposition)
        layer = torch.jit.script(layer)
        output = layer(weights)
        self.assertTrue(output.shape == expected_shape)

    def test_sparse_batched(self):
        """ Test the superposition system still works properly when using a batched hybrid tensor"""
        shape = [20, 30, 10]
        superposition = [10, 5]
        func = init.kaiming_uniform_

        weights = torch.rand([10, 5, 7])
        mask = torch.rand([10, 5]) > 0.5
        weights = Core.convert_dense_to_hybrid(weights, mask)

        expected_shape = torch.Size([7, 20, 30, 10])
        layer = Core.Parameter(func, shape, superposition)
        layer = torch.jit.script(layer)
        output = layer(weights)
        self.assertTrue(output.shape == expected_shape)

    def test_different_dtype(self):
        """ Test that all is well when a different dtype is specified"""
        shape = [20, 30, 10]
        superposition = [10, 7]
        weights = torch.rand([10, 7])
        func = init.kaiming_uniform_
        dtype = torch.float64

        expected_shape = torch.Size([20, 30, 10])
        layer = Core.Parameter(func, shape, superposition, dtype=dtype)
        layer = torch.jit.script(layer)
        output = layer(weights.to(dtype=dtype))
        self.assertTrue(expected_shape == output.shape)
        self.assertTrue(output.dtype == dtype)


class TestCallErrors(unittest.TestCase):
    """
    Tests errors are being thrown when appropriate
    """
    def test_not_called_with_weights_when_needed(self):
        """Error test: Not called with weights"""
        shape = [20, 30, 10]
        superposition = [10, 10]
        func = init.kaiming_uniform_

        try:
            layer = Core.Parameter(func, shape, superposition)
            layer()
            raise RuntimeError("Did not throw when required")
        except Core.KernelSetupError as err:
            if PRINT_ERRORS:
                print_the_error(err)


    def test_called_with_weights_when_not_needed(self):
        """ Error test: Called with weights"""

        shape = [20, 30, 10]
        weights = torch.rand([10, 10])
        func = init.kaiming_uniform_

        try:
            layer = Core.Parameter(func, shape)
            layer(weights)
            raise RuntimeError("Did not throw when expected")
        except Core.KernelSetupError as err:
            if PRINT_ERRORS:
                print_the_error(err)

    def test_weights_wrong_shape(self):
        """ Error test: weights wrong shape"""

        shape = [20, 30, 10]
        superposition = [10, 10]
        weights = torch.rand([10])
        func = torch.nn.init.kaiming_uniform_

        try:
            layer = Core.Parameter(func, shape, superposition)
            layer(weights)
            raise RuntimeError("Did not throw when expected")
        except Core.KernelSetupError as err:
            if PRINT_ERRORS:
                print_the_error(err)

    def test_weights_almost_correct(self):
        """ test the common case where batch dimensions are not put last. """
        shape = [20, 30, 10]
        superposition = [10, 10]
        weights = torch.rand([5, 10, 10])
        func = torch.nn.init.kaiming_uniform_

        try:
            layer = Core.Parameter(func, shape, superposition)
            layer(weights)
            raise RuntimeError("Did not throw when expected")
        except Core.KernelSetupError as err:
            if PRINT_ERRORS:
                print_the_error(err)


    def test_weights_bad_dtype(self):
        """Error test: Bad dtype"""

        shape = [20, 30, 10]
        superposition = [10, 10]
        weights = torch.rand([10, 10, 5]).to(dtype=torch.int64)
        func = torch.nn.init.kaiming_uniform_

        try:
            layer = Core.Parameter(func, shape, superposition)
            layer(weights)
            raise RuntimeError("Did not throw when expected")
        except Core.KernelSetupError as err:
            if PRINT_ERRORS:
                print_the_error(err)
