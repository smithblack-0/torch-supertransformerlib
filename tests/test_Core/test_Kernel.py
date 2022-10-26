import unittest

import torch
from torch.nn import init

from src.supertransformerlib.Core import Kernel
from src.supertransformerlib.Core import SparseUtils


class testKernelSuperposition(unittest.TestCase):
    """
    Test that the kernel superposition
    function will work properly no matter
    the circumstances.
    """

    def test_dense_superposition(self):

        kernel = torch.randn([10, 20, 30, 5, 4])
        dynamics = torch.randn([10, 20, 7, 8])
        dynamic_shape = torch.tensor([10, 20])
        expected_shape = torch.Size([7, 8, 30, 5, 4])


        #Standard

        output = Kernel.make_dense_superposition(dynamics, kernel, dynamic_shape)
        self.assertTrue(output.shape == expected_shape)

        #Torchscript

        function = torch.jit.script(Kernel.make_dense_superposition)
        output = function(dynamics, kernel, dynamic_shape)
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

        #Standard

        output = Kernel.make_sparse_superposition(sparse_dynamics, kernel, dynamic_shape)
        self.assertTrue(output.shape == expected_shape)
        self.assertTrue(output.is_sparse == False)

        output.sum().backward()
        self.assertTrue(kernel.grad != None)
        self.assertTrue(dynamics.grad != None)

        #Torchscript

        mask = torch.randn([10, 20]) > 0
        indices = SparseUtils.gen_indices_from_mask(mask)
        values = dynamics[..., mask].movedim(-1, 0)
        hybrid_shape = [10, 20, 7, 8]
        sparse_dynamics = torch.sparse_coo_tensor(indices, values, size=hybrid_shape).coalesce()

        func = torch.jit.script(Kernel.make_sparse_superposition)
        output = func(sparse_dynamics, kernel, dynamic_shape)
        self.assertTrue(output.shape == expected_shape)
        self.assertTrue(output.is_sparse == False)

        output.sum().backward()
        self.assertTrue(kernel.grad != None)
        self.assertTrue(dynamics.grad != None)

class test_Parameter(unittest.TestCase):
    """
    Test the actual parameter manager itself.
    """
    def test_make_basic_parameter(self):
        """ Test that a parameter with no superposition functions like normal"""
        shape = [20, 30, 10]
        func = init.kaiming_uniform_

        layer = Kernel.Parameter(func, shape)
        layer = torch.jit.script(layer)
        output = layer()
        self.assertTrue(torch.Size(shape) == output.shape)

    def test_make_superposition_parameter_dense(self):

        shape = [20, 30, 10]
        superposition = [10, 7]
        weights = torch.rand([10, 7, 5])
        func = init.kaiming_uniform_


        expected_shape = torch.Size([5, 20, 30, 10])
        layer = Kernel.Parameter(func, shape, superposition)
        layer = torch.jit.script(layer)
        output = layer(weights)
        self.assertTrue(expected_shape == output.shape)

    def test_make_superposition_parameter_sparse(self):

        shape = [20, 30, 10]
        superposition = [10, 10]

        weights = torch.rand([10, 10])
        mask = weights > 0.5
        weights = weights*mask
        weights = weights.to_sparse_coo()

        mode = "sparse"

    def test_different_dtype(self):

        shape = [20, 30, 10]
        superposition = None
        dtype = torch.float64



class test_Parameter_Call_Errors(unittest.TestCase):
    """
    Tests errors are being thrown when appropriate
    """
    def test_not_called_with_weights(self):
        shape = [20, 30, 10]
        superposition = [10, 10]

    def test_called_with_weights_when_not_needed(self):

        shape = [20, 30, 10]
        superposition = None
        weights = torch.rand([10, 10])

    def test_weights_wrong_shape(self):

        shape = [20, 30, 10]
        superposition = [10, 10]
        weights = torch.rand([10])
        mode = "dense"

    def test_weights_sparse_when_should_be_dense(self):
        shape = [20, 30, 10]
        superposition = [10, 10]

        weights = torch.rand([10, 10])
        mask = weights > 0.5
        weights = weights*mask
        weights = weights.to_sparse_coo()

        mode = "dense"

    def test_weights_dense_when_should_be_sparse(self):
        shape = [20, 30, 10]
        superposition = [10, 10]
        weights = torch.rand([10, 10])
        mode = "sparse"

    def test_weights_bad_dtype(self):

        shape = [20, 30, 10]
        superposition = [10, 10]
        weights = torch.rand([10, 10]).to(dtype=torch.int64)
        mode = "dense"

    