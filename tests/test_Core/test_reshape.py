"""

Test the reshape mechanisms

"""
import unittest
from typing import Type

import torch

import src.supertransformerlib.Core.Errors
import src.supertransformerlib.Core.Functions
import src.supertransformerlib.Core as Core
from src.supertransformerlib.Core import Reshape
from torch.nn import functional


#Controls whether or not to print error messages
#
# Printing them to the console can help debug or validate them
# It also aids in forming reasonable error messages.

print_errors = False



class testfunctionreshape(unittest.TestCase):
    """
    Test the function reshape in glimpses
    """
    def test_should_succeed(self):
        """Test reshape cases where the action should succeed"""


        def should_succeed(tensor: torch.Tensor,
                           input_shape: src.supertransformerlib.Core.Functions.StandardShapeType,
                           output_shape: src.supertransformerlib.Core.Functions.StandardShapeType,
                           expected_shape: torch.Size,
                           validation=True,
                           task=None):
            output = Reshape.reshape(tensor, input_shape, output_shape, validation, task)
            self.assertTrue(output.shape == expected_shape)

        good_simple = (torch.randn([5]), 5, 5, torch.Size([5]))
        good_batched_tensor = (torch.randn([5, 5, 5]), [5, 5], [25], torch.Size([5, 25]))
        good_tensor_defined = (torch.randn([5, 5, 5]), torch.tensor([5, 5]), torch.tensor([5, 5]), torch.Size([5, 5, 5]))
        good_no_validation = (torch.randn([5, 5, 5]), [5, 5], [25], torch.Size([5, 25]), False )
        good_has_taskstack = (torch.randn([5,5,5]), [5, 5], [5, 5], torch.Size([5, 5, 5]), True, "Testing")

        should_succeed(*good_simple)
        should_succeed(*good_batched_tensor)
        should_succeed(*good_tensor_defined)
        should_succeed(*good_no_validation)
        should_succeed(*good_has_taskstack)

    def tests_should_fail(self):
        """Test that the validation conditions are caught"""


        def should_fail(tensor: torch.Tensor,
                        input_shape: src.supertransformerlib.Core.Functions.StandardShapeType,
                        output_shape: src.supertransformerlib.Core.Functions.StandardShapeType,
                        expected_failure_type: Type[Exception]):


            try:
                Reshape.reshape(tensor, input_shape, output_shape)
                raise RuntimeError("Did not throw exception")
            except torch.jit.Error as err:
                if print_errors:
                    print(err)
            except expected_failure_type as err:
                if print_errors:
                    print(err)

        bad_element_num = (torch.randn([5]), 5, 3, Reshape.ReshapeException)
        bad_shape = (torch.randn([5]), [-5], [-5], src.supertransformerlib.Core.Errors.StandardizationError)
        bad_dim_number = (torch.randn([5]), [3,6], 18, Reshape.ReshapeException)

        should_fail(*bad_shape)
        should_fail(*bad_dim_number)
        should_fail(*bad_element_num)

class test_sparse_reshape(unittest.TestCase):
    """
    Test that sparse reshape is functional.
    """

    def test_basic_sparse_reshape(self):
        """Tests if sparse reshape works at all"""

        tensor = torch.randn([20, 10, 5, 4])
        tensor = functional.dropout(tensor, 0.5)
        initial_shape = torch.tensor([5, 4])
        final_shape = torch.tensor([20])
        expected_shape = torch.Size([20, 10, 20])

        sparse_tensor = tensor.to_sparse_coo()

        sparse_reshaped = Reshape._sparse_reshape(sparse_tensor, initial_shape, final_shape)
        self.assertTrue(sparse_reshaped.shape == expected_shape)

    def test_corrolated_sparse_reshape(self):
        """Tests that performing a sparse reshape and a dense reshape do not work differently."""
        tensor = torch.randn([20, 10, 5, 4])
        tensor = functional.dropout(tensor, 0.5)
        initial_shape = torch.tensor([5, 4])
        final_shape = torch.tensor([20])

        sparse_tensor = tensor.to_sparse_coo()


        sparse_reshaped = Reshape._sparse_reshape(sparse_tensor, initial_shape, final_shape)
        dense_reshape = Reshape.reshape(tensor, initial_shape, final_shape)
        correlation = sparse_reshaped.to_dense() == dense_reshape
        self.assertTrue(torch.all(correlation))

    def test_backprop_sparse_reshape(self):
        """Test backpropagation travels through sparse reshape"""

        tensor = torch.randn([20, 10, 5, 4], requires_grad=True)

        update = functional.dropout(tensor, 0.5)
        sparse_tensor = update.to_sparse_coo()

        initial_shape = torch.tensor([20, 10, 5, 4])
        final_shape = torch.tensor([20*10*5*4])

        sparse_reshaped = Reshape._sparse_reshape(sparse_tensor, initial_shape, final_shape)
        sparse_reshaped = torch.sparse.sum(sparse_reshaped, dim=0)
        sparse_reshaped.backward()

        tensor.sum().backward()
        self.assertTrue(tensor.grad != None)

    def test_hybrid_sparse_reshape(self):
        tensor = torch.randn([10, 9, 8, 7])

        mask = torch.rand([10, 9, 8]) > 0.5
        index = Core.gen_indices_from_mask(mask)
        values = tensor[mask]
        sparse_tensor = torch.sparse_coo_tensor(index, values)

        initial_shape = [9, 8]
        final_shape = [72]

        reshaped_tensor = Reshape.reshape(sparse_tensor, initial_shape, final_shape)
        print(reshaped_tensor.shape)
