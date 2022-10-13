import unittest

import numpy as np
import torch

from src.supertransformerlib import Glimpses
from src.supertransformerlib import Core
print_errors = True


class testReshape(unittest.TestCase):
    """
    Test whether the reshape mechanism is working sanely
    """

    def test_constructor(self):
        Glimpses.Reshape(1, 1)

    def test_prebuild_reshape(self):
        """
        Test that prebuilding then executing the reshape
        raises errors when appropriate and performs a reshape when not
        """

        def test_should_succeed(tensor: torch.Tensor,
                                input_shape: Core.StandardShapeType,
                                output_shape: Core.StandardShapeType,
                                expected_shape: torch.Size):

            #Standard

            input_shape = Core.standardize_shape(input_shape, "input_shape")
            output_shape = Core.standardize_shape(output_shape, "output_shape")
            reshape = Glimpses.Reshape(input_shape, output_shape)
            output = reshape(tensor)
            self.assertTrue(output.shape == expected_shape)

            #Torchscript

            input_shape = Core.standardize_shape(input_shape, "input_shape")
            output_shape = Core.standardize_shape(output_shape, "output_shape")
            reshape = Glimpses.Reshape(input_shape, output_shape)
            reshape = torch.jit.script(reshape)
            output = reshape(tensor)
            self.assertTrue(output.shape == expected_shape)


        #Tests good cases

        good_simple = (torch.randn([5]), 5, 5, torch.Size([5]))
        good_batched_tensor = (torch.randn([5, 5, 5]), [5, 5], [25], torch.Size([5, 25]))
        good_tensor_defined = (torch.randn([5, 5, 5]), torch.tensor([5, 5]), torch.tensor([5, 5]), torch.Size([5, 5, 5]))

        #Run cases

        test_should_succeed(*good_simple)
        test_should_succeed(*good_batched_tensor)
        test_should_succeed(*good_tensor_defined)


    def test_validate_reshape(self):
        """Test whether validation is operating correctly when used"""

        task = ["Running tests"]
        reshape = Glimpses.Reshape(1, 1)

        def test_should_success(tensor: torch.Tensor,
                                input_shape: torch.Tensor,
                                output_shape: torch.Tensor):

            input_shape = Core.standardize_shape(input_shape, "input_shape")
            output_shape = Core.standardize_shape(output_shape, "output_shape")

            #Standard
            reshape.validate_reshape(tensor, input_shape, output_shape)

            #Torchscript


            func = torch.jit.script(reshape.validate_reshape)
            func(tensor, input_shape, output_shape)

        def test_should_fail(tensor: torch.Tensor,
                             input_shape: torch.Tensor,
                             output_shape: torch.Tensor):

            try:
                reshape.validate_reshape(tensor, input_shape, output_shape)
                raise RuntimeError("No error thrown")
            except Glimpses.ReshapeException as err:
                if print_errors:
                    print(err)

            try:
                func = torch.jit.script(reshape.validate_reshape)
                func(tensor, input_shape, output_shape)
                raise RuntimeError("No error thrown")
            except torch.jit.script as err:
                pass


        #Tests good cases

        good_simple = (torch.randn([5]), 5, 5)
        good_batched_tensor = (torch.randn([5, 5, 5]), [5, 5], [25])
        good_tensor_defined = (torch.randn([5, 5, 5]), torch.tensor([5, 5]), torch.tensor([5, 5]))

        #Run cases

        test_should_success(*good_simple)
        test_should_success(*good_batched_tensor)
        test_should_success(*good_tensor_defined)





class testView(unittest.TestCase):
    def testBasic(self):
        """ Tests whether view works """
        test_tensor = torch.randn([20, 30, 10])
        Glimpses.reshape(test_tensor, 10, [5, 2])
        Glimpses.reshape(test_tensor, [30, 10], [50, 6])
        Glimpses.reshape(test_tensor, torch.tensor([30, 10]), torch.tensor([50, 6]))
        Glimpses.reshape(test_tensor, torch.tensor([30, 10]), torch.tensor([50, 6], dtype=torch.int32))


class testLocal(unittest.TestCase):
    def testAsLayer(self):
        """
        Test if a simple layer works.
        """

        # Perform direct logic test
        tensor = torch.arange(30)
        kernel, stride, dilation = 1, 1, 1
        final = tensor.unsqueeze(-1)

        test = Glimpses.local(tensor, kernel, stride, dilation)
        test = torch.all(test == final)

        self.assertTrue(test, "Logical failure: results did not match manual calculation")
    def testKernel(self):
        """
        Test if a straightforward local kernel, as used in a convolution, works
        """

        # Perform kernel compile and logical test
        tensor = torch.tensor([0, 1, 2, 3, 4, 5])
        final = torch.tensor([[0 ,1] ,[1, 2], [2, 3], [3, 4], [4, 5]])
        kernel, stride, dilation = 2, 1, 1

        test = Glimpses.local(tensor, kernel, stride, dilation)
        test = torch.all(test == final)
        self.assertTrue(test, "Logical failure: Kernels not equal")
    def testStriding(self):
        """
        Test if a strided kernel, as used in a convolution, works
        """

        # Perform striding compile and logical test
        tensor = torch.tensor([0, 1, 2, 3, 4, 5])
        final = torch.tensor([[0], [2], [4]])
        kernel, stride, dilation = 1, 2, 1

        test = Glimpses.local(tensor, kernel, stride, dilation)
        test = torch.all(test == final)
        self.assertTrue(test, "Logical failure: striding did not match")
    def testDilation(self):
        """
        Test if a combination of dilated kernels works.
        """

        # Perform dilation test
        tensor = torch.tensor([0, 1, 2, 3, 4, 5])
        final = torch.tensor([[0, 2], [1, 3], [2, 4], [3, 5]])
        final2 = torch.tensor([[0, 2, 4], [1, 3, 5]])
        final3 = torch.tensor([[0, 3] ,[1 ,4] ,[2 ,5]])

        kernel1, stride1, dilation1 = 2, 1, 2
        kernel2, stride2, dilation2 = 3, 1, 2
        kernel3, stride3, dilation3 = 2, 1, 3

        test = Glimpses.local(tensor, kernel1, stride1, dilation1)
        test2 = Glimpses.local(tensor, kernel2, stride2, dilation2)
        test3 = Glimpses.local(tensor, kernel3, stride3, dilation3)

        test = torch.all(final == test)
        test2 = torch.all(final2 == test2)
        test3 = torch.all(final3 == test3)

        self.assertTrue(test, "Logical failure: dilation with kernel did not match")
        self.assertTrue(test2, "Logical failure: dilation with kernel did not match")
        self.assertTrue(test3, "Logical failure: dilation with kernel did not match")
    def testRearranged(self):
        """
        Test if a tensor currently being viewed, such as produced by swapdims, works
        """
        # make tensor
        tensor = torch.arange(20)
        tensor = tensor.view((2, 10))  # This is what the final buffer should be viewed with respect to
        tensor = tensor.swapdims(-1, -2).clone()  # Now a new tensor with a new data buffer
        tensor = tensor.swapdims(-1, -2)  # Buffer is being viewed by stridings. This could fuck things up

        # Declare kernel, striding, final
        kernel, striding, dilation = 2, 2, 2

        # Make expected final
        final = []
        final.append([[0 ,2] ,[2 ,4], [4 ,6] ,[6 ,8]])
        final.append([[10, 12] ,[12, 14] ,[14, 16], [16, 18]])
        final = torch.tensor(final)

        # test
        test = Glimpses.local(tensor, kernel, striding, dilation)
        test = torch.all(final == test)
        self.assertTrue(test, "Logical failure: buffer issues")
    def testWhenSliced(self):
        """
        Test if a tensor which is a view through a slice works."""

        # make tensor
        tensor = torch.arange(20)
        tensor = tensor.view((2, 10))  # This is what the final buffer should be viewed with respect to
        tensor = tensor[:, 2:6]
        # Declare kernel, striding, final
        kernel, striding, dilation = 2, 2, 2

        # Make expected final
        final = []
        final.append([[2, 4]])
        final.append([[12, 14]])
        final = torch.tensor(final)

        # test
        test = Glimpses.local(tensor, kernel, striding, dilation)
        test = torch.all(final == test)
        self.assertTrue(test, "Logical failure: buffer issues")

        tensor[..., 0] = 30
        test = torch.all(final != test)
        self.assertTrue(test, "Logical failure: sync issues")
    def test_Striding2(self):
        test_tensor = torch.randn([10, 16])
        output = Glimpses.local(test_tensor, 2, 2, 1)
        self.assertTrue(output.shape[-2] == 8)
        output = Glimpses.local(test_tensor, 4, 4, 1)
        self.assertTrue(output.shape[-2] == 4)


class testDilocal(unittest.TestCase):
    def test_basic(self):
        """ Tests whether an uncomplicated, unchanging case works. This means stride, kernel is 1"""
        tensor = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        outcome = Glimpses.dilocal(tensor, 1, 1, [1, 2], pad_justification="forward")
        outcome = Glimpses.dilocal(tensor, 1, 1, [1, 2], pad_justification="backward")
        outcome = Glimpses.dilocal(tensor, 1, 1, [1, 2])
        self.assertTrue(np.array_equal(outcome.shape, [2, 2, 4, 1]))

    def testDilation(self):
        """
        Test if a combination of dilated kernels works.
        """

        # Setup constants
        tensor = torch.tensor([0, 1, 2, 3, 4, 5])
        stride = 1
        kernel=3
        dilation = [1, 2, 3]

        #State expected result
        final = []
        final.append(torch.tensor([[0, 0, 1], [0,1,2],[1,2,3],[2, 3, 4],[3,4,5], [4, 5, 0]]))
        final.append(torch.tensor([[0, 0, 2],[0, 1, 3], [0, 2, 4], [1,3,5], [2,4,0],[3,5,0]]))
        final.append(torch.tensor([[0, 0, 3], [0, 1, 4], [0, 2,5], [0, 3, 0], [1,4,0], [2,5,0]]))
        final = torch.stack(final)

        #Perform test
        test = Glimpses.dilocal(tensor, kernel, stride, dilation)
        self.assertTrue(np.array_equal(test, final))