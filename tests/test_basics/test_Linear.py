import unittest

import torch
from src.supertransformerlib import Glimpses
from src.supertransformerlib.Basics import Linear

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

        #...
        output = ...
        self.assertTrue(output.shape == expected_shape)
    def test_batched_linear(self):
        """Test the linear operator works when handling batches"""
        tensor = torch.randn([15, 10])
        kernel = torch.randn([10, 5])
        bias = torch.randn([5])
        expected_shape = torch.Size([15, 5])

        #...
        output = ...
        self.assertTrue(output.shape == expected_shape)

    def test_multibatched_linear(self):
        """Test the linear operator works when handling arbitrary batches"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([10, 5])
        bias = torch.randn([5])
        expected_shape = torch.Size([40, 20, 15, 5])

        # ...
        output = ...
        self.assertTrue(output.shape == expected_shape)
    def test_parallel_kernel_linear(self):
        """Test the linear operators broadcasts across parallel kernels properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        # ...
        output = ...
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

        # ...
        output = ...
        self.assertTrue(output.shape == expected_shape)

    def test_batched_linear(self):
        """Test the linear operator works when handling batches"""
        tensor = torch.randn([15, 10])
        kernel = torch.randn([10, 5])
        expected_shape = torch.Size([15, 5])

        # ...
        output = ...
        self.assertTrue(output.shape == expected_shape)

    def test_multibatched_linear(self):
        """Test the linear operator works when handling arbitrary batches"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([10, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        # ...
        output = ...
        self.assertTrue(output.shape == expected_shape)
    def test_parallel_kernel_linear(self):
        """Test the linear operators broadcasts across parallel kernels properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        expected_shape = torch.Size([40, 20, 15, 5])

        # ...
        output = ...
        self.assertTrue(output.shape == expected_shape)

class testReshapingForward(unittest.TestCase):
    """
    Test the full forward mechanism with reshapes included
    within teh package
    """
    def test_reshape(self):
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([150, 5])
        bias = torch.randn([5])
        input_reshape = Glimpses.ReshapeClosure([15, 10], 150)
        output_reshape = Glimpses.ReshapeClosure(5, 5)
        expected_shape = torch.Size([40, 20, 5])

        # ...
        output = ...
        self.assertTrue(output.shape == expected_shape)

class testForwardExceptions(unittest.TestCase):
    def test_wrong_linear_dim(self):
        """
        Test a reasonable error occurs when the last dimension is the
        wrong shape for the kernel
        """
        tensor = torch.randn([40, 20, 15, 7])
        kernel = torch.randn([15, 10, 5])
        expected_error = ...

        try:
            ...
        except expected_error as err:
            pass

    def test_wrong_dtype(self):
        """
        Test a reasonable error occurs when the tensor
        and kernel do not match.
        """
        tensor = torch.randn([40, 20, 15, 7]).to(torch.complex64)
        kernel = torch.randn([15, 10, 5])
        expected_error = ...
        try:
            ...
        except expected_error as err:
            pass

    def test_wrong_parallel_dim(self):
        """
        Test a reasonable error occurs when the parallelization
        dimensions do not match
        """
        tensor = torch.randn([40, 20, 4, 10])
        kernel = torch.randn([15, 10, 5])
        expected_error = ...

        try:
            ...
        except expected_error as err:
            pass




class testForwardClosure(unittest.TestCase):
    """ Test that the closure mechanism is being constructed and used properly"""
    def test_closure_unbiased(self):
        """Test the linear operators broadcasts across parallel kernels properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        input_map = Glimpses.ReshapeClosure(10, 10)
        output_map = Glimpses.ReshapeClosure(5, 5)
        expected_shape = torch.Size([40, 20, 15, 5])

        # ...
        output = ...
        self.assertTrue(output.shape == expected_shape)

    def test_closure_biased(self):
        """Test the linear operators broadcasts across parallel kernels properly"""
        tensor = torch.randn([40, 20, 15, 10])
        kernel = torch.randn([15, 10, 5])
        bias = torch.randn([15, 5])
        input_map = Glimpses.ReshapeClosure(10, 10)
        output_map = Glimpses.ReshapeClosure(5, 5)
        expected_shape = torch.Size([40, 20, 15, 5])

        # ...
        output = ...
        self.assertTrue(output.shape == expected_shape)

class testKernelConstruct(unittest.TestCase):
    """Test the ability to construct the required linear kernels"""
    def test_basic(self):
        input_shape = 10
        output_shape = 5
        expected_shape = torch.Size([10, 5])
    def test_parallel(self):
        input_shape = 30
        output_shape = 5
        parallel_shape = [10, 15]




class testLinear(unittest.TestCase):
    """
    Tests the linear factory layer. It emits
    linear closures when called.
    """
    def test_basic(self):
        """Tests a basic linear layer"""
        tensor = torch.randn([10])
        keywords = {
            'input_shape' : 10,
            'output_shape' : 5
        }
        expected_shape = torch.Size([5])
        ...
        closure = ...
        output = closure(tensor)
        self.assertTrue(output.shape == expected_shape)
    def test_parallel_utilization(self):
        """Tests usage on parallel tensors"""
        tensor = torch.randn([10])
        keywords = {
            'input_shape' : 10,
            'output_shape' : 5
        }
        expected_shape = torch.Size([5])
        ...
        closure = ...
        output = closure(tensor)
        self.assertTrue(output.shape == expected_shape)





class testLinear(unittest.TestCase):
    """
    This is the test feature for the linear layer.
    """

    def test_Regular(self):
        """ Tests if the standard pytorch linear layer behavior is reproduced"""

        tensor = torch.rand([2, 5])
        tester = src.supertransformerlib.Basics.Linear(5, 10)
        tester = torch.jit.script(tester)
        test = tester(tensor)
        self.assertTrue(test.shape == torch.Size([2, 10]), "Regular pytorch layer not reproduced")

    def test_Reshapes(self):
        """ Tests whether the reshape functionality is working in isolation """
        # Define test tensor
        tensor = torch.rand([30, 20, 15])

        # Define test layers
        test_expansion = src.supertransformerlib.Basics.Linear(15, [5, 3])
        test_collapse = src.supertransformerlib.Basics.Linear([20, 15], 300)
        test_both = src.supertransformerlib.Basics.Linear([20, 15], [10, 30])

        # Perform tests

        test_expansion_result = test_expansion(tensor)
        test_collapse_result = test_collapse(tensor)
        test_both_result = test_both(tensor)

        expansion_bool = [*test_expansion_result.shape] == [30, 20, 5, 3]
        collapse_bool = [*test_collapse_result.shape] == [30, 300]
        both_bool = [*test_both_result.shape] == [30, 10, 30]

        # Assert results
        self.assertTrue(expansion_bool, "Reshape: Expansion failed")
        self.assertTrue(collapse_bool, "Reshape: collapse failed")
        self.assertTrue(both_bool, "Reshape: Compound failed")

    def test_Heading(self):
        """ Tests whether the parallel kernels and bias are implemented such that calling works"""

        tensor = torch.randn([10, 30, 20, 10])

        # Create test layers

        test_single = src.supertransformerlib.Basics.Linear(10, 20, 20)
        test_multiple = src.supertransformerlib.Basics.Linear(10, 20, [30, 20])

        # Run tests

        test_single_result = test_single(tensor)
        test_multiple_result = test_multiple(tensor)

    def test_Head_Independence(self):
        """ Tests whether each parallel is completely independent"""

        # Create tensors
        tensor_a = torch.stack([torch.zeros([20]), torch.zeros([20])])
        tensor_b = torch.stack([torch.zeros([20]), torch.ones([20])])

        # create tester

        test_head_independence = src.supertransformerlib.Basics.Linear(20, 20, 2)

        # Run tests

        test_result_a = test_head_independence(tensor_a)
        test_result_b = test_head_independence(tensor_b)

        # Analyze and assert result
        result_bool = torch.all(test_result_a[0] == test_result_b[0])
        self.assertTrue(result_bool, "Heads were found to be interacting")

    def test_gradients(self):
        """Test whether or not gradients are propogating properly"""
        test_tensor = torch.randn([20, 10])

        # Develop test layer
        test_grad = src.supertransformerlib.Basics.Linear([20, 10], 1)

        # Develop optim
        test_optim = torch.optim.SGD(test_grad.parameters(), lr=0.01)

        # perform test
        test_result = test_grad(test_tensor)
        test_result.backward()

        test_optim.step()

    def test_jit_basic(self):
        """ Test whether or not the module is scriptable when instanced"""
        # Develop test layer
        test_tensor = torch.randn([30, 20, 20])
        test_script = src.supertransformerlib.Basics.Linear(20, 10, 1)

        # Perform test
        scripted = torch.jit.script(test_script)
        scripted(test_tensor)

    def test_dynamics(self):
        """Test whether or not dynamic assignment works."""
        test_tensor = torch.randn([30, 20, 20])
        test_layer = src.supertransformerlib.Basics.Linear([20, 20], 10, 30)
        test_layer = torch.jit.script(test_layer)
        output = test_layer(test_tensor)
        self.assertTrue(output.shape == torch.Size([30, 10]))

    def test_passable(self):
        """Test whether or not passing and executing linear later on is possible"""
        test_tensor = torch.randn([30, 20, 20])
        test_layer = src.supertransformerlib.Basics.Linear([20, 20], 10, 30)
        test_layer = torch.jit.script(test_layer)

        @torch.jit.script
        def perform_linear(forward: src.supertransformerlib.Basics.Linear.ForwardType, tensor: torch.Tensor):
            return forward(tensor)

        forward = test_layer.setup_forward()
        output = perform_linear(forward, test_tensor)
        self.assertTrue(output.shape == torch.Size([30, 10]))

    def test_gradient_passable(self):
        """Test whether or not a passable feature updates on gradient descent"""

        test_tensor = torch.randn([30, 20, 20])
        test_layer = src.supertransformerlib.Basics.Linear([20, 20], 10, 30)
        test_optim = torch.optim.SGD(test_layer.parameters(), lr=0.01)

        test_layer = torch.jit.script(test_layer)

        @torch.jit.script
        def perform_linear(forward: src.supertransformerlib.Basics.Linear.ForwardType, tensor: torch.Tensor):
            return forward(tensor)



        forward = test_layer.setup_forward()
        output = perform_linear(forward, test_tensor)
        output = output.sum()
        output.backward()

        test_optim.step()
        test_optim.zero_grad()
        new_output = perform_linear(forward, test_tensor)
        new_output = new_output.sum()

        print(new_output)
        print(output)

        self.assertTrue(new_output != output)