import unittest
import torch
from torch import nn
import torch.nn
import itertools
import src.supertransformerlib.Core


### Fixtures ###
#
# These must be located at the top level so pickle
# is happy

class buffer_mockup(nn.Module):
    """
    Dynamically varying kernel
    Simple task: add
    """

    def __init__(self):
        super().__init__(5, top_p=0.5)

        self.d_model = 20
        self.batch_fun = 4
        self.kernel = torch.randn([self.native_ensemble_width, self.d_model])
        self.kernel = nn.Parameter(self.kernel)
        self.register_ensemble("kernel")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        config = torch.randn([self.batch_fun, self.native_ensemble_width])
        self.set_config(config, True)
        return x + self.kernel

#Fixtures. Must be top level for pickle to handle


class testLinear(unittest.TestCase):
    """
    This is the test feature for the linear layer.
    """

    def test_Regular(self):
        """ Tests if the standard pytorch linear layer behavior is reproduced"""

        tensor = torch.rand([2, 5])
        tester = src.supertransformerlib.Core.Linear(5, 10)
        tester = torch.jit.script(tester)
        test = tester(tensor)
        self.assertTrue(test.shape == torch.Size([2, 10]), "Regular pytorch layer not reproduced")

    def test_Reshapes(self):
        """ Tests whether the reshape functionality is working in isolation """
        # Define test tensor
        tensor = torch.rand([30, 20, 15])

        # Define test layers
        test_expansion = src.supertransformerlib.Core.Linear(15, [5, 3])
        test_collapse = src.supertransformerlib.Core.Linear([20, 15], 300)
        test_both = src.supertransformerlib.Core.Linear([20, 15], [10, 30])

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

        test_single = src.supertransformerlib.Core.Linear(10, 20, 20)
        test_multiple = src.supertransformerlib.Core.Linear(10, 20, [30, 20])

        # Run tests

        test_single_result = test_single(tensor)
        test_multiple_result = test_multiple(tensor)

    def test_Head_Independence(self):
        """ Tests whether each parallel is completely independent"""

        # Create tensors
        tensor_a = torch.stack([torch.zeros([20]), torch.zeros([20])])
        tensor_b = torch.stack([torch.zeros([20]), torch.ones([20])])

        # create tester

        test_head_independence = src.supertransformerlib.Core.Linear(20, 20, 2)

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
        test_grad = src.supertransformerlib.Core.Linear([20, 10], 1)

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
        test_script = src.supertransformerlib.Core.Linear(20, 10, 1)

        # Perform test
        scripted = torch.jit.script(test_script)
        scripted(test_tensor)

    def test_dynamics(self):
        """Test whether or not dynamic assignment works."""
        test_tensor = torch.randn([30, 20, 20])
        test_layer = src.supertransformerlib.Core.Linear([20,20], 10, 30)
        test_layer = torch.jit.script(test_layer)
        output = test_layer(test_tensor)
        self.assertTrue(output.shape == torch.Size([30, 10]))

    def test_passable(self):
        """Test whether or not passing and executing linear later on is possible"""
        test_tensor = torch.randn([30, 20, 20])
        test_layer = src.supertransformerlib.Core.Linear([20,20], 10, 30)
        test_layer = torch.jit.script(test_layer)

        @torch.jit.script
        def perform_linear(forward: src.supertransformerlib.Core.Linear.ForwardType, tensor: torch.Tensor):
            return forward(tensor)

        forward = test_layer.setup_forward()
        output = perform_linear(forward, test_tensor)
        self.assertTrue(output.shape == torch.Size([30, 10]))

    def test_gradient_passable(self):
        """Test whether or not a passable feature updates on gradient descent"""

        test_tensor = torch.randn([30, 20, 20])
        test_layer = src.supertransformerlib.Core.Linear([20,20], 10, 30)
        test_optim = torch.optim.SGD(test_layer.parameters(), lr=0.01)

        test_layer = torch.jit.script(test_layer)

        @torch.jit.script
        def perform_linear(forward: src.supertransformerlib.Core.Linear.ForwardType, tensor: torch.Tensor):
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


class test_ViewPoint(unittest.TestCase):
    """
    Test the viewpoint mechanism.

    Verify it currently functions correctly
    using manually designed tests with known
    results.
    """
    def test_viewpoint_one(self):
        tensor = torch.tensor([[[1, 0],[0,1]]]
        index = torch.tensor([]

        print(tensor.shape)


class test_ViewpointFactory(unittest.TestCase):
    def test_constructor(self):
        """test that the constructor works at all"""

        src.supertransformerlib.Core.ViewPointFactory(32, 32, 8, 20, 4)


    def test_viewpoint_shape(self):
        query_tensor = torch.randn([2, 3, 32])
        text_tensor = torch.randn([2, 10, 32])
        factory = src.supertransformerlib.Core.ViewPointFactory(32, 32, 8, 5, 4)
        viewpoint = factory(query_tensor, text_tensor)
        expected_shape = torch.Size([2, 8, 3, 5, 32])

        outcome = viewpoint(text_tensor)
        self.assertTrue(expected_shape == outcome.shape)

