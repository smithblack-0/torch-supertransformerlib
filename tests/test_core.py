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

class buffer_mockup(src.supertransformerlib.Core.KernelSpace):
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

class TestKit(unittest.TestCase):
    """
    A bunch of useful utilities for testing purposes
    """
    def assert_tensors_about_equal(self, tensor1: torch.Tensor, tensor2: torch.Tensor, max_epsilon):
        """

        :param tensor1:
        :param tensor2:
        :param max_epsilon:
        :return:
        """
        if tensor1.shape != tensor2.shape:
            raise AssertionError("Tensors not equal: Shapes not equal")
        diff = max_epsilon - (tensor1-tensor2).abs()
        if torch.any(diff < 0):
            msg = "Tensors not about equal. Max epsilon exceeded. Happend at: \n"
            msg += "%s " % torch.argwhere(diff < 0)
            raise AssertionError(msg)


class test_Config(TestKit):
    """
    Test the ability of the config subsection to properly
    process inputs into valid configurations.
    """
    def test_statistics_logits(self):
        about_equal_threshold = 0.001
        config_tensor = torch.randn([3, 10])
        config_basic = src.supertransformerlib.Core.Config(config_tensor)
        config_top_k = src.supertransformerlib.Core.Config(config_tensor, top_k=4)
        config_top_p = src.supertransformerlib.Core.Config(config_tensor, top_p=0.6)

        #Test that basic config sums to about one on the last dim

        test = (config_basic.config.sum(dim=-1) -1 ).abs() > 0.011
        if torch.any(test):
            raise AssertionError("Probabilities for basic configuration did not add up to one")

        #Test that top_k is engaged, and that probs add up to one

        test = (config_top_k.config.sum(dim=-1) -1 ).abs() > 0.011
        if torch.any(test):
            raise AssertionError("Probabilities for top-k configuration did not add up to one")
        inactive = config_top_k.config == 0
        inactive = inactive.sum(dim=-1)
        inactive = inactive.sum()//inactive.numel()
        if inactive != 6:
            raise AssertionError("Invalid number of configurations are active")

        #Test that it is the case that top-p adds up to one


        test = (config_top_p.config.sum(dim=-1) -1 ).abs() > 0.011
        if torch.any(test):
            raise AssertionError("Probabilities for top-p configuration did not add up to one")

class test_Kernel(TestKit):
    """
    Test the ability of the kernel layer to
    put together a working kernel, store the configuration, and
    be updated later on.
    """
    def test_ensemble_auto(self):
        """Test the ability to load a defined ensemble given a variety of configurations"""

        #Build the various test cases. This consists of combinations
        #of ensemble definitions along with configurations
        ensemble_width = 10

        ensemble_registration_cases = {
            "simple_ensemble_case" : torch.randn([ensemble_width, 5]),
            "complex_ensemble_case" : torch.randn([ensemble_width, 5, 3,6, 2])
        }
        configuration_cases = {
            "primitive_configuration" : torch.randn([2, ensemble_width]),
            "basic_configuration" : torch.randn([5, ensemble_width]),
            "advanced_configuration" : torch.randn([20, 5, ensemble_width])
        }

        #Construct and test the cases.
        ensemble_cases = [(key, value) for key, value in ensemble_registration_cases.items()]
        config_cases = [(key, value) for key, value in configuration_cases.items()]
        for case in itertools.product(ensemble_cases, config_cases):
            (ensemble_name, ensemble_value), (config_name, config_value) = case
            expected_final_shape = torch.Size([*config_value.shape[:-1], *ensemble_value.shape[1:]])
            try:
                kernel = src.supertransformerlib.Core.Kernel(ensemble_value)
                kernel = torch.jit.script(kernel)

                config = src.supertransformerlib.Core.Config(config_value)
                config = torch.jit.script(config)

                kernel.update_config(config)

                final_kernel = kernel()
                self.assertTrue(final_kernel.shape == expected_final_shape)

            except Exception as err:
                msg = "Error produced while testing ensemble case: %s with config case: %s"\
                      % (ensemble_name, config_name)
                raise Exception(msg) from err

    def test_load_ensemble_manual(self):
        """Test that load ensemble is working correctly for a few known, manually configured values"""
        epsilion = 0.0001

        off = -1e+8
        test_cases = [
            {"ensemble": torch.tensor([[0, 1.], [2, 3]]),
             "config": torch.tensor([[0.5, 0.5]]),
             "output": torch.tensor([[1.0, 2.0]])
             },
            {"ensemble": torch.tensor([[0, 1], [2, 3.]]),
             "config": torch.tensor([[1, off], [0.5, 0.5], [off, 1]]),
             "output": torch.tensor([[0, 1], [1.0, 2.0], [2, 3]])
             },
            {"ensemble": torch.tensor([[0, 1], [2, 3.]]),
             "config": torch.tensor([[[0.5, 0.5]], [[off, 1]]]),
             "output": torch.tensor([[[1.0, 2.0]], [[2, 3]]])
             },
            {"ensemble": torch.tensor([[[0, 1.], [2, 3]], [[3, 4], [4, 5]]]),
             "config": torch.tensor([[1, off], [0.5, 0.5], [off, 1]]),
             "output": torch.tensor([[[0, 1, ], [2, 3]], [[1.5, 2.5], [3, 4]], [[3, 4], [4, 5]]])
             }
        ]

        for case in test_cases:
            ensemble = case["ensemble"]
            config = case["config"]
            expectations = case["output"]

            config = src.supertransformerlib.Core.Config(config, logits=False)
            kernel = src.supertransformerlib.Core.Kernel(ensemble)
            kernel = torch.jit.script(kernel)
            kernel.update_config(config)
            test_result = kernel()

            diff = (test_result - expectations).abs()
            passed = torch.all(epsilion > diff)
            self.assertTrue(passed)
    def test_tag_restriction(self):
        """Test that tag restrictions are respected"""
        kernel_restricted = src.supertransformerlib.Core.Kernel(nn.Parameter(torch.randn([20, 10, 5])), tags=["a", "b"])
        kernel_restricted = torch.jit.script(kernel_restricted)

        #Generic is applied correctly
        config = src.supertransformerlib.Core.Config(torch.randn([10, 20]))
        kernel_restricted.update_config(config)
        output = kernel_restricted()
        self.assertTrue(output.shape == torch.Size([10, 10, 5]))

        #When a non included tag pops up, no update occurs
        config = src.supertransformerlib.Core.Config(torch.randn([5, 20]), tags=["c"])
        kernel_restricted.update_config(config)
        output = kernel_restricted()
        self.assertTrue(output.shape == torch.Size([10, 10, 5]))

        #When an included tag is detected, it does cause an update
        config = src.supertransformerlib.Core.Config(torch.randn([3, 20]), tags=["a"])
        kernel_restricted.update_config(config)
        output = kernel_restricted()
        self.assertTrue(output.shape == torch.Size([3, 10, 5]))



#Fixtures. Must be top level for pickle to handle


class mockup(src.supertransformerlib.Core.KernelSpace):
    def __init__(self):
        super().__init__()
        self.kernel = src.supertransformerlib.Core.Kernel(nn.Parameter(torch.randn([10, 20, 30])))

    def forward(self):
        return self.kernel()


class mockup2(src.supertransformerlib.Core.KernelSpace):
    def __init__(self):
        super().__init__()
        self.kernel = src.supertransformerlib.Core.Kernel(nn.Parameter(torch.randn([10, 20, 30])))
        self.kernel2 = mockup()

    def forward(self):
        return self.kernel(), self.kernel2()


class test_KernelSpace(unittest.TestCase):
    """
    Test suite for the kernelSpace.

    Test we can subclass successfully
    and assign new configurations,
    even in nested environments.
    """
    def test_kernel_accumulation(self):
        """
        Test that we can properly detect and modify
        the kernels
        """


        config = src.supertransformerlib.Core.Config(torch.randn([30, 10]))

        instance = mockup()
        instance = torch.jit.script(instance)
        instance.update_children(config)
        output = instance()

        self.assertTrue(output.shape == torch.Size([30, 20, 30]), output.shape)
    def test_nested_kernel_updates(self):


        config = src.supertransformerlib.Core.Config(torch.randn([30, 10]))

        instance = mockup2()
        instance = torch.jit.script(instance)
        instance.update_descendents(config)
        output1, output2 = instance()

        self.assertTrue(output1.shape == torch.Size([30, 20, 30]))
        self.assertTrue(output2.shape == torch.Size([30, 20, 30]))
        print(output1.shape, output2.shape)


    def test_saving_loading(self):
        """Test that it is the case that an instance can be saved, and loaded, properly """

        config = src.supertransformerlib.Core.Config(torch.randn([30, 10]))

        python_instance = mockup2()
        torchscript_instance = torch.jit.script(python_instance)

        python_instance.update_descendents(config)
        torchscript_instance.update_descendents(config)

        python_initial_output1, python_initial_output2 = python_instance()
        torchscript_initial_output1, torchscript_initial_output2 = torchscript_instance()

        torch.save(python_instance, "python_test_save.raw")
        python_instance = torch.load("python_test_save.raw")

        torch.jit.save(torchscript_instance, "torchscript_save.raw")
        torchscript_instance = torch.jit.load("torchscript_save.raw")

        python_final1, python_final2 = python_instance()
        torchscript_final1, torchscript_final2 = torchscript_instance()

        self.assertTrue(torch.all(python_initial_output1 == python_final1))
        self.assertTrue(torch.all(python_initial_output2 == python_final2))
        self.assertTrue(torch.all(torchscript_initial_output1 == torchscript_final1))
        self.assertTrue(torch.all(torchscript_initial_output2 == torchscript_final2))


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
        test_config = src.supertransformerlib.Core.Config(torch.randn([30, 10]))
        test_layer = src.supertransformerlib.Core.Linear([20,20], 10, dynamics=10)
        test_layer = torch.jit.script(test_layer)
        test_layer.update_descendents(test_config)
        output = test_layer(test_tensor)
        self.assertTrue(output.shape == torch.Size([30, 10]))
