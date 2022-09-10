import unittest
import torch
from torch import nn
import itertools

import src.supertransformerlib.Core


### Fixtures ###
#
# These must be located at the top level so pickle
# is happy

class buffer_mockup(src.supertransformerlib.Core.EnsembleSpace):
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


class test_EnsembleSpace(unittest.TestCase):
    """

    Test fixture for the base ensemblespace

    """
    def test_ensemble_registration(self):
        """ Test the ability to register kernels as ensemble features. """
        instance = src.supertransformerlib.Core.EnsembleSpace(10)
        ensemble_feature = torch.randn([10, 12, 14])
        ensemble_feature_2 = torch.randn([10, 4])
        ensemble_bad_feature = torch.randn([4, 2, 5])
        ensemble_not_long_enough = torch.randn([4])

        # Test register when not existing
        instance.register_ensemble("test_fixture", ensemble_feature)
        instance.register_ensemble("test_fixture", ensemble_feature)

        # Test register when already attached
        instance.test_fixture_2 = ensemble_feature_2
        instance.register_ensemble("test_fixture_2")

        # Test errors are thrown
        def bad_type_error():
            instance.register_ensemble("bad", 3)

        def bad_not_long_enough():
            instance.register_ensemble("bad", ensemble_not_long_enough)

        def bad_dim_error():
            instance.register_ensemble("bad", ensemble_bad_feature)

        self.assertRaises(AttributeError, bad_type_error)
        self.assertRaises(AttributeError, bad_not_long_enough)
        self.assertRaises(AttributeError, bad_dim_error)
    def test_configuration_assignment(self):
        """Tests that the configuration can be modified and assigned to without too much trouble"""
        python_instance = src.supertransformerlib.Core.EnsembleSpace(10)
        instance = torch.jit.script(python_instance)
        config_basic_good = torch.randn([5, 10])
        config_complex_good = torch.randn([12, 5, 10])

        config_primitive_bad = torch.randn([3, 5])
        config_moderate_bad = torch.randn([5, 4,20])

        instance.set_config(config_basic_good)
        instance.set_config(config_complex_good)

        def bad_config_type():
            python_instance.set_config(34)
        def bad_config_primitive():
            python_instance.set_config(config_primitive_bad)
        def bad_config_complex():
            python_instance.set_config(config_moderate_bad)

        self.assertRaises(ValueError, bad_config_type)
        self.assertRaises(ValueError, bad_config_primitive)
        self.assertRaises(ValueError, bad_config_complex)
    def test_load_ensemble_auto(self):
        """Test the ability to load a defined ensemble given a variety of configurations"""

        #Build the various test cases. This consists of combinations
        #of ensemble definitions along with configurations
        ensemble_width = 10

        ensemble_registration_cases = {
            "simple_ensemble_case" : torch.randn([ensemble_width, 5]),
            "complex_ensemble_case" : torch.randn([ensemble_width, 5, 3,6, 2])
        }
        configuration_cases = {
            "primitive_configuration" : torch.randn([1, ensemble_width]),
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
                instance = src.supertransformerlib.Core.EnsembleSpace(ensemble_width)
                instance.register_ensemble("test", ensemble_value)

                instance.set_config(config_value)
                output = instance.test
                self.assertTrue(expected_final_shape == output.shape)
                print(output.shape)
            except Exception as err:
                msg = "Error produced while testing ensemble case: %s with config case: %s"\
                      % (ensemble_name, config_name)
                raise Exception(msg) from err
    def test_load_ensemble_manual(self):
        """Test that load ensemble is working correctly for a few known, manually configured values"""
        epsilion = 0.0001

        off = -1e+8
        test_cases = [
            {"ensemble" : torch.tensor([[0,1.],[2,3]]),
             "config" : torch.tensor([[0.5, 0.5]]),
             "output" : torch.tensor([[1.0, 2.0]])
             },
            {"ensemble" : torch.tensor([[0,1],[2,3.]]),
             "config" : torch.tensor([[1, off], [0.5, 0.5], [off, 1]]),
             "output" : torch.tensor([[0, 1],[1.0, 2.0],[2, 3]])
             }       ,
            {"ensemble": torch.tensor([[0, 1], [2, 3.]]),
             "config": torch.tensor([[[0.5, 0.5]],[[off,1]]]),
             "output": torch.tensor([[[1.0, 2.0]],[[2, 3]]])
             },
            {"ensemble" : torch.tensor([[[0, 1.],[2,3]], [[3, 4],[4,5]]]),
             "config" : torch.tensor([[1, off], [0.5, 0.5], [off, 1]]),
             "output" : torch.tensor([[[0, 1,],[2,3]],[[1.5, 2.5], [3, 4]],[[3,4],[4,5]]])
             }
        ]


        for case in test_cases:
            instance = src.supertransformerlib.Core.EnsembleSpace(2)
            ensemble = case["ensemble"]
            config = case["config"]
            expectations = case["output"]

            instance.register_ensemble("test", ensemble)
            instance.set_config(config)
            test_result = instance.test

            diff = (test_result - expectations).abs()
            passed = torch.all(epsilion > diff)
            self.assertTrue(passed)

    def test_top_k(self):
        """ Test the top k selection ability"""
        k_num = 3
        ensemble_width = 10
        ensemble= torch.randn([ensemble_width, 5, 3,6, 10])
        config = torch.randn([12, 5, ensemble_width])
        instance = src.supertransformerlib.Core.EnsembleSpace(ensemble_width, k_num)
        instance.set_config(config)
        instance.register_ensemble("test", ensemble)
        instance.test
        instance.set_top_k(4)
        self.assertTrue(instance.get_top_k() == 4)
    def test_top_p(self):
        """Test the top p selection ability"""
        top_p = 0.7
        ensemble_width = 10
        ensemble = torch.randn([ensemble_width, 5, 3, 6, 10])
        config = torch.randn([12, 5, ensemble_width])
        expected_shape = torch.Size([12, 5, 5, 3, 6, 10])
        instance = src.supertransformerlib.Core.EnsembleSpace(ensemble_width, top_p=top_p)
        instance.set_config(config)
        instance.register_ensemble("test", ensemble)
        test_result = instance.test
        self.assertTrue(expected_shape == test_result.shape)
    def test_basic_subclassing(self):
        """Test that we can meaningfully subclass and that torchscript is happy"""
        class mockup(src.supertransformerlib.Core.EnsembleSpace):
            """
            Dynamically varying kernel
            Simple task: add
            """
            def __init__(self):
                super().__init__(5)

                self.d_model = 20
                self.batch_fun = 4
                self.kernel = torch.randn([self.native_ensemble_width, self.d_model])
                self.kernel = torch.nn.Parameter(self.kernel)
                self.register_ensemble("kernel")

            def forward(self, x: torch.Tensor)->torch.Tensor:
                config = torch.randn([self.batch_fun, self.native_ensemble_width])
                self.set_config(config, True)
                return x + self.kernel

        instance = mockup()
        tensor = torch.randn([instance.batch_fun, instance.d_model])
        assert hasattr(instance, "_configuration")
        instance = torch.jit.script(instance)
        instance(tensor)

    def test_topk_subclassing(self):
        """Test that we can meaningfully subclass when using topk"""
        class mockup(src.supertransformerlib.Core.EnsembleSpace):
            """
            Dynamically varying kernel
            Simple task: add
            """
            def __init__(self):
                super().__init__(5, top_k=2)

                self.d_model = 20
                self.batch_fun = 4
                self.kernel = torch.randn([self.native_ensemble_width, self.d_model])
                self.register_ensemble("kernel")

            def forward(self, x: torch.Tensor)->torch.Tensor:
                config = torch.randn([self.batch_fun, self.native_ensemble_width])
                self.set_config(config, True)
                return x + self.kernel

        instance = mockup()
        tensor = torch.randn([instance.batch_fun, instance.d_model])
        instance = torch.jit.script(instance)
        instance(tensor)

    def test_top_p_subclassing(self):
        """Test that we can meaningfully subclass when using top_p"""
        class mockup(src.supertransformerlib.Core.EnsembleSpace):
            """
            Dynamically varying kernel
            Simple task: add
            """
            def __init__(self):
                super().__init__(5, top_p=0.5)

                self.d_model = 20
                self.batch_fun = 4
                self.kernel = torch.randn([self.native_ensemble_width, self.d_model])
                self.register_ensemble("kernel")

            def forward(self, x: torch.Tensor)->torch.Tensor:
                config = torch.randn([self.batch_fun, self.native_ensemble_width])
                self.set_config(config, True)
                return x + self.kernel

        instance = mockup()
        tensor = torch.randn([instance.batch_fun, instance.d_model])
        instance = torch.jit.script(instance)
        instance(tensor)
    def test_buffer_save_load_compatible(self):
        """ Test the layer is still compatible with torch's save and load system """


        created_instance = buffer_mockup()
        statedict = created_instance.state_dict()
        torch.save(created_instance, "test_save.txt")
        loaded_instance = torch.load("test_save.txt")

        for param_in_created, param_in_loaded in zip(created_instance.parameters(), loaded_instance.parameters()):
            self.assertTrue(torch.all(param_in_created == param_in_loaded))


    def test_torchscript_assignment(self):
        """Test that all assignments work when using torchscript"""
        class mockup(src.supertransformerlib.Core.EnsembleSpace):
            """
            Dynamically varying kernel
            Simple task: add
            """
            def __init__(self):
                super().__init__(5)

                self.d_model = 20
                self.batch_fun = 4
                self.kernel = torch.randn([self.native_ensemble_width, self.d_model])
                self.kernel = torch.nn.Parameter(self.kernel)
                self.register_ensemble("kernel")

            def forward(self, x: torch.Tensor)->torch.Tensor:
                return x + self.kernel


        config = torch.randn([4, 5])
        instance = mockup()
        instance = torch.jit.script(instance)
        instance.set_config(config, False)
        self.assertTrue(torch.all(instance.get_config() == config))

class testLinear(unittest.TestCase):
    """
    This is the test feature for the linear layer.
    """

    def test_Regular(self):
        """ Tests if the standard pytorch linear layer is reproduced"""

        tensor = torch.rand([2, 5])
        tester = src.supertransformerlib.Core.Linear(5, 10)
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
    def test_dynamic_basic(self):
        """Test whether or not it is the case that the dynamic ensembling system works"""
        test_tensor = torch.randn([30, 20, 20])
        layer = src.supertransformerlib.Core.Linear(20, 10, dynamics=2)
#        layer = torch.jit.script(layer)

        configuration_1 = torch.tensor([[0, 1]])
        configuration_2 = torch.randn([20, 2])
        configuration_3 = torch.randn([30, 20, 2])
        configuration_4 = torch.randn([10, 30, 20, 2])

        layer.set_config(configuration_1)
        output_1 = layer(test_tensor)
        layer.set_config(configuration_2)
        output_2 = layer(test_tensor)
        layer.set_config(configuration_3)
        output_3 = layer(test_tensor)
        layer.set_config(configuration_4)
        output_4 = layer(test_tensor)

        self.assertTrue(torch.any(output_1 != output_2))
        self.assertTrue(torch.any(output_1 != output_3))
        self.assertTrue(output_1.shape != output_4.shape)