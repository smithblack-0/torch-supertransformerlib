import unittest
import torch
import itertools

from src.supertransformerlib import DynamicEnsemble

class test_EnsembleSpace(unittest.TestCase):
    """

    Test fixture for the base ensemblespace

    """
    def test_ensemble_registration(self):
        """ Test the ability to register kernels as ensemble features. """
        instance = DynamicEnsemble.EnsembleSpace(10)
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
        instance = DynamicEnsemble.EnsembleSpace(10)
        config_basic_good = torch.randn([5, 10])
        config_complex_good = torch.randn([12, 5, 10])

        config_primitive_bad = torch.randn([3, 5])
        config_moderate_bad = torch.randn([5, 4,20])

        instance.configuration = config_basic_good
        instance.configuration = config_complex_good

        def bad_config_type():
            instance.configuration = 34
        def bad_config_primitive():
            instance.configuration = config_primitive_bad
        def bad_config_complex():
            instance.configuration = config_moderate_bad

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
                instance = DynamicEnsemble.EnsembleSpace(ensemble_width)
                instance.register_ensemble("test", ensemble_value)
                instance.configuration = config_value
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

        test_cases = [
            {"ensemble" : torch.tensor([[0,1.],[2,3]]),
             "config" : torch.tensor([[0.5, 0.5]]),
             "output" : torch.tensor([[1.0, 2.0]])
             },
            {"ensemble" : torch.tensor([[0,1],[2,3.]]),
             "config" : torch.tensor([[1, 0], [0.5, 0.5], [0, 1]]),
             "output" : torch.tensor([[0, 1],[1.0, 2.0],[2, 3]])
             }       ,
            {"ensemble": torch.tensor([[0, 1], [2, 3.]]),
             "config": torch.tensor([[[0.5, 0.5]],[[0,1]]]),
             "output": torch.tensor([[[1.0, 2.0]],[[2, 3]]])
             },
            {"ensemble" : torch.tensor([[[0, 1.],[2,3]], [[3, 4],[4,5]]]),
             "config" : torch.tensor([[1, 0], [0.5, 0.5], [0, 1]]),
             "output" : torch.tensor([[[0, 1,],[2,3]],[[1.5, 2.5], [3, 4]],[[3,4],[4,5]]])
             }
        ]


        for case in test_cases:
            instance = DynamicEnsemble.EnsembleSpace(2)
            ensemble = case["ensemble"]
            config = case["config"]
            expectations = case["output"]

            instance.configuration = config
            instance.register_ensemble("test", ensemble)
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
        instance = DynamicEnsemble.EnsembleSpace(ensemble_width, k_num)
        instance.configuration = config
        instance.register_ensemble("test", ensemble)
        instance.test
    def test_top_p(self):
        """Test the top p selection ability"""
        top_p = 0.7
        ensemble_width = 10
        ensemble = torch.randn([ensemble_width, 5, 3, 6, 10])
        config = torch.randn([12, 5, ensemble_width])
        expected_shape = torch.Size([12, 5, 5, 3, 6, 10])
        instance = DynamicEnsemble.EnsembleSpace(ensemble_width, top_p=top_p)
        instance.configuration = config
        instance.register_ensemble("test", ensemble)
        test_result = instance.test
        self.assertTrue(expected_shape == test_result.shape)
    def test_basic_subclassing(self):
        """Test that we can meaningfully subclass and that torchscript is happy"""
        class mockup(DynamicEnsemble.EnsembleSpace):
            """
            Dynamically varying kernel
            Simple task: add
            """
            def __init__(self):
                super().__init__(5)

                self.d_model = 20
                self.batch_fun = 4
                self.kernel = torch.randn([self.native_ensemble_width, self.d_model])
                self.register_ensemble("kernel")

            def forward(self, x: torch.Tensor)->torch.Tensor:
                config = torch.randn([self.batch_fun, self.native_ensemble_width])
                self.configuration = config
                return x + self.kernel

        instance = mockup()
        tensor = torch.randn([instance.batch_fun, instance.d_model])
        instance = torch.jit.script(instance)
        instance(tensor)
    def test_topk_subclassing(self):
        """Test that we can meaningfully subclass when using topk"""
        class mockup(DynamicEnsemble.EnsembleSpace):
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
                self.configuration = config
                return x + self.kernel

        instance = mockup()
        tensor = torch.randn([instance.batch_fun, instance.d_model])
        instance = torch.jit.script(instance)
        instance(tensor)

    def test_top_p_subclassing(self):
        """Test that we can meaningfully subclass when using top_p"""
        class mockup(DynamicEnsemble.EnsembleSpace):
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
                self.configuration = config
                return x + self.kernel

        instance = mockup()
        tensor = torch.randn([instance.batch_fun, instance.d_model])
        instance = torch.jit.script(instance)
        instance(tensor)