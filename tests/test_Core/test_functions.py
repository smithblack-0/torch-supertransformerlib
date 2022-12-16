"""
Test features for some of the simple
functions in core

"""


import unittest
import torch
from src.supertransformerlib import Core

class TestFunctions(unittest.TestCase):
    """
    The test suite for functions
    """
    def test_standardize(self):
        """
        Test the ability of the standardize function to perform standardization.
        """
        print_error_messages = True
        task = "Performing Testing"

        def test_should_succeed(input,
                                name,
                                allow_negatives,
                                allow_zeros,
                                task,
                                expected_result):

            #Standard
            output = Core.standardize_shape(input,
                                                                              name,
                                                                              allow_negatives,
                                                                              allow_zeros,
                                                                              task,
                                                                              )
            self.assertTrue(torch.all(output == expected_result))

            #Torchscript
            func = torch.jit.script(Core.standardize_shape)
            output =func(input,
                                            name,
                                            allow_negatives,
                                            allow_zeros,
                                            task,
                                            )
            self.assertTrue(torch.all(output == expected_result))

        def test_should_fail(input,
                             name,
                             allow_negatives,
                             allow_zeros,
                             task):

            try:
                func = torch.jit.script(Core.standardize_shape)
                func(input,
                              name,
                              allow_negatives,
                              allow_zeros,
                              task,
                              )
                raise RuntimeError("No error thrown")
            except torch.jit.Error as err:
                if print_error_messages:
                    print(err)

        #Define test cases

        basic_int = (1, "basic", False, False, task, torch.tensor([1]))
        basic_list = ([1, 2], "list", False, False, task, torch.tensor([1, 2]))
        basic_tensor = (torch.tensor([1, 2, 3]), "tensor", False, False, task,
                        torch.tensor([1, 2, 3]))
        allow_negatives = (-1, "basic", True, True, task, torch.tensor([-1]))
        allow_zeros = (0, "zeros", False, True, task, torch.tensor([0]))

        input_less_than_zero = ([-1, 2], "bad_domain", False, False, task)
        input_equal_to_zero = ([0, 1, 2], "bad_domain", False, False, task)
        input_floating = (torch.tensor([0.2]), "bad_type", False, False, task)
        input_complex = (torch.tensor([0], dtype = torch.complex64), "bad_type", False, False, task)

        #Run tests
        test_should_succeed(*basic_int)
        test_should_succeed(*basic_list)
        test_should_succeed(*basic_tensor)
        test_should_succeed(*allow_negatives)
        test_should_succeed(*allow_zeros)

        test_should_fail(*input_less_than_zero)
        test_should_fail(*input_equal_to_zero)
        test_should_fail(*input_floating)
        test_should_fail(*input_complex)


    def test_validate_shape(self):
        """
        Test the ability of validate dynamic_shape to correctly
        validate incoming dynamic_shape information.
        """
        #Test fixtures

        def test_valid(tensor: torch.Tensor):
            #Standard
            Core.validate_shape_tensor(tensor)

            #torchscript

            func = torch.jit.script(Core.validate_shape_tensor)
            func(tensor)


        def test_invalid(tensor: torch.Tensor):

            #Standard
            try:
                Core.validate_shape_tensor(tensor)
                raise RuntimeError("Did not throw")
            except ValueError:
                pass
            #Torchscript

            func = torch.jit.script(Core.validate_shape_tensor)
            try:
                func(tensor)
                raise RuntimeError("Did not stop")
            except torch.jit.Error:
                pass

        #Valid cases

        good_one = torch.tensor([1])
        good_two = torch.tensor([1, 10, 20])

        #Invalid cases

        invalid_shape = torch.randint(0, 10, [1, 10])
        invalid_dtype = torch.randn([10])
        invalid_dim = torch.tensor([-5, 2])

        #Run tests

        test_valid(good_one)
        test_valid(good_two)

        test_invalid(invalid_shape)
        test_invalid(invalid_dtype)
        test_invalid(invalid_dim)

    def test_validate_string_in_options(self):

        def test_succeed(string, stringname, options, optionsname):

            #Test standard
            Core.validate_string_in_options(string, stringname, options, optionsname)

            #Test torchscript
            func = torch.jit.script(Core.validate_string_in_options)
            func(string, stringname, options, optionsname)

        def test_failure(string, stringname, options, optionsname):
            #Test standard

            try:
                Core.validate_string_in_options(string, stringname, options, optionsname)
            except ValueError as err:
                print(err)

            #Test torchscript
            func = torch.jit.script(Core.validate_string_in_options)
            try:
                func(string, stringname, options, optionsname)
            except torch.jit.Error:
                pass
            except RuntimeError:
                pass


        #Define cases

        succeed = ("standard", "mode", ("standard", "advanced"), "modes")
        wrong_type = (10, "mode", ("standard", "advanced"), "modes")
        not_in = ("orthogonal", "mode", ("standard", "advanced"), "modes")

        test_succeed(*succeed)
        test_failure(*wrong_type)
        test_failure(*not_in)
