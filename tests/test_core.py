import textwrap
import unittest
from typing import Type

import torch
from torch import nn
import torch.nn
import itertools

import src.supertransformerlib.Basics
import src.supertransformerlib.Core
import src.supertransformerlib.Core as Core


### Fixtures ###
#
# These must be located at the top level so pickle
# is happy

class test_functions(unittest.TestCase):
    def test_dedent(self):
        """
        Test dedent, which is used for generating
        error messages. It must be torchscript compatible
        """

        message = """\
        This is a test
        
        It has extra space due to being defined inline. 
        These need to be removed by dedent.
        """

        expected = textwrap.dedent(message)
        gotten = Core.dedent(message)
        equivalent = expected == gotten
        self.assertTrue(equivalent, "textwrap dedent and torchscript dedent did not match.")

    def test_format(self):
        """
        Test that the torchscript formatting method works as required
        """
        replace = "potato"
        replace2 = "item"
        string_to_format = "  blah blah {replace} blah {replace2}   "

        expected = string_to_format.format(
            replace = replace,
            replace2 = replace2
        )
        received = Core.format(string_to_format,
                               {
                                   "replace" : replace,
                                   "replace2" : replace2
                               }
                               )
        self.assertTrue(expected == received)

    def test_standardize(self):
        """
        Test the ability of the standardize function to perform standardization.
        """
        print_error_messages = False
        tasks = ["Testing"]

        def test_should_succeed(input,
                                name,
                                allow_negatives,
                                allow_zeros,
                                tasks,
                                expected_result):

            #Standard
            output = Core.standardize_shape(input,
                                            name,
                                            allow_negatives,
                                            allow_zeros,
                                            tasks,
                                            )
            self.assertTrue(torch.all(output == expected_result))

            #Torchscript
            func = torch.jit.script(Core.standardize_shape)
            output =func(input,
                                            name,
                                            allow_negatives,
                                            allow_zeros,
                                            tasks,
                                            )
            self.assertTrue(torch.all(output == expected_result))

        def test_should_fail(input,
                             name,
                             allow_negatives,
                             allow_zeros,
                             tasks):

            try:
                func = torch.jit.script(Core.standardize_shape)
                output = func(input,
                              name,
                              allow_negatives,
                              allow_zeros,
                              tasks,
                              )
                raise RuntimeError("No error thrown")
            except torch.jit.Error as err:
                if print_error_messages:
                    print(err)

        #Define test cases

        basic_int = (1, "basic", False, False, tasks, torch.tensor([1]))
        basic_list = ([1, 2], "list", False, False, tasks, torch.tensor([1, 2]))
        basic_tensor = (torch.tensor([1, 2, 3]), "tensor", False, False, tasks, torch.tensor([1, 2, 3]))
        allow_negatives = (-1, "basic", True, True, tasks, torch.tensor([-1]))
        allow_zeros = (0, "zeros", False, True, tasks, torch.tensor([0]))

        input_less_than_zero = ([-1, 2], "bad_domain", False, False, tasks)
        input_equal_to_zero = ([0, 1, 2], "bad_domain", False, False, tasks)
        input_floating = (torch.tensor([0.2]), "bad_type", False, False, tasks)
        input_complex = (torch.tensor([0], dtype = torch.complex64), "bad_type", False, False, tasks)

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
        Test the ability of validate shape to correctly
        validate incoming shape information.
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
                self.assertTrue(False)
            except ValueError as err:
                pass
            #Torchscript

            func = torch.jit.script(Core.validate_shape_tensor)
            try:
                func(tensor)
                raise RuntimeError("Did not stop")
            except torch.jit.Error as err:
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

class test_AddressBook(unittest.TestCase):
    """

    Tests the address book. This is a memory manager built
    (ugh) in python which can be used to make pointers
    within an address space. The class is initialized
    with the space to address, which must be finite,
    and can then be passed nonnegative integers to associate
    with the address space.
    """

    def test_basics_series_constructor(self):
        """Test that the constructor works at all"""

        def test_should_pass(addresses: torch.Tensor):
            src.supertransformerlib.Core.AddressSpace(addresses)
        def test_should_fail(addresses: torch.Tensor, error: Type[Exception]):
            def to_fail():
                src.supertransformerlib.Core.AddressSpace(addresses)
            self.assertRaises(error, to_fail)

        
        #Should pass
        addresses_naive = torch.arange(120)
        addresses_shifted = torch.arange(120) + 30
        addresses_negative = torch.arange(120) - 10
        addresses_only_one = torch.tensor([0])

        #Should fail

        addresses_wrong_shape = torch.arange(1000).view(10, 10, 10)
        addresses_wrong_dtype = torch.randn(120)
        addresses_empty = torch.empty([0])

        #tests
        test_should_pass(addresses_naive)
        test_should_pass(addresses_shifted)
        test_should_pass(addresses_negative)
        test_should_pass(addresses_only_one)

        test_should_fail(addresses_wrong_shape, AssertionError)
        test_should_fail(addresses_wrong_dtype, AssertionError)
        test_should_fail(addresses_empty, AssertionError)

    def test_basics_series_malloc(self):
        """Test the memory allocator is working, ignoring any interactions with other methods."""

        addresses = torch.arange(120)
        def test_should_pass(pointer_ids: torch.Tensor):
            addressbook = src.supertransformerlib.Core.AddressSpace(addresses)
            addressbook.malloc(pointer_ids)

            addressbook = src.supertransformerlib.Core.AddressSpace(addresses)
            addressbook = torch.jit.script(addressbook)
            addressbook.malloc(pointer_ids)

        def test_should_pass_double(
                                    pointers1: torch.Tensor,
                                    pointers2: torch.Tensor):
            addressbook = src.supertransformerlib.Core.AddressSpace(addresses)
            addressbook.malloc(pointers1)
            addressbook.malloc(pointers2)

            addressbook = src.supertransformerlib.Core.AddressSpace(addresses)
            addressbook = torch.jit.script(addressbook)
            addressbook.malloc(pointers1)
            addressbook.malloc(pointers2)

        def test_should_fail(
                             pointer_ids: torch.Tensor,
                             error: Type[Exception]):

            def to_fail():
                addressbook = src.supertransformerlib.Core.AddressSpace(addresses)
                addressbook.malloc(pointer_ids)
            self.assertRaises(error, to_fail)

            def to_fail():
                addressbook = src.supertransformerlib.Core.AddressSpace(addresses)
                addressbook = torch.jit.script(addressbook)
                try:
                    addressbook.malloc(pointer_ids)
                except torch.jit.Error:
                    raise error

            self.assertRaises(error, to_fail)

        def test_should_fail_double(
                             pointers1: torch.Tensor,
                             pointers2: torch.Tensor,
                             error: Type[Exception]):
            def to_fail():
                addressbook = src.supertransformerlib.Core.AddressSpace(addresses)
                addressbook.malloc(pointers1)
                addressbook.malloc(pointers2)

            self.assertRaises(error, to_fail)

            def to_fail():
                addressbook = src.supertransformerlib.Core.AddressSpace(addresses)
                addressbook = torch.jit.script(addressbook)
                try:
                    addressbook.malloc(pointers1)
                    addressbook.malloc(pointers2)
                except torch.jit.Error:
                    raise error

            self.assertRaises(error, to_fail)

        #Should pass

        completely_full = torch.arange(120, dtype=torch.int64)
        shifted = torch.arange(120, dtype=torch.int64) + 1000
        partially_full = torch.arange(30, dtype=torch.int64)

        double_a_full, double_b_full = torch.arange(100, dtype=torch.int64), torch.arange(20, dtype=torch.int64) + 100
        partial_a_full, partial_b_full = torch.arange(20, dtype=torch.int64), torch.arange(20, dtype=torch.int64) + 20

        #Should fail

        overfull = torch.arange(130, dtype=torch.int64)
        double_overfull_a, double_overfull_b = torch.arange(20, dtype=torch.int64), torch.arange(130, dtype=torch.int64) + 20
        attempt_reassign_a, attempt_reassign_b = torch.arange(120, dtype=torch.int64), torch.arange(120, dtype=torch.int64) #Notably, should warn.
        wrong_dtype = torch.arange(120, dtype=torch.int16)

        #Tests

        test_should_pass(completely_full)
        test_should_pass(shifted)
        test_should_pass(partially_full)

        test_should_pass_double(double_a_full, double_b_full)
        test_should_pass_double(partial_a_full, partial_b_full)

        test_should_fail(overfull, RuntimeError)
        test_should_fail(wrong_dtype, AssertionError)
        test_should_fail_double(double_overfull_a, double_overfull_b, RuntimeError)
        test_should_fail_double(attempt_reassign_a, attempt_reassign_b, RuntimeError)

    def test_basic_series_dereference(self):
        """Test that dereference is working properly"""

        addresses = torch.arange(120)
        malloc_pointers = torch.arange(40)

        def test_should_succeed(pointers: torch.Tensor, expected: torch.Tensor):

            #Standard
            addressbook = src.supertransformerlib.Core.AddressSpace(addresses)
            addressbook.malloc(malloc_pointers)
            output = addressbook.dereference(pointers)
            self.assertTrue(torch.all(expected == output))

            #Torchscript
            addressbook = src.supertransformerlib.Core.AddressSpace(addresses)
            addressbook = torch.jit.script(addressbook)
            addressbook.malloc(malloc_pointers)
            output = addressbook.dereference(pointers)
            self.assertTrue(torch.all(expected == output))

        def test_should_fail(pointers: torch.Tensor, error: Type[Exception]):

            #Standard

            def to_fail():
                addressbook = src.supertransformerlib.Core.AddressSpace(addresses)
                addressbook.malloc(malloc_pointers)
                output = addressbook.dereference(pointers)

            self.assertRaises(error, to_fail)

            #Torchscript
            def to_fail():
                addressbook = src.supertransformerlib.Core.AddressSpace(addresses)
                addressbook = torch.jit.script(addressbook)
                try:
                    addressbook.malloc(malloc_pointers)
                    output = addressbook.dereference(pointers)
                except torch.jit.Error:
                    raise error("mockup", torch.tensor([3]))

            self.assertRaises(error, to_fail)

        #Should pass

        partial, partial_expected = torch.arange(20), torch.arange(20)
        full, full_expected = torch.arange(40), torch.arange(40)
        out_of_order, out_of_order_expected = torch.arange(20).flip(dims=[-1]), torch.arange(20).flip(dims=[-1])
        nd, nd_expected = torch.arange(40).view(5, 2, 4), torch.arange(40).view(5,2,4)

        #Should fail

        null_ptr = torch.arange(40) + 1
        invalid_ptr = torch.arange(40) - 1
        bad_dtype = torch.ones([10], dtype=torch.int32)

        #Tests

        test_should_succeed(partial, partial_expected)
        test_should_succeed(full, full_expected)
        test_should_succeed(out_of_order, out_of_order_expected)
        test_should_succeed(nd, nd_expected)

        test_should_fail(null_ptr, src.supertransformerlib.Core.NullPtr)
        test_should_fail(invalid_ptr, AssertionError)
        test_should_fail(bad_dtype, AssertionError)


class test_DynamicKernelReservoir(unittest.TestCase):
    """
    Test the dynamic kernel reservoir system.

    Verify it will work correctly.
    """
    def test_constructor(self):

        def successful_constructor(
                kernel_shape,
                resevoir_size,
                mode
        ):

            layer = Core.DynamicKernelReservoir(kernel_shape, resevoir_size, mode)
            layer = torch.jit.script(layer)

        def fail_constructor(
                kernel_shape,
                reservoir_size,
                mode,
        ):

            try:
                layer = Core.DynamicKernelReservoir(kernel_shape, reservoir_size, mode)
                raise RuntimeError("Did not raise an error")
            except Core.ReservoirError as err:
                pass

        #Setup parameters for test
        succeed_a = (1, 1, None)
        succeed_b = ([10, 10], 20, "sparse")
        succeed_c = ([10, 10, 4, 3], 5, "dense")

        fail_kerneltype = ("potato", 20, None)
        fail_kernelshape = ([-10, 5], 20, None)
        fail_mode = ([10, 10], 5, "illegal")
        fail_reservoir_shape = ([10, 10], -30, None)

        #Run tests

        successful_constructor(*succeed_a)
        successful_constructor(*succeed_b)
        successful_constructor(*succeed_c)

        fail_constructor(*fail_kerneltype)
        fail_constructor(*fail_kernelshape)
        fail_constructor(*fail_mode)
        fail_constructor(*fail_reservoir_shape)

    def test_validate_raw_key(self):
        """ Test that the mechanism for validating a key covered all the bases"""

        shape = [10, 10, 10]
        mode = "id"
        reservoir = 10
        layer = Core.DynamicKernelReservoir(shape, reservoir, mode)

        def successful_test(key: torch.Tensor):

            #Standard

            layer._validate_raw_key("Testing", key)

            #torchscript
            func = torch.jit.script_method(layer._validate_raw_key)


        def failure_test(key: torch.Tensor, torchscript = True):

            #Standard

            try:
                layer._validate_raw_key("Testing", key)
            except Core.ReservoirKeyError as err:
                print(err)

            #Torchscript

            if torchscript:
                func = torch.jit.script_method(layer._validate_raw_key)

        #Define cases

        good_key_a = torch.arange(7)
        good_key_b = torch.randint(0, 10, [10, 10, 10])

        bad_type = "Potato"
        bad_dtype = torch.arange(3).to(dtype=torch.float32)
        key_too_low = torch.arange(10) -1
        key_too_high = torch.arange(10) + 1

        #Run tests

        print("Test error contents")
        successful_test(good_key_a)
        successful_test(good_key_b)

        failure_test(bad_type, False)
        failure_test(bad_dtype)
        failure_test(key_too_low)
        failure_test(key_too_high)

    def test_validate_key_active(self):
        """Test that we can successfully see when a key is active"""

        shape = [10, 10, 10]
        mode = "id"
        reservoir = 10
        layer = Core.DynamicKernelReservoir(shape, reservoir, mode)
        layer._active[torch.arange(5)] = True

        def successful_test(key: torch.Tensor):

            #Standard

            layer._validate_key_active("Testing", "Doing testing", key)

            #torchscript
            func = torch.jit.script_method(layer._validate_raw_key)

        def failure_test(key: torch.Tensor):

            #Standard

            try:
                layer._validate_key_active("Testing", "Doing testing", key)
                raise RuntimeError("Did not throw")
            except Core.ReservoirKeyError as err:
                pass

        #Define cases

        good_key_a = torch.arange(2)
        good_key_b = torch.arange(5)
        good_key_c = torch.randint(0, 5, [10, 10])

        bad_key_mix = torch.arange(6)
        bad_key_many=  torch.randint(0, 10, [10, 10])
        bad_excludively = torch.randint(5, 10, [10])

        #Run tests

        successful_test(good_key_a)
        successful_test(good_key_b)
        successful_test(good_key_c)

        failure_test(bad_key_mix)
        failure_test(bad_key_many)
        failure_test(bad_excludively)

    def test_validate_key_inactive(self):
        """Test that we can validate when a key is inactive. Test vi"""

        shape = [10, 10, 10]
        mode = "id"
        reservoir = 10
        layer = Core.DynamicKernelReservoir(shape, reservoir, mode)
        layer._active[torch.arange(5)] = True

        def successful_test(key: torch.Tensor):

            #Standard

            layer._validate_key_inactive("Testing", "Doing testing", key)

            #torchscript
            func = torch.jit.script_method(layer._validate_raw_key)

        def failure_test(key: torch.Tensor):

            #Standard

            try:
                layer._validate_key_inactive("Testing", "Doing testing", key)
                raise RuntimeError("Did not throw")
            except Core.ReservoirKeyError as err:
                pass

        #Define cases

        good_key_a = torch.arange(5, 10)
        good_key_b = torch.arange(7, 10)
        good_key_c = torch.randint(5, 10, [10, 10])

        bad_key_mix = torch.arange(10)
        bad_key_complex_mix = torch.randint(0, 10, [10, 10, 10])
        bad_exclusively = torch.randint(0, 5, [10, 10])
        #Run tests

        successful_test(good_key_a)
        successful_test(good_key_b)
        successful_test(good_key_c)

        failure_test(bad_key_mix)
        failure_test(bad_key_complex_mix)
        failure_test(bad_exclusively)

    def test_validate_kernel(self):
        """Test that the validate kernel mechanism is sane"""

        shape = [10, 10, 10]
        mode = "id"
        reservoir = 10
        layer = Core.DynamicKernelReservoir(shape, reservoir, mode)
        layer._active[torch.arange(5)] = True

        def test_successful(kernel: torch.Tensor):

            layer._validate_kernel("Testing", kernel)

        def test_failure(kernel: torch.Tensor):

            try:
                layer._validate_kernel("Testing", kernel)
                raise RuntimeError("Did not throw")
            except Core.ReservoirKernelError as err:
                print(err)

        #Define cases

        kernel_valid = torch.randn([5, 10, 10, 10])

        wrong_type = "illegal"
        wrong_shape = torch.randn([5, 3, 3, 10])
        wrong_dtype = torch.arange(5000).reshape(5, 10, 10, 10)

        #Run some tests

        test_successful(kernel_valid)
        test_failure(wrong_type)
        test_failure(wrong_shape)
        test_failure(wrong_dtype)

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

class test_ViewPoint(unittest.TestCase):
    """
    Test the viewpoint mechanism.

    Verify it currently functions correctly
    using manually designed tests with known
    results.
    """
    def test_basic_viewpoint(self):

        # Define situation
        #
        #
        # Let there be two views.
        # Let there be one query
        # Let there be two batches
        #

        # Construct a batch to sample.

        tensor_batch_1 = torch.tensor([[1, 0], [0, 1], [0.5, 0.5]])
        tensor_batch_2 = torch.tensor([[2, 0,], [0, 3], [1.0, 1.0]])
        tensor = torch.stack([tensor_batch_1, tensor_batch_2])

        # Define sampling index.
        #
        # Let there be two views.
        # Let there be one query
        # Let there be two batches
        #
        # For the query:
        # Let the first batch view from elements 0, 1. and again 0, 1,
        # Let the second batch view from elements 0, 1, and then 1, 2

        # Index_shape: (..., view, query, top_k)
        index_batch_1 = torch.tensor([[0, 1],[0, 1]]).unsqueeze(-2)
        index_batch_2 = torch.tensor([[0, 1], [1, 2]]).unsqueeze(-2)
        index = torch.stack([index_batch_1, index_batch_2])

        # Define weights
        #
        #
        # Let there be two views.
        # Let there be one query
        # Let there be two batches
        #
        # On batch one,
        # Let view 1 weight element 0 to max, view 2 weight element 1 to max
        # On batch two
        # Let view 1 weight both elements equally, let view 2 weight both elements equally

        weights_batch_1 = torch.tensor([[1.0, 0], [0, 1.0]]).unsqueeze(-2)
        weights_batch_2 = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).unsqueeze(-2)
        weights = torch.stack([weights_batch_1, weights_batch_2])

        # Construct expected output.
        #
        # It is the case I expect to draw a window of size three for each view, and
        # then put the views together for each batch. Construct each subview, then
        # each view, then each batch.

        batch1_view1_k1_sample = torch.tensor([[0, 0], [1., 0], [0, 1]])
        batch1_view1_k2_sample = torch.tensor([[1, 0],[0,1],[0.5, 0.5]])
        batch1_view1_sample = torch.stack([batch1_view1_k1_sample, batch1_view1_k2_sample], dim=-3)

        batch1_view2_k1_sample = torch.tensor([[0, 0], [1., 0], [0, 1]])
        batch1_view2_k2_sample = torch.tensor([[1, 0],[0,1],[0.5, 0.5]])
        batch1_view2_sample = torch.stack([batch1_view2_k1_sample, batch1_view2_k2_sample], dim=-3)

        batch1_sample = torch.stack([batch1_view1_sample, batch1_view2_sample], dim=0)

        batch2_view1_k1_sample = torch.tensor([[0, 0],[2, 0], [0, 3.0]])
        batch2_view1_k2_sample = torch.tensor([[2, 0],[0, 3],[1.0, 1.0]])

        batch2_view1_sample = torch.stack([batch2_view1_k1_sample, batch2_view1_k2_sample], dim=-3)

        batch2_view2_k1_sample = torch.tensor([[2, 0],[0, 3],[1.0, 1.0]])
        batch2_view2_k2_sample = torch.tensor([[0, 3],[1.0, 1.0], [0, 0]])

        batch2_view2_sample = torch.stack([batch2_view2_k1_sample, batch2_view2_k2_sample], dim=-3)

        batch2_sample = torch.stack([batch2_view1_sample, batch2_view2_sample], dim=0)
        tensor_sample = torch.stack([batch1_sample, batch2_sample], dim=0)
        tensor_sample = tensor_sample.unsqueeze(-4)

        #Weight the tensor sample. Eliminate k
        tensor_sample = tensor_sample*weights.unsqueeze(-1).unsqueeze(-1)
        expected_tensor = tensor_sample.sum(dim=-3)

        # Run process

        viewer = src.supertransformerlib.Basics.ViewPoint(
            views=2,
            view_width=3,
            weights=weights,
            index=index
        )

        #Verify gather and such logic is working correctly
        output = viewer(tensor)
        self.assertTrue(torch.all(expected_tensor==output))

class test_ViewpointFactory(unittest.TestCase):
    def test_constructor(self):
        """test that the constructor works at all"""

        src.supertransformerlib.Basics.ViewPointFactory(32, 32, 8, 20, 4)


    def test_viewpoint_shape(self):
        query_tensor = torch.randn([2, 3, 32])
        text_tensor = torch.randn([2, 10, 32])
        factory = src.supertransformerlib.Basics.ViewPointFactory(32, 32, 8, 5, 4)
        viewpoint = factory(query_tensor, text_tensor)
        expected_shape = torch.Size([2, 8, 3, 5, 32])

        outcome = viewpoint(text_tensor)
        self.assertTrue(expected_shape == outcome.shape)

