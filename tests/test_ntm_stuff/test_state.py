import unittest
from typing import Callable, Tuple

import torch
from supertransformerlib.NTM import state_utilities

PRINT_ERRORS_MESSAGES = True

class TestImmutableDict(unittest.TestCase):
    """
    Tests the small immutable dictionary class works properly
    and behaves like a dictionary.
    """

    def test_init(self):
        data = {'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])}
        imm_dict = state_utilities.ImmutableDict(data)
        self.assertEqual(imm_dict['a'].tolist(), [1, 2, 3])
        self.assertEqual(imm_dict['b'].tolist(), [4, 5, 6])
        self.assertEqual(len(imm_dict), 2)
        self.assertEqual(str(imm_dict), "{'a': tensor([1, 2, 3]), 'b': tensor([4, 5, 6])}")
        self.assertEqual(repr(imm_dict), "{'a': tensor([1, 2, 3]), 'b': tensor([4, 5, 6])}")

    def test_immutable(self):
        data = {'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])}
        imm_dict = state_utilities.ImmutableDict(data)
        with self.assertRaises(TypeError):
            imm_dict['a'] = torch.tensor([7, 8, 9])
        with self.assertRaises(TypeError):
            imm_dict['c'] = torch.tensor([10, 11, 12])

    def test_set(self):
        data = {'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])}
        imm_dict = state_utilities.ImmutableDict(data)
        new_dict = imm_dict.set('a', torch.tensor([7, 8, 9]))
        self.assertEqual(new_dict['a'].tolist(), [7, 8, 9])
        self.assertEqual(new_dict['b'].tolist(), [4, 5, 6])
        self.assertEqual(imm_dict['a'].tolist(), [1, 2, 3])
        self.assertEqual(len(new_dict), 2)
        self.assertEqual(len(imm_dict), 2)

    def test_hash(self):
        data = {'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])}
        imm_dict = state_utilities.ImmutableDict(data)
        hash1 = hash(imm_dict)
        data2 = {'a': torch.tensor([7, 8, 9]), 'b': torch.tensor([4, 5, 6])}
        imm_dict2 = state_utilities.ImmutableDict(data2)
        hash2 = hash(imm_dict2)
        self.assertNotEqual(hash1, hash2)

    def test_items(self):
        # create test data
        data = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6]), "c": torch.tensor([7, 8, 9])}
        # create ImmutableDict instance
        immutable_dict = state_utilities.ImmutableDict(data)
        # get the items as a list of tuples
        items = immutable_dict.items()
        # check that the length is correct
        assert len(items) == len(data)
        # check that each item is a tuple containing a key-value pair from the original dictionary
        for key, value in items:
            assert key in data
            assert torch.equal(value, data[key])

    def test_keys(self):
        # create test data
        data = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6]), "c": torch.tensor([7, 8, 9])}
        # create ImmutableDict instance
        immutable_dict = state_utilities.ImmutableDict(data)
        # get the keys as a list
        keys = immutable_dict.keys()
        # check that the length is correct
        assert len(keys) == len(data)
        # check that each key is in the original dictionary
        for key in keys:
            assert key in data

    def test_values(self):
        # create test data
        data = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6]), "c": torch.tensor([7, 8, 9])}
        # create ImmutableDict instance
        immutable_dict = state_utilities.ImmutableDict(data)
        # get the values as a list
        values = immutable_dict.values()
        # check that the length is correct
        assert len(values) == len(data)
        # check that each value is in the original dictionary
        for value in values:
            assert any(torch.equal(value, x) for x in data.values())

    def test_update(self):
        # create test data
        data1 = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])}
        data2 = {"c": torch.tensor([7, 8, 9]), "d": torch.tensor([10, 11, 12])}
        # create ImmutableDict instances
        immutable_dict1 = state_utilities.ImmutableDict(data1)
        immutable_dict2 = state_utilities.ImmutableDict(data2)
        # update the first dictionary with the second
        updated_dict = immutable_dict1.update(immutable_dict2)
        # check that the updated dictionary contains all key-value pairs from both original dictionaries
        for key, value in data1.items():
            assert key in updated_dict
            assert torch.equal(updated_dict[key], value)
        for key, value in data2.items():
            assert key in updated_dict
            assert torch.equal(updated_dict[key], value)

class TestStorage(unittest.TestCase):
    """ Test the storage feature, primarily by testing the validation"""

    def test_dtype_validation(self):
        memory = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
        read_weights = state_utilities.ImmutableDict({
            'reader1': torch.zeros((2, 3, 4), dtype=torch.float32),
            'reader2': torch.zeros((2, 3, 4), dtype=torch.float32),
        })
        write_weights = state_utilities.ImmutableDict({
            'writer1': torch.zeros((2, 3, 4), dtype=torch.float32),
            'writer2': torch.zeros((2, 3, 4), dtype=torch.float32),
        })
        read_defaults = state_utilities.ImmutableDict({
            'reader1': torch.zeros((3, 4), dtype=torch.float32),
            'reader2': torch.zeros((3, 4), dtype=torch.float32),
        })
        write_defaults = state_utilities.ImmutableDict({
            'writer1': torch.zeros((3, 4), dtype=torch.float32),
            'writer2': torch.zeros((3, 4), dtype=torch.float32),
        })

        # Test that a value error is raised when an entry in read_weights is not the same dtype as memory
        with self.assertRaises(ValueError):
            bad_read_weights = read_weights.set('reader1', torch.zeros((2, 3, 4), dtype=torch.int32))
            storage = state_utilities.Storage(memory, bad_read_weights, write_weights, read_defaults, write_defaults)

        # Test that a value error is raised when an entry in write_weights is not the same dtype as memory
        with self.assertRaises(ValueError):
            bad_write_weights = write_weights.set('writer2', torch.zeros((2, 3, 4), dtype=torch.int32))
            storage = state_utilities.Storage(memory, read_weights, bad_write_weights, read_defaults, write_defaults)

        # Test that a value error is raised when an entry in read_defaults is not the same dtype as memory
        with self.assertRaises(ValueError):
            bad_read_defaults = read_defaults.set('reader2', torch.zeros((3, 4), dtype=torch.int32))
            storage = state_utilities.Storage(memory, read_weights, write_weights, bad_read_defaults, write_defaults)

        # Test that a value error is raised when an entry in write_defaults is not the same dtype as memory
        with self.assertRaises(ValueError):
            bad_write_defaults = write_defaults.set('writer1', torch.zeros((3, 4), dtype=torch.int32))
            storage = state_utilities.Storage(memory, read_weights, write_weights, read_defaults, bad_write_defaults)

    def test_common_keys_validation(self):
        # Initialize valid inputs
        memory = torch.randn((1, 2, 3, 4))
        read_weights = state_utilities.ImmutableDict({"reader1": torch.randn((1, 2, 3, 4)), "reader2": torch.randn((1, 2, 3, 4))})
        write_weights = state_utilities.ImmutableDict(
            {"writer1": torch.randn((1, 2, 3, 4)), "writer2": torch.randn((1, 2, 3, 4))})
        read_defaults = state_utilities.ImmutableDict({"reader1": torch.randn((2, 3, 4)), "reader2": torch.randn((2, 3, 4))})
        write_defaults = state_utilities.ImmutableDict({"writer1": torch.randn((2, 3, 4)), "writer2": torch.randn((2, 3, 4))})

        # Check that common keys are properly validated between read_weights and read_defaults
        with self.assertRaises(ValueError):
            bad_read_defaults = read_defaults.remove("reader1")
            state_utilities.Storage(memory, read_weights, write_weights, bad_read_defaults, write_defaults, validate=True)

        # Check that common keys are properly validated between write_weights and write_defaults
        with self.assertRaises(ValueError):
            bad_write_weights = write_weights.remove("writer2")
            state_utilities.Storage(memory, read_weights, bad_write_weights, read_defaults, write_defaults, validate=True)

    def test_validate_weights_shape_sane(self):
        # Test that the batch size mismatches between memory and weights are caught

        # Setup: create inputs with valid shapes
        memory = torch.randn(5, 7, 6, 8)
        read_weights = state_utilities.ImmutableDict({'reader1': torch.randn(5, 7, 2, 6), 'reader2': torch.randn(5, 7, 2, 6)})
        write_weights = state_utilities.ImmutableDict({'writer1': torch.randn(5, 7, 2, 6), 'writer2': torch.randn(5, 7, 2, 6)})
        read_defaults = state_utilities.ImmutableDict({'reader1': torch.randn(2, 6), 'reader2': torch.randn(2, 6)})
        write_defaults = state_utilities.ImmutableDict({'writer1': torch.randn(2, 6), 'writer2': torch.randn(2, 6)})
        storage = state_utilities.Storage(memory, read_weights, write_weights, read_defaults, write_defaults)

        # Test with valid shapes - should not raise an error
        storage._check_weights_shape_sane("read weight", memory, read_weights)
        storage._check_weights_shape_sane("write weight", memory, write_weights)

        # Test with invalid shapes - should raise an error
        invalid_read_weights = read_weights.set('reader1', torch.randn(3, 7, 2, 6))
        with self.assertRaises(ValueError):
            storage._check_weights_shape_sane("read weight", memory, invalid_read_weights)

        invalid_write_weights = write_weights.set('writer2', torch.randn(5, 7, 2, 7))
        with self.assertRaises(ValueError):
            storage._check_weights_shape_sane("write weight", memory, invalid_write_weights)

    def test_jit_compiles(self):
        memory = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
        read_weights = state_utilities.ImmutableDict({
            'reader1': torch.zeros((2, 3, 7, 4), dtype=torch.float32),
            'reader2': torch.zeros((2, 3, 7, 4), dtype=torch.float32),
        })
        write_weights = state_utilities.ImmutableDict({
            'writer1': torch.zeros((2, 3, 7, 4), dtype=torch.float32),
            'writer2': torch.zeros((2, 3, 7, 4), dtype=torch.float32),
        })
        read_defaults = state_utilities.ImmutableDict({
            'reader1': torch.zeros((3, 7, 4), dtype=torch.float32),
            'reader2': torch.zeros((3, 7, 4), dtype=torch.float32),
        })
        write_defaults = state_utilities.ImmutableDict({
            'writer1': torch.zeros((3, 7, 4), dtype=torch.float32),
            'writer2': torch.zeros((3, 7, 4), dtype=torch.float32),
        })
        func = torch.jit.script(state_utilities.Storage)
        output = func(memory, read_weights, write_weights, read_defaults, write_defaults)\



class TestStateTensor(unittest.TestCase):
    """
    A test suite for the primary state_utilities tensor of the module, which
    is the piece which is actually designed to be used.
    """
    def test_getters(self):
        memory = torch.randn(2, 3, 4, 5)
        read_weights = {'read1': torch.randn(2, 3, 7, 4), 'read2': torch.randn(2, 3, 8, 4)}
        write_weights = {'write1': torch.randn(2, 3, 7, 4), 'write2': torch.randn(2, 3, 8, 4)}
        read_defaults = {'read1': torch.randn(7, 4), 'read2': torch.randn(8, 4)}
        write_defaults = {'write1': torch.randn(7, 4), 'write2': torch.randn(8, 4)}

        state_tensor = state_utilities.StateTensor(memory, read_weights, write_weights, read_defaults, write_defaults)

        # test memory getter
        self.assertTrue(torch.allclose(state_tensor.memory, memory))

        # test read weights getter
        self.assertEqual(len(state_tensor.read_weights), len(read_weights))
        for key, value in read_weights.items():
            self.assertTrue(torch.allclose(state_tensor.read_weights[key], value))

        # test write weights getter
        self.assertEqual(len(state_tensor.write_weights), len(write_weights))
        for key, value in write_weights.items():
            self.assertTrue(torch.allclose(state_tensor.write_weights[key], value))

        # test read defaults getter
        self.assertEqual(len(state_tensor.read_defaults), len(read_defaults))
        for key, value in read_defaults.items():
            self.assertTrue(torch.allclose(state_tensor.read_defaults[key], value))

        # test write defaults getter
        self.assertEqual(len(state_tensor.write_defaults), len(write_defaults))
        for key, value in write_defaults.items():
            self.assertTrue(torch.allclose(state_tensor.write_defaults[key], value))

        # test immutable properties
        with self.assertRaises(AttributeError):
            state_tensor.memory = torch.randn(2, 3, 4, 5)
        with self.assertRaises(AttributeError):
            state_tensor.read_weights = {}
        with self.assertRaises(AttributeError):
            state_tensor.write_weights = {}
        with self.assertRaises(AttributeError):
            state_tensor.read_defaults = {}
        with self.assertRaises(AttributeError):
            state_tensor.write_defaults = {}

    def test_set_weight(self):
        # initialize StateTensor
        memory = torch.randn(2, 3, 4, 5)
        read_weights = {'read1': torch.randn(2, 3, 7, 4), 'read2': torch.randn(2, 3, 8, 4)}
        write_weights = {'write1': torch.randn(2, 3, 7, 4), 'write2': torch.randn(2, 3, 8, 4)}
        read_defaults = {'read1': torch.randn(7, 4), 'read2': torch.randn(8, 4)}
        write_defaults = {'write1': torch.randn(7, 4), 'write2': torch.randn(8, 4)}
        state_tensor = state_utilities.StateTensor(memory, read_weights, write_weights, read_defaults, write_defaults)

        # test setting a read weight
        new_read_weight = torch.randn(2, 3, 7, 4)
        new_state_tensor = state_tensor.set_weight('read1', new_read_weight)
        self.assertTrue(torch.allclose(new_state_tensor.read_weights['read1'], new_read_weight))

        # test setting a write weight
        new_write_weight = torch.randn(2, 3, 8, 4)
        new_state_tensor = state_tensor.set_weight('write2', new_write_weight)
        self.assertTrue(torch.allclose(new_state_tensor.write_weights['write2'], new_write_weight))

        # test non-similar shapes for read weight raises error
        wrong_shape_read_weight = torch.randn(2, 3, 9, 4)
        with self.assertRaises(AssertionError):
            state_tensor.set_weight('read1', wrong_shape_read_weight)

        # test non-similar shapes for write weight raises error
        wrong_shape_write_weight = torch.randn(2, 3, 8, 3)
        with self.assertRaises(AssertionError):
            state_tensor.set_weight('write2', wrong_shape_write_weight)

    def test_set_default(self):
        # initialize StateTensor
        memory = torch.randn(2, 3, 4, 5)
        read_weights = {'read1': torch.randn(2, 3, 7, 4), 'read2': torch.randn(2, 3, 8, 4)}
        write_weights = {'write1': torch.randn(2, 3, 7, 4), 'write2': torch.randn(2, 3, 8, 4)}
        read_defaults = {'read1': torch.randn(7, 4), 'read2': torch.randn(8, 4)}
        write_defaults = {'write1': torch.randn(7, 4), 'write2': torch.randn(8, 4)}
        state_tensor = state_utilities.StateTensor(memory, read_weights, write_weights, read_defaults, write_defaults)

        # test setting a read default
        new_read_default = torch.randn(7, 4)
        new_state_tensor = state_tensor.set_default('read1', new_read_default)
        self.assertTrue(torch.allclose(new_state_tensor.read_defaults['read1'], new_read_default))

        # test setting a write default
        new_write_default = torch.randn(8, 4)
        new_state_tensor = state_tensor.set_default('write2', new_write_default)
        self.assertTrue(torch.allclose(new_state_tensor.write_defaults['write2'], new_write_default))

        # test non-similar shapes for read default raises error
        wrong_shape_read_default = torch.randn(9, 4)
        with self.assertRaises(AssertionError):
            state_tensor.set_default('read1', wrong_shape_read_default)

        # test non-similar shapes for write default raises error
        wrong_shape_write_default = torch.randn(8, 3)
        with self.assertRaises(AssertionError):
            state_tensor.set_default('write2', wrong_shape_write_default)

    def test_set_all_defaults(self):
        # initialize StateTensor
        memory = torch.randn(2, 3, 4, 5)
        read_weights = {'read1': torch.randn(2, 3, 7, 4), 'read2': torch.randn(2, 3, 8, 4)}
        write_weights = {'write1': torch.randn(2, 3, 7, 4), 'write2': torch.randn(2, 3, 8, 4)}
        read_defaults = {'read1': torch.randn(7, 4), 'read2': torch.randn(8, 4)}
        write_defaults = {'write1': torch.randn(7, 4), 'write2': torch.randn(8, 4)}
        state_tensor = state_utilities.StateTensor(memory, read_weights, write_weights, read_defaults, write_defaults)

        # test setting all read and write defaults
        new_read_defaults = {'read1': torch.zeros(7, 4), 'read2': torch.zeros(8, 4)}
        new_write_defaults = {'write1': torch.zeros(7, 4), 'write2': torch.zeros(8, 4)}
        new_state_tensor = state_tensor.set_all_defaults(new_read_defaults, new_write_defaults)

        # check if new defaults were set properly
        for key in new_read_defaults.keys():
            self.assertTrue(torch.allclose(new_state_tensor.read_defaults[key], new_read_defaults[key]))
        for key in new_write_defaults.keys():
            self.assertTrue(torch.allclose(new_state_tensor.write_defaults[key], new_write_defaults[key]))

        # test setting read defaults with wrong dtype raises error
        wrong_dtype_read_defaults = {'read1': torch.randn(7, 4).int(), 'read2': torch.randn(8, 4).float()}
        with self.assertRaises(AssertionError):
            state_tensor.set_all_defaults(wrong_dtype_read_defaults, new_write_defaults)

        # test setting read defaults with wrong shape raises error
        wrong_shape_read_defaults = {'read1': torch.randn(7, 5), 'read2': torch.randn(8, 5)}
        with self.assertRaises(AssertionError):
            state_tensor.set_all_defaults(wrong_shape_read_defaults, new_write_defaults)

        # test setting write defaults with wrong dtype raises error
        wrong_dtype_write_defaults = {'write1': torch.randn(7, 4).int(), 'write2': torch.randn(8, 4).float()}
        with self.assertRaises(AssertionError):
            state_tensor.set_all_defaults(new_read_defaults, wrong_dtype_write_defaults)

        # test setting write defaults with wrong shape raises error
        wrong_shape_write_defaults = {'write1': torch.randn(7, 5), 'write2': torch.randn(8, 5)}
        with self.assertRaises(AssertionError):
            state_tensor.set_all_defaults(read_defaults, wrong_shape_write_defaults)
    def test_to(self):
        memory = torch.randn(2, 3, 4, 5)
        read_weights = {'read1': torch.randn(2, 3, 7, 4), 'read2': torch.randn(2, 3, 8, 4)}
        write_weights = {'write1': torch.randn(2, 3, 7, 4), 'write2': torch.randn(2, 3, 8, 4)}
        read_defaults = {'read1': torch.randn(7, 4), 'read2': torch.randn(8, 4)}
        write_defaults = {'write1': torch.randn(7, 4), 'write2': torch.randn(8, 4)}
        state_tensor = state_utilities.StateTensor(memory, read_weights, write_weights, read_defaults, write_defaults)

        state_tensor.to(dtype = torch.float64)
    def test_jit_compiles(self):
        memory = torch.randn(2, 3, 4, 5)
        read_weights = {'read1': torch.randn(2, 3, 7, 4), 'read2': torch.randn(2, 3, 8, 4)}
        write_weights = {'write1': torch.randn(2, 3, 7, 4), 'write2': torch.randn(2, 3, 8, 4)}
        read_defaults = {'read1': torch.randn(7, 4), 'read2': torch.randn(8, 4)}
        write_defaults = {'write1': torch.randn(7, 4), 'write2': torch.randn(8, 4)}
        func = torch.jit.script(state_utilities.StateTensor)
        state_tensor = func(memory, read_weights, write_weights, read_defaults, write_defaults)
        state_tensor + 1

class TestArithmeticValidation(unittest.TestCase):
    """
    Test the arithmetic validation subroutines
    work properly.
    """

    def setUp(self):
        self.memory = torch.randn(2, 3, 4, 5)
        self.read_weights = {'read1': torch.randn(2, 3, 7, 4), 'read2': torch.randn(2, 3, 8, 4)}
        self.write_weights = {'write1': torch.randn(2, 3, 7, 4), 'write2': torch.randn(2, 3, 8, 4)}
        self.read_defaults = {'read1': torch.randn(7, 4), 'read2': torch.randn(8, 4)}
        self.write_defaults = {'write1': torch.randn(7, 4), 'write2': torch.randn(8, 4)}
        self.test_state = state_utilities.StateTensor(self.memory, self.read_weights, self.write_weights, self.read_defaults, self.write_defaults)

    @unittest.skipUnless(torch.cuda.is_available(), "gpu was not available")
    def test_different_devices(self):
        # Tensor device testing
        with self.assertRaises(ValueError) as err:
            tensor = torch.randn(2, 3).to(device=torch.device('cuda'))
            self.test_state._validate_arithmetic_operand(tensor)
        if PRINT_ERRORS_MESSAGES:
            print(err.exception)

        # StateTensor device Testing
        with self.assertRaises(ValueError) as err:
            derivative_state = self.test_state.to(device=torch.device('cuda'))
            self.test_state._validate_arithmetic_operand(derivative_state)
        if PRINT_ERRORS_MESSAGES:
            print(err.exception)

    def test_different_dtypes(self):
        # Tensor dtype testing. Control then test
        tensor = torch.randn(2, 3)
        self.test_state._validate_arithmetic_operand(tensor)
        with self.assertRaises(ValueError) as err:
            tensor = tensor.to(dtype=torch.float64)
            self.test_state._validate_arithmetic_operand(tensor)
        if PRINT_ERRORS_MESSAGES:
            print(err.exception)

        # StateTensor dtype Testing. control then test
        self.test_state._validate_arithmetic_operand(self.test_state)
        with self.assertRaises(ValueError) as err:
            derivative_state = self.test_state.to(dtype=torch.float64)
            self.test_state._validate_arithmetic_operand(derivative_state)
        if PRINT_ERRORS_MESSAGES:
            print(err.exception)

    def test_reverse_broadcast_validation(self):
        # Control case. This should not fail unless something else is wrong
        tensor = torch.randn(2, 3)
        tensor_broadcast = torch.randn(1, 3)
        self.test_state._validate_arithmetic_operand(tensor)
        self.test_state._validate_arithmetic_operand(tensor_broadcast)

        # Reverse broadcast length failure case
        tensor = torch.randn(2, 3, 4, 5)
        with self.assertRaises(ValueError) as err:
            self.test_state._validate_arithmetic_operand(tensor)

        if PRINT_ERRORS_MESSAGES:
            print(err.exception)
        # Reverse broadcast test case
        tensor = torch.randn(3, 1)
        with self.assertRaises(ValueError) as err:
            self.test_state._validate_arithmetic_operand(tensor)
        if PRINT_ERRORS_MESSAGES:
            print(err.exception)

    def test_key_mismatch_detected(self):

        # Control case
        self.test_state._validate_arithmetic_operand(self.test_state)

        # Test malformed read key throws
        with self.assertRaises(ValueError) as err:
            malformed_keys = state_utilities.StateTensor(self.test_state.memory,
                                                         self.test_state.read_weights.remove("read1"),
                                                         self.test_state.write_weights,
                                                         self.test_state.read_defaults.remove("read1"),
                                                         self.test_state.write_defaults)
            self.test_state._validate_arithmetic_operand(malformed_keys)
        if PRINT_ERRORS_MESSAGES:
            print(err.exception)

        # test malformed write keys throw
        with self.assertRaises(ValueError) as err:
            malformed_keys = state_utilities.StateTensor(self.test_state.memory,
                                                         self.test_state.read_weights,
                                                         self.test_state.write_weights.remove("write1"),
                                                         self.test_state.read_defaults,
                                                         self.test_state.write_defaults.remove("write1"))
            self.test_state._validate_arithmetic_operand(malformed_keys)
        if PRINT_ERRORS_MESSAGES:
            print(err.exception)

    def test_defaults_not_same_throws(self):
        # Control case
        self.test_state._validate_arithmetic_operand(self.test_state)

        # Test malformed read defaults throws
        with self.assertRaises(ValueError) as err:
            malformed_defaults = state_utilities.StateTensor(self.test_state.memory,
                                                             self.test_state.read_weights,
                                                             self.test_state.write_weights,
                                                             self.test_state.read_defaults.set("read1",
                                                                                                   torch.randn([7, 4])),
                                                             self.test_state.write_defaults)
            self.test_state._validate_arithmetic_operand(malformed_defaults)
        if PRINT_ERRORS_MESSAGES:
            print(err.exception)

        # Test malformed write defaults throws
        with self.assertRaises(ValueError) as err:
            malformed_defaults = state_utilities.StateTensor(self.test_state.memory,
                                                             self.test_state.read_weights,
                                                             self.test_state.write_weights,
                                                             self.test_state.read_defaults,
                                                             self.test_state.write_defaults.set("write1",
                                                                                                    torch.randn(
                                                                                                        [7, 4])))
            self.test_state._validate_arithmetic_operand(malformed_defaults)
        if PRINT_ERRORS_MESSAGES:
            print(err.exception)
    def test_shape_mismatch_detected(self):
        # Control case
        self.test_state._validate_arithmetic_operand(self.test_state)

        # Test memory shape mismatch
        with self.assertRaises(ValueError) as err:
            memory = torch.randn(2, 3, 5, 6)
            state_utilities.StateTensor(memory, self.test_state.read_weights,
                                         self.test_state.write_weights, self.test_state.read_defaults,
                                         self.test_state.write_defaults)._validate_arithmetic_operand(self.test_state)
        if PRINT_ERRORS_MESSAGES:
            print(err.exception)

        # Test read weights shape mismatch
        with self.assertRaises(ValueError) as err:
            read_weights = {'read1': torch.randn(2, 3, 6, 4), 'read2': torch.randn(2, 3, 8, 4)}
            state_utilities.StateTensor(self.test_state.memory, read_weights, self.test_state.write_weights,
                                         self.test_state.read_defaults, self.test_state.write_defaults)._validate_arithmetic_operand(self.test_state)
        if PRINT_ERRORS_MESSAGES:
            print(err.exception)

        # Test write weights shape mismatch
        with self.assertRaises(ValueError) as err:
            write_weights = {'write1': torch.randn(2, 3, 6, 4), 'write2': torch.randn(2, 3, 8, 4)}
            state_utilities.StateTensor(self.test_state.memory, self.test_state.read_weights, write_weights,
                                         self.test_state.read_defaults, self.test_state.write_defaults)._validate_arithmetic_operand(self.test_state)
        if PRINT_ERRORS_MESSAGES:
            print(err.exception)



class TestStateTensorArithmetic(unittest.TestCase):
    """ Test the arithmetic mechanisms for state tensor operate cleanly"""
    @staticmethod
    def get_expected_scalar_results(state: state_utilities.StateTensor,
                             operator: Callable
                             ) -> Tuple[
                                        torch.Tensor,
                                        state_utilities.ImmutableDict,
                                        state_utilities.ImmutableDict
                                            ]:

        # Figure out what the modified memory will be
        memory = operator(state.memory)

        # Figure out the modified read weights
        read_weights = {}
        for name, tensor in state.read_weights.items():
            read_weights[name] = operator(tensor)
        read_weights = state_utilities.ImmutableDict(read_weights)

        # Figure out the modified write weights
        write_weights = {}
        for name, tensor in state.write_weights.items():
            write_weights[name] = operator(tensor)
        write_weights = state_utilities.ImmutableDict(write_weights)
        return memory, read_weights, write_weights

    @staticmethod
    def get_expected_tensor_results(state: state_utilities.StateTensor,
                             operator: Callable
                             ) -> Tuple[
                                        torch.Tensor,
                                        state_utilities.ImmutableDict,
                                        state_utilities.ImmutableDict
                                        ]:
        # Create the reversal mechanism, which will turn a reverse broadcast into a normal
        # one from the perspective of an operator

        op_dim = operator(1.0).dim()

        def synthetic_operator(tensor: torch.Tensor)->torch.Tensor:
            # We basically move the dimensions that the
            # operator is targetting onto the
            # last dimension of the tensor,
            # add like that, then restore

            roller = torch.arange(tensor.dim())
            setup_roller = roller.roll(-op_dim)
            restoration_roller = roller.roll(op_dim)

            tensor = tensor.permute(setup_roller.tolist())
            tensor = operator(tensor)
            tensor = tensor.permute(restoration_roller.tolist())
            return tensor

        # Figure out what the modified memory will be
        memory = synthetic_operator(state.memory)

        # Figure out the modified read weights
        read_weights = {}
        for name, tensor in state.read_weights.items():
            read_weights[name] = synthetic_operator(tensor)
        read_weights = state_utilities.ImmutableDict(read_weights)

        # Figure out the modified write weights
        write_weights = {}
        for name, tensor in state.write_weights.items():
            write_weights[name] = synthetic_operator(tensor)
        write_weights = state_utilities.ImmutableDict(write_weights)
        return memory, read_weights, write_weights

    def setUp(self):
        self.memory = torch.randn(2, 3, 4, 5)
        self.read_weights = {'read1': torch.randn(2, 3, 7, 4), 'read2': torch.randn(2, 3, 8, 4)}
        self.write_weights = {'write1': torch.randn(2, 3, 7, 4), 'write2': torch.randn(2, 3, 8, 4)}
        self.read_defaults = {'read1': torch.randn(7, 4), 'read2': torch.randn(8, 4)}
        self.write_defaults = {'write1': torch.randn(7, 4), 'write2': torch.randn(8, 4)}
        self.test_state = state_utilities.StateTensor(self.memory,
                                                      self.read_weights,
                                                      self.write_weights,
                                                      self.read_defaults,
                                                      self.write_defaults)



    def test_add(self):
        # Test add scalars
        #
        # We test adding from the left and right, with
        # both ints and floats

        test_cases = []
        test_cases.append(lambda x: x + 2)
        test_cases.append(lambda x : 2 + x)
        test_cases.append(lambda x : 2.0 + x)
        test_cases.append(lambda x : x + 2.0)

        for case in test_cases:
            # Get expected values
            mem, read_weights, write_weights = self.get_expected_scalar_results(self.test_state, case)
            result = case(self.test_state)
            self.assertTrue(torch.allclose(mem, result.memory))
            self.assertTrue(read_weights == result.read_weights)
            self.assertTrue(write_weights == result.write_weights)

        # Test add tensors. These tensors must be reverse
        # broadcastable. Test from the left and the right

        test_cases = []
        test_cases.append(lambda x : x + torch.ones(2))
        test_cases.append(lambda x : torch.ones(2) + x)
        test_cases.append(lambda x : x + torch.ones(1, 3))
        test_cases.append(lambda x : torch.ones(1, 3) + x)
        test_cases.append(lambda x : x + torch.ones(2, 3))
        test_cases.append(lambda x : torch.ones(2, 3) + x)

        for case in test_cases:
            # Get expected values
            mem, read_weights, write_weights = self.get_expected_tensor_results(self.test_state, case)
            result = case(self.test_state)
            self.assertTrue(torch.allclose(mem, result.memory))
            self.assertTrue(read_weights == result.read_weights)
            self.assertTrue(write_weights == result.write_weights)

        # Test add actual state tensors together.

        self.test_state + self.test_state

    def test_subtract(self):
        # Test subtract scalars
        #
        # We test subtracting from the left and right, with
        # both ints and floats

        test_cases = []
        test_cases.append(lambda x: x - 2)
        test_cases.append(lambda x: 2 - x)
        test_cases.append(lambda x: 2.0 - x)
        test_cases.append(lambda x: x - 2.0)

        for case in test_cases:
            # Get expected values
            mem, read_weights, write_weights = self.get_expected_scalar_results(self.test_state, case)
            result = case(self.test_state)
            self.assertTrue(torch.allclose(mem, result.memory))
            self.assertTrue(read_weights == result.read_weights)
            self.assertTrue(write_weights == result.write_weights)

        # Test subtract tensors. These tensors must be reverse
        # broadcastable. Test from the left and the right

        test_cases = []
        test_cases.append(lambda x: x - torch.ones(2))
        test_cases.append(lambda x: torch.ones(2) - x)
        test_cases.append(lambda x: x - torch.ones(1, 3))
        test_cases.append(lambda x: torch.ones(1, 3) - x)
        test_cases.append(lambda x: x - torch.ones(2, 3))
        test_cases.append(lambda x: torch.ones(2, 3) - x)

        for case in test_cases:
            # Get expected values
            mem, read_weights, write_weights = self.get_expected_tensor_results(self.test_state, case)
            result = case(self.test_state)
            self.assertTrue(torch.allclose(mem, result.memory))
            self.assertTrue(read_weights == result.read_weights)
            self.assertTrue(write_weights == result.write_weights)

        # Test subtract actual state tensors from each other.
        self.test_state - self.test_state

    def test_multiply(self):
        # Test multiply scalars
        #
        # We test multiplying from the left and right, with
        # both ints and floats

        test_cases = []
        test_cases.append(lambda x: x * 2)
        test_cases.append(lambda x : 2 * x)
        test_cases.append(lambda x : 2.0 * x)
        test_cases.append(lambda x : x * 2.0)

        for case in test_cases:
            # Get expected values
            mem, read_weights, write_weights = self.get_expected_scalar_results(self.test_state, case)
            result = case(self.test_state)
            self.assertTrue(torch.allclose(mem, result.memory))
            self.assertTrue(read_weights == result.read_weights)
            self.assertTrue(write_weights == result.write_weights)

        # Test multiply tensors. These tensors must be broadcastable

        test_cases = []
        test_cases.append(lambda x : x * torch.randn(2))
        test_cases.append(lambda x : torch.randn(2) * x)
        test_cases.append(lambda x : x * torch.randn(1, 3))
        test_cases.append(lambda x : torch.randn(1, 3) * x)
        test_cases.append(lambda x : x * torch.randn(2, 3))
        test_cases.append(lambda x : torch.randn(2, 3) * x)

        for case in test_cases:
            # Get expected values
            mem, read_weights, write_weights = self.get_expected_tensor_results(self.test_state, case)
            result = case(self.test_state)
            self.assertTrue(torch.allclose(mem, result.memory))
            self.assertTrue(read_weights == result.read_weights)
            self.assertTrue(write_weights == result.write_weights)

        # Test multiply states

        self.test_state * self.test_state

    def test_multiply(self):
        # Test multiply scalars
        #
        # We test multiplying from the left and right, with
        # both ints and floats

        test_cases = []
        test_cases.append(lambda x: x * 2)
        test_cases.append(lambda x: 2 * x)
        test_cases.append(lambda x: 2.0 * x)
        test_cases.append(lambda x: x * 2.0)

        for case in test_cases:
            # Get expected values
            mem, read_weights, write_weights = self.get_expected_scalar_results(self.test_state, case)
            result = case(self.test_state)
            self.assertTrue(torch.allclose(mem, result.memory))
            self.assertTrue(read_weights == result.read_weights)
            self.assertTrue(write_weights == result.write_weights)

        # Test multiply tensors. These tensors must be broadcastable.
        # Test from the left and the right

        test_cases = []
        test_cases.append(lambda x: x * torch.ones(2))
        test_cases.append(lambda x: torch.ones(2) * x)
        test_cases.append(lambda x: x * torch.ones(1, 3))
        test_cases.append(lambda x: torch.ones(1, 3) * x)
        test_cases.append(lambda x: x * torch.ones(2, 3))
        test_cases.append(lambda x: torch.ones(2, 3) * x)

        for case in test_cases:
            # Get expected values
            mem, read_weights, write_weights = self.get_expected_tensor_results(self.test_state, case)
            result = case(self.test_state)
            self.assertTrue(torch.allclose(mem, result.memory))
            self.assertTrue(read_weights == result.read_weights)
            self.assertTrue(write_weights == result.write_weights)

        # Test multiply state tensors

        self.test_state * self.test_state

    def test_divide(self):
        # Test divide scalars
        #
        # We test dividing from the left and right, with
        # both ints and floats

        test_cases = []
        test_cases.append(lambda x: x / 2)
        test_cases.append(lambda x: 2 / x)
        test_cases.append(lambda x: 2.0 / x)
        test_cases.append(lambda x: x / 2.0)

        for case in test_cases:
            # Get expected values
            mem, read_weights, write_weights = self.get_expected_scalar_results(self.test_state, case)
            result = case(self.test_state)
            self.assertTrue(torch.allclose(mem, result.memory))
            self.assertTrue(read_weights == result.read_weights)
            self.assertTrue(write_weights == result.write_weights)

        # Test divide tensors. These tensors must be broadcastable.
        # Test from the left and the right

        test_cases = []
        test_cases.append(lambda x: x / torch.ones(2))
        test_cases.append(lambda x: torch.ones(2) / x)
        test_cases.append(lambda x: x / torch.ones(1, 3))
        test_cases.append(lambda x: torch.ones(1, 3) / x)
        test_cases.append(lambda x: x / torch.ones(2, 3))
        test_cases.append(lambda x: torch.ones(2, 3) / x)

        for case in test_cases:
            # Get expected values
            mem, read_weights, write_weights = self.get_expected_tensor_results(self.test_state, case)
            result = case(self.test_state)
            self.assertTrue(torch.allclose(mem, result.memory))
            self.assertTrue(read_weights == result.read_weights)
            self.assertTrue(write_weights == result.write_weights)

        # Test divide by each other

        self.test_state / self.test_state