import torch
import unittest
import itertools
from supertransformerlib.Core import bundle_tensor

PRINT_ERROR_MESSAGES = True

class TestImmutableDictBehavior(unittest.TestCase):
    """
    Tests the tensor behaves like an immutable dictionary
    when setup, with tensors locked together
    """

    def test_init(self):

        data = {'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])}
        imm_dict = bundle_tensor.BundleTensor(0, data)
        self.assertEqual(imm_dict['a'].tolist(), [1, 2, 3])
        self.assertEqual(imm_dict['b'].tolist(), [4, 5, 6])
        self.assertEqual(len(imm_dict), 2)
        self.assertEqual(str(imm_dict), "{'a': tensor([1, 2, 3]), 'b': tensor([4, 5, 6])}")
        self.assertEqual(repr(imm_dict), "{'a': tensor([1, 2, 3]), 'b': tensor([4, 5, 6])}")

    def test_immutable(self):
        data = {'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])}
        imm_dict = bundle_tensor.BundleTensor(0, data)
        with self.assertRaises(TypeError):
            imm_dict['a'] = torch.tensor([7, 8, 9])
        with self.assertRaises(TypeError):
            imm_dict['c'] = torch.tensor([10, 11, 12])

    def test_set(self):
        data = {'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])}
        constraint = {"a" : ["embedding"], "b" : ["embedding"]}
        imm_dict = bundle_tensor.BundleTensor(0, data, constraint)


        new_dict = imm_dict.set('a', torch.tensor([7, 8, 9]), dim_names=["embedding"])
        self.assertEqual(new_dict['a'].tolist(), [7, 8, 9])
        self.assertEqual(new_dict['b'].tolist(), [4, 5, 6])
        self.assertEqual(imm_dict['a'].tolist(), [1, 2, 3])
        self.assertEqual(len(new_dict), 2)
        self.assertEqual(len(imm_dict), 2)

    def test_hash(self):
        data = {'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])}
        imm_dict = bundle_tensor.BundleTensor(0, data)

        hash1 = hash(imm_dict)
        data2 = {'a': torch.tensor([7, 8, 9]), 'b': torch.tensor([4, 5, 6])}
        imm_dict2 = bundle_tensor.BundleTensor(0, data2)
        hash2 = hash(imm_dict2)

        data3 = {key : value.clone() for key, value in data.items()}
        imm_dict3 = bundle_tensor.BundleTensor(0, data3)
        hash3 = hash(imm_dict3)
        self.assertNotEqual(hash1, hash2)
        self.assertEqual(hash1, hash3)

    def test_items(self):
        # create test data
        data = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6]), "c": torch.tensor([7, 8, 9])}
        # create ImmutableDict instance
        immutable_dict = bundle_tensor.BundleTensor(0,
                                                   data)
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
        immutable_dict = bundle_tensor.BundleTensor(0, data)
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
        immutable_dict = bundle_tensor.BundleTensor(0, data)
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
        immutable_dict1 = bundle_tensor.BundleTensor(0, data1)
        immutable_dict2 = bundle_tensor.BundleTensor(0, data2)
        # update the first dictionary with the second
        updated_dict = immutable_dict1.update(immutable_dict2)
        # check that the updated dictionary contains all key-value pairs from both original dictionaries
        for key, value in data1.items():
            assert key in updated_dict
            assert torch.equal(updated_dict[key], value)
        for key, value in data2.items():
            assert key in updated_dict
            assert torch.equal(updated_dict[key], value)


class TestBatchStateTensorConstructor(unittest.TestCase):

    def test_successful_creation(self):
        tensors = {
            "a": torch.randn(5, 3),
            "b": torch.randn(5, 2),
        }
        bst = bundle_tensor.BundleTensor(1, tensors)
        self.assertEqual(bst.batch_dim, 1)
        self.assertEqual(bst.tensors, tensors)
        self.assertEqual(bst.batch_shape, (5,))

    def test_no_tensors(self):
        tensors = {}
        with self.assertRaises(bundle_tensor.ConstructorNoTensorsError) as err:
            bst = bundle_tensor.BundleTensor(1, tensors)

        if PRINT_ERROR_MESSAGES:
            print(err.exception)

    def test_different_dtype_tensors(self):
        tensors = {
            "a": torch.randn(5, 3),
            "b": torch.randn(5, 2).to(torch.int32),
        }
        with self.assertRaises(bundle_tensor.ConstructorBadTensorDtypeError) as err:
            bst = bundle_tensor.BundleTensor(1, tensors)

        if PRINT_ERROR_MESSAGES:
            print(err.exception)

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda available for device test")
    def test_different_device_tensors(self):
        tensors = {
            "a": torch.randn(5, 3),
            "b": torch.randn(5, 2).to("cuda"),
        }
        with self.assertRaises(bundle_tensor.ConstructorBadTensorDeviceError) as err:
            bst = bundle_tensor.BundleTensor(1, tensors)

        if PRINT_ERROR_MESSAGES:
            print(err.exception)

    def test_non_positive_batch_dim(self):
        tensors = {
            "a": torch.randn(5, 3),
            "b": torch.randn(5, 2),
        }
        with self.assertRaises(AssertionError) as err:
            bst = bundle_tensor.BundleTensor(-1, tensors)

        if PRINT_ERROR_MESSAGES:
            print(err.exception)


    def test_incorrect_batch_dim(self):
        tensors = {
            "a": torch.randn(5, 3),
            "b": torch.randn(5, 2),
        }
        with self.assertRaises(bundle_tensor.ConstructorBadTensorBatchShape) as err:
            bst = bundle_tensor.BundleTensor(2, tensors)

        if PRINT_ERROR_MESSAGES:
            print(err.exception)

    def test_invalid_constraints_spec_violation(self):
        tensors = {
            "a": torch.randn(5, 4, 2, 3),
            "b": torch.randn(5, 4, 2, 7),
        }
        invalid_constraints_spec = {
            "a": ["items", "embeddings"],
            "b": ["items", "embeddings"],
        }

        with self.assertRaises(bundle_tensor.ConstructorTensorsViolateDimensionalConstraint) as err:
            bst = bundle_tensor.BundleTensor(1, tensors, invalid_constraints_spec)

        if PRINT_ERROR_MESSAGES:
            print(err.exception)

class TestPerformBioperandArithmetic(unittest.TestCase):
    """
    This section tests the underlying arithmatic function which
    is used to impliment the magic methods.

    Tests are performed by creating the various test i
    """
    def setUp(self):
        # In

        # Construct the root instance. This is what the
        # function will be called from

        data_tensors = {
            "a": torch.rand(5, 3),
            "b": torch.rand(5, 2),
        }
        root_bundle_tensor = bundle_tensor.BundleTensor(1, data_tensors)

        # Construct the various numeric entities which we can interact with peacefully. This
        # consists of floats, ints, tensors, and state tensors. Store them in a dictionary

        operand_examples_dict = {}
        operand_examples_dict["float"] = 2.0
        operand_examples_dict["int"] = 4
        operand_examples_dict["batch_tensor"] = torch.rand([5])
        operand_examples_dict["same_shape_state"] = root_bundle_tensor

        batch_tensors = {"a" : torch.rand(5),
                         "b" : torch.rand(5)}
        operand_examples_dict["broadcast_bundle_tensor"] = bundle_tensor.BundleTensor(1, batch_tensors)

        # Construct the various operator keywords which need to
        # be passed into the class

        operators_dict = {}
        operators_dict["add"] = "add"
        operators_dict["subtract"] = "subtract"
        operators_dict["multiply"] = "multiply"
        operators_dict["divide"] = "divide"
        operators_dict["power"] = "power"

        # Construct the functional definitions of the operators before,
        # so we can do easy validation.

        operators_functions = {}
        operators_functions["add"] = lambda x, y : x + y
        operators_functions["subtract"] = lambda x, y : x - y
        operators_functions["multiply"] = lambda x, y : x * y
        operators_functions["divide"] = lambda x, y : x / y
        operators_functions["power"] = lambda x, y : x ** y

        # Store the cases

        self.data_tensors = data_tensors
        self.root_state = root_bundle_tensor
        self.operand_cases = operand_examples_dict
        self.operator_cases = operators_dict
        self.operator_functions = operators_functions


    def test_permutation_options(self):
        # Test the various permutations of operand and operator. Make sure
        # they all work together. Product, from itertools, produces every combination

        for operand_name, operator_name in itertools.product(self.operand_cases, self.operator_cases):
            operand = self.operand_cases[operand_name]
            operator = self.operator_cases[operator_name]

            try:
                # Test that the operator actually is functioning when called
                result1 = self.root_state.perform_bioperand_arithmetic(self.root_state, operator, operand)
                result2 = self.root_state.perform_bioperand_arithmetic(operand, operator, self.root_state)

                # Test the tensors are being properly updated. We do this
                # by manually iterating over the operand and operator, and
                # calculating what the results SHOULD be.

                operator = self.operator_functions[operator_name]

                for key in self.data_tensors.keys():
                    if isinstance(operand, bundle_tensor.BundleTensor):
                        # It was a state tensor. Fetch, expand the key
                        suboperand = operand[key]
                        while suboperand.dim() < self.root_state[key].dim():
                            suboperand = suboperand.unsqueeze(-1)
                    elif isinstance(operand, torch.Tensor):
                        suboperand = operand
                        while suboperand.dim() < self.root_state[key].dim():
                            suboperand = suboperand.unsqueeze(-1)

                    else:
                        suboperand = operand

                    expected1 = operator(self.root_state[key], suboperand)
                    expected2 = operator(suboperand, self.root_state[key])

                    self.assertTrue(torch.allclose(expected1, result1[key]))
                    self.assertTrue(torch.allclose(expected2, result2[key]))

            except Exception as err:
                msg = f"Issue when handling operand {operand_name} and operator {operator_name}/n"
                msg = msg + str(err)
                raise Exception(msg)
