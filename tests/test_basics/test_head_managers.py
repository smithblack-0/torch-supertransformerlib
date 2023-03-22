import unittest
import torch
from supertransformerlib.Basics import MakeHead, MergeHeads, ReductiveMergeHeads
from supertransformerlib import Core

class TestMakeHeadReshape(unittest.TestCase):
    """ Tests the make head by reshape varient of make heads"""

    def test_simple_batch(self):
        d_model = 8
        heads = 2
        items = 10
        batch = torch.rand(5, items, d_model)
        make_head = MakeHead(d_model, heads, mode="reshape")
        result = make_head(batch)
        self.assertEqual(result.shape, (5, heads, items, d_model // heads))

    def test_complex_batch(self):
        d_model = 8
        heads = 2
        items = 10
        batch = torch.rand(5, 3, 4, items, d_model)
        make_head = MakeHead(d_model, heads, mode="reshape")
        result = make_head(batch)
        self.assertEqual(result.shape, (5, 3, 4, heads, items, d_model // heads))

    def test_torchscript_compiles(self):
        d_model = 8
        heads = 2
        items = 10
        batch = torch.rand(5, 3, 4, items, d_model)
        make_head = MakeHead(d_model, heads, mode="reshape")
        make_head = torch.jit.script(make_head)
        result = make_head(batch)
        self.assertEqual(result.shape, (5, 3, 4, heads, items, d_model // heads))
    def test_error_heads_less_than_one(self):
        d_model = 8
        heads = 10
        with self.assertRaises(Core.ValidationError):
            MakeHead(d_model, heads, mode="reshape")

    def test_error_not_divisible(self):
        d_model = 10
        heads = 3
        with self.assertRaises(Core.ValidationError) as err:
            MakeHead(d_model, heads, mode="reshape")


class TestMakeHeadProject(unittest.TestCase):
    """ Test make head operating in the projection mode."""
    def test_simple_batch(self):
        """ Test a simple batch works properly"""
        d_model = 8
        heads = 2
        items = 10
        batch = torch.rand(5, items, d_model)
        make_head = MakeHead(d_model, heads, mode="project")
        result = make_head(batch)
        self.assertEqual(result.shape, (5, heads, items , d_model // heads))

    def test_complex_batch(self):
        """ Test a complex batch works properly"""
        d_model = 8
        heads = 2
        items = 10
        batch = torch.rand(5, 3, 4, items, d_model)
        make_head = MakeHead(d_model, heads, mode="project")
        result = make_head(batch)
        self.assertEqual(result.shape, (5, 3, 4, heads, items, d_model // heads))

    def test_ensemble(self):
        """ Test ensembles are handled correctly"""
        d_model = 8
        heads = 2
        parallel = 10
        items = 12
        batch = torch.rand(5, 3, parallel, items, d_model)
        make_head = MakeHead(d_model, heads, mode="project", parallel=parallel)
        result = make_head(batch)
        self.assertEqual(result.shape, (5, 3, parallel,  heads, items, d_model // heads))

class TestMergeHeads(unittest.TestCase):
    """ Test the merge head class"""
    def test_reshape_simple_batch(self):
        d_model = 8
        heads = 2
        tensor = torch.randn(3, 4, heads, d_model//heads)
        merge_heads = MergeHeads(d_model, heads, mode="reshape")
        output = merge_heads(tensor)
        expected_output_shape = (3, 4, d_model)
        self.assertEqual(output.shape, expected_output_shape)

    def test_reshape_complex_batch(self):
        d_model = 8
        heads = 2
        tensor = torch.randn(3, 4, 5, 6, heads, d_model//heads)
        merge_heads = MergeHeads(d_model, heads, mode="reshape")
        output = merge_heads(tensor)
        expected_output_shape = (3, 4, 5, 6, d_model)
        self.assertEqual(output.shape, expected_output_shape)

    def test_reshape_head_less_than_one(self):
        d_model = 8
        heads = 10
        with self.assertRaises(Core.ValidationError):
            MergeHeads(d_model, heads, mode="reshape")

    def test_reshape_incompatible_d_model_and_heads(self):
        d_model = 8
        heads = 3
        with self.assertRaises(Core.ValidationError):
            MergeHeads(d_model, heads, mode="reshape")

    def test_linear_simple_batch(self):
        d_model = 8
        heads = 2
        tensor = torch.randn(3, 4, heads, d_model // heads)
        merge_heads = MergeHeads(d_model, heads, mode="linear")
        output = merge_heads(tensor)
        expected_output_shape = (3, 4, d_model)
        self.assertEqual(output.shape, expected_output_shape)

    def test_linear_complex_batch(self):
        d_model = 8
        heads = 2
        tensor = torch.randn(3, 4, 5, heads, d_model // heads)
        merge_heads = MergeHeads(d_model, heads, mode="linear")
        output = merge_heads(tensor)
        expected_output_shape = (3, 4, 5, d_model)
        self.assertEqual(output.shape, expected_output_shape)

    def test_invalid_mode(self):
        d_model = 8
        heads = 2
        with self.assertRaises(ValueError):
            MergeHeads(d_model, heads, mode="invalid_mode")

    def test_merge_heads_ensemble(self):
        d_model = 12
        heads = 3
        batch_size = 2
        ensemble = 4
        input_tensor = torch.randn(batch_size, ensemble, 2, heads, d_model // heads)
        merge_heads = MergeHeads(d_model, heads, mode="linear", parallel=ensemble)

        output_tensor = merge_heads(input_tensor)
        expected_shape = (batch_size, ensemble, 2, d_model)
        self.assertEqual(output_tensor.shape, expected_shape)


class TestReductiveMergeHeads(unittest.TestCase):
    """ Test the various merge head cases"""
    def test_sum(self):
        d_head = 4
        heads = 3
        batch_size = 2
        items = 5
        input_tensor = torch.randn(batch_size, heads, items, d_head)
        merge_heads = ReductiveMergeHeads(d_head, heads, mode="sum")

        output_tensor = merge_heads(input_tensor)
        expected_shape = (batch_size, items, d_head)

        self.assertEqual(output_tensor.shape, expected_shape)

    def test_weighted_sum(self):
        d_head = 4
        heads = 3
        batch_size = 2
        items = 5
        input_tensor = torch.randn(batch_size, heads, items, d_head)
        merge_heads = ReductiveMergeHeads(d_head, heads, mode="weighted_sum")

        output_tensor = merge_heads(input_tensor)
        expected_shape = (batch_size, items, d_head)

        self.assertEqual(output_tensor.shape, expected_shape)

    def test_project(self):
        d_head = 4
        heads = 3
        batch_size = 2
        items = 5
        input_tensor = torch.randn(batch_size, heads, items, d_head)
        merge_heads = ReductiveMergeHeads(d_head, heads, mode="project")

        output_tensor = merge_heads(input_tensor)
        expected_shape = (batch_size, items, d_head)

        self.assertEqual(output_tensor.shape, expected_shape)

    def test_sum_ensemble(self):
        d_head = 4
        heads = 3
        batch_size = 2
        items = 5
        ensemble = 4
        input_tensor = torch.randn(batch_size, ensemble, heads, items, d_head)
        merge_heads = ReductiveMergeHeads(d_head, heads, mode="sum", parallel=ensemble)

        output_tensor = merge_heads(input_tensor)
        expected_shape = (batch_size, ensemble, items, d_head)

        self.assertEqual(output_tensor.shape, expected_shape)

    def test_weighted_sum_ensemble(self):
        d_head = 4
        heads = 3
        batch_size = 2
        items = 5
        ensemble = 4
        input_tensor = torch.randn(batch_size, ensemble, heads, items, d_head)
        merge_heads = ReductiveMergeHeads(d_head, heads, mode="weighted_sum", parallel=ensemble)

        output_tensor = merge_heads(input_tensor)
        expected_shape = (batch_size, ensemble, items, d_head)

        self.assertEqual(output_tensor.shape, expected_shape)

    def test_project_ensemble(self):
        d_head = 4
        heads = 3
        batch_size = 2
        items = 5
        ensemble = 4
        input_tensor = torch.randn(batch_size, ensemble, heads, items, d_head)
        merge_heads = ReductiveMergeHeads(d_head, heads, mode="project", parallel=ensemble)

        output_tensor = merge_heads(input_tensor)
        expected_shape = (batch_size, ensemble, items, d_head)

        self.assertEqual(output_tensor.shape, expected_shape)