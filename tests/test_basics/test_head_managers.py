import unittest
import torch
from src.supertransformerlib.Basics import MakeHead
from src.supertransformerlib import Core

class TestMakeHeadReshape(unittest.TestCase):
    """ Tests the make head by reshape varient of make heads"""

    def test_simple_batch(self):
        d_model = 8
        heads = 2
        batch = torch.rand(5, 10, d_model)
        make_head = MakeHead(d_model, heads, mode="reshape")
        result = make_head(batch)
        self.assertEqual(result.shape, (5, 10, heads, d_model // heads))

    def test_complex_batch(self):
        d_model = 8
        heads = 2
        batch = torch.rand(5, 3, 4, 10, d_model)
        make_head = MakeHead(d_model, heads, mode="reshape")
        result = make_head(batch)
        self.assertEqual(result.shape, (5, 3, 4, 10, heads, d_model // heads))

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
        batch = torch.rand(5, 10, d_model)
        make_head = MakeHead(d_model, heads, mode="project")
        result = make_head(batch)
        self.assertEqual(result.shape, (5, 10, heads, d_model // heads))

    def test_complex_batch(self):
        """ Test a complex batch works properly"""
        d_model = 8
        heads = 2
        batch = torch.rand(5, 3, 4, 10, d_model)
        make_head = MakeHead(d_model, heads, mode="project")
        result = make_head(batch)
        self.assertEqual(result.shape, (5, 3, 4, 10, heads, d_model // heads))

    def test_ensemble(self):
        """ Test ensembles are handled correctly"""
        d_model = 8
        heads = 2
        parallel = 10
        batch = torch.rand(5, 3, parallel, 12, d_model)
        make_head = MakeHead(d_model, heads, mode="project", parallel=parallel)
        result = make_head(batch)
        self.assertEqual(result.shape, (5, 3, parallel, 12, heads, d_model // heads))