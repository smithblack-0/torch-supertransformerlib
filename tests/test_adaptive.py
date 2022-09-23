import torch
import unittest
from src.supertransformerlib import Adaptive


class test_Adaptive_Attention_Basics(unittest.TestCase):
    def test_constructor(self):
        """Test the constructor functions reasonably well"""
        Adaptive.Adaptive_Attention(10, 20, 30, 5, 5, 5)
    def test_startup(self):
        """Test that the startup feature of the layer is able to make correct accumulators"""

        query = torch.zeros([10, 3, 5, 32])
        key = torch.zeros([10, 3, 5, 64])
        value = torch.zeros([10, 3, 5, 48])
        layer = Adaptive.Adaptive_Attention(32, 64, 48, 5, 5, 5)

        accumulator = layer.start_accumulator(query, key, value)
        self.assertTrue(accumulator.Halting_Probabilities.shape == query.shape[:-1])
        self.assertTrue(accumulator.Residuals.shape == query.shape[:-1])
        self.assertTrue(accumulator.Output.shape == torch.Size([*query.shape[:-1], value.shape[-1]]))

    def test_call_simple(self):
        """Test if call works in the simplest of cases"""
        #Setup

        query = torch.randn([5, 32])
        key = torch.randn([5, 32])
        value = torch.randn([5, 32])
        layer = Adaptive.Adaptive_Attention(32, 32, 32, 5, 6, 7)
        accumulator = layer.start_accumulator(query, key, value)

        #Call
        new_accumulator = layer(accumulator, query, key, value)
        self.assertTrue(torch.any(new_accumulator.Output != accumulator.Output))
    def test_call_intermediate(self):
        """Test if forward works in a mildly complicated situation"""
        query = torch.randn([10, 5, 32])
        key = torch.randn([10, 5, 48])
        value = torch.randn([10, 5, 64])
        layer = Adaptive.Adaptive_Attention(32, 48, 64, 5, 6, 7)
        accumulator = layer.start_accumulator(query, key, value)

        new_accumulator = layer(accumulator, query, key, value)
        self.assertTrue(torch.any(new_accumulator.Output != accumulator.Output))

    def test_halted(self):
        """Test that when the probability gets full, we capture residuals and halt"""
        query = torch.randn([5, 32])
        key = torch.randn([5, 48])
        value = torch.randn([5, 64])
        layer = Adaptive.Adaptive_Attention(32, 48, 64, 5, 6, 7)

        accumulator = layer.start_accumulator(query, key, value)
        accumulator = Adaptive.Adaptive_Accumulator(torch.ones_like(accumulator.Halting_Probabilities),
                                                    accumulator.Residuals,
                                                    accumulator.Output,
                                                    accumulator.Mask,
                                                    )
        new_accumulator = layer(accumulator, query, key, value)
        self.assertTrue(torch.all(new_accumulator.Output == accumulator.Output))
        self.assertTrue(torch.all(new_accumulator.Residuals == accumulator.Residuals))
    def test_collects_residuals(self):
        """Test that when the probabity is fully exhausted residuals are collected"""
        query = torch.randn([10, 5, 32])
        key = torch.randn([10, 5, 48])
        value = torch.randn([10, 5, 64])
        layer = Adaptive.Adaptive_Attention(32, 48, 64, 5, 6, 7)

        accumulator = layer.start_accumulator(query, key, value)
        accumulator = Adaptive.Adaptive_Accumulator(0.9*torch.ones_like(accumulator.Halting_Probabilities),
                                                    accumulator.Residuals,
                                                    accumulator.Output,
                                                    accumulator.Mask
                                                    )
        new_accumulator = layer(accumulator, query, key, value)


        freshly_halted = new_accumulator.Halting_Probabilities > 1 - 0.00001
        residuals = new_accumulator.Residuals
        halted_residuals = residuals.masked_select(freshly_halted)
        expected_residuals = 0.1*torch.ones_like(halted_residuals)
        self.assertTrue(freshly_halted.sum() > 0) #With these kinds of conditions, SOMETHING must have halted\
        self.assertTrue(torch.all((expected_residuals-halted_residuals).abs() < 0.001))


class test_meshmap_focus(unittest.TestCase):
    """
    Test the ability to remap tensors using
    meshes.
    """
    def test_makemesh(self):
        """Test the meshmap is creating itself properly"""
        halting_probabilities = torch.randn([3, 2, 4,5])
        mesh = Adaptive.make_meshmap(halting_probabilities)
        mesh = torch.stack(mesh, dim=-1)
        self.assertTrue(mesh.shape == torch.Size([3, 2, 4, 5, 4]))
    def test_batchmeshprune(self):
        """Test the ability to prune off dead/halted batches from the construction mesh"""
        """Test that the finished batch is excluded."""
        halted_probabilities = torch.tensor([[0.0, 0.0],[0.1, 1.0], [1.0, 1.0]])
        mesh = Adaptive.make_meshmap(halted_probabilities)
        mapping = Adaptive.mesh_batchprune(halted_probabilities, mesh)
        expected_mapping = torch.tensor([[[0, 0], [0, 1]],[[1,0],[1,1]]])
    def test_batchmeshprune_2(self):
        halted_probabilities = torch.randn([20,5,3,6,7])
        mesh = Adaptive.make_meshmap(halted_probabilities)
        mapping = Adaptive.mesh_batchprune(halted_probabilities)
class test_focus_get_batch_mesh(unittest.TestCase):
    """
    Test case for using the get batch mesh
    function of focus.
    """
    def test_manually(self):
        """Test that the finished batch is excluded."""
        halted_probabilities = torch.tensor([[0.0, 0.0],[0.1, 1.0], [1.0, 1.0]])
        mapping = Adaptive.get_batch_mesh(halted_probabilities)
        expected_mapping = torch.tensor([[[0, 0], [0, 1]],[[1,0],[1,1]]])
        self.assertTrue(torch.all(mapping == expected_mapping))
    def test_complex_shape(self):
        """Test that mapping works with a complex batch."""
        halted_probabilities = torch.randn([10, 20, 4, 2, 6])
        mapping = Adaptive.get_batch_mesh(halted_probabilities)
        halted_probabilities[mapping.unbind(-1)]

class test_focus_get_querymesh(unittest.TestCase):
    def test_manually(self):
        halted_probabilities = torch.tensor([[0.0, 0.0, 1.0],[0.1, 1.0, 1.0], [1.0, 0.2, 1.0]])
        print(halted_probabilities.data_ptr())
        mapping, mask = Adaptive.get_query_meshmask(halted_probabilities)
        outcome = halted_probabilities[mapping.unbind(-1)]
        print(outcome.data_ptr())

        outcome = outcome*mask
        expected_outcome = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        self.assertTrue(torch.all(outcome == expected_outcome))
