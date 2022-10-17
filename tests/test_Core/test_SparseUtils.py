import unittest
import torch
from src.supertransformerlib.Core import SparseUtils

class test_calculate_stride(unittest.TestCase):
    def test_calculate_strides(self):
        """tests that the calculate stride function works correctly and mirrors reality."""
        tensor = torch.randn([10, 20, 30, 40])
        expected = tensor.stride()
        expected = torch.tensor(expected)
        func = torch.jit.script(SparseUtils.calculate_shape_strides)
        output = func(tensor.shape)
        print(expected, output)

        self.assertTrue(torch.all(expected == output))

class test_gen_indices_from_mask(unittest.TestCase):
    def test_simple_manual(self):
        """Test simple gen case works"""
        mask = torch.tensor([False, True, True, False])
        expected = torch.tensor([1, 2]).unsqueeze(0)

        output = SparseUtils.gen_indices_from_mask(mask)
        self.assertTrue(torch.all(expected == output))
    def test_multidim_manual(self):
        """Test a multidimensional gen works"""
        mask = torch.tensor([[False, True],[True, False]])
        expected = torch.tensor([[0, 1],[1,0]])
        output = SparseUtils.gen_indices_from_mask(mask)
        self.assertTrue(torch.all(output==expected))
    def test_makes_correct_sparse(self):
        """Test that we can make a correct sparse tensor this way, and
        it behaves like the dense instance where appropriate."""
        tensor = torch.randn([10, 20, 30])
        mask = torch.randn([10, 20, 30]) > 0

        indices = SparseUtils.gen_indices_from_mask(mask)
        values = tensor[mask]
        sparse = torch.sparse_coo_tensor(indices, values)
        dense = sparse.to_dense()

        self.assertTrue(torch.all(dense == tensor*mask))


