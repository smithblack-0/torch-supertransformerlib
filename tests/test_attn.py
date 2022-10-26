import unittest
import torch

from src.supertransformerlib import Attention
from src.supertransformerlib import Core

def shape_equal(shape1, shape2):
    shape1 = torch.tensor(shape1)
    shape2 = torch.tensor(shape2)
    return torch.all(shape1 == shape2)


class test_dotproductattention(unittest.TestCase):
    """
    Tests the dot product attention function loose in the
    function module.
    """
    def test_basic_dotproductattn(self):
        """Tests if a random, no frills case works safely."""
        query = torch.randn([3, 5, 10, 20])
        content = torch.randn([3, 5, 30, 20])
        attn = Attention._dot_product_attention(query, content, content)

    def test_masked_dotproductattn(self):
        """ tests if a masked case functions correctly """
        query = torch.randn([3, 5, 10, 20])
        content = torch.randn([3, 5, 30, 20])
        mask = torch.randn([10, 30]) > 0.5
        attn = Attention._dot_product_attention(query, content, content, mask)

    def test_torchscript_compile(self):
        test = torch.jit.script(Attention._dot_product_attention)

    @unittest.skipUnless(torch.cuda.is_available(), "gpu test requires valid gpu install")
    def test_gpu(self):
        """ tests if the gpu runs okay """
        device = torch.device("cuda")
        query = torch.randn([3, 5, 10, 20]).to(device)
        content = torch.randn([3, 5, 30, 20]).to(device)
        mask = (torch.randn([10, 30]) > 0.5).to(device)
        attn = Attention._dot_product_attention(query, content, content, mask)


class test_Feedforward(unittest.TestCase):
    def test_straightforward(self):
        """Test feedforward works without any tricks"""
        test_tensor = torch.randn([10, 20, 4, 16])
        instance = Attention.FeedForward(16)
        instance = torch.jit.script(instance)
        self.assertTrue(instance(test_tensor).shape == torch.Size([10, 20, 4, 16]))
        self.assertTrue(torch.any(instance(test_tensor) != test_tensor))

    def test_parallel(self):
        """ Test the parallel processing system is engaging"""
        test_tensor = torch.randn([10, 20, 4, 16])
        instance = Attention.FeedForward(16, parallelization=[10, 20])
        instance = torch.jit.script(instance)
        output = instance(test_tensor)
        self.assertTrue(output.shape == torch.Size([10, 20, 4, 16]))
        self.assertTrue(torch.any(output != test_tensor))



class test_MultiHeadedAttention(unittest.TestCase):
    """
    Tests whether multiheaded attention is operating correctly.

    Tests include:

    -- does it work at all
    -- does it work when reshaping embeddings
    -- does it work when masking
    -- do ensembles work
    -- are ensembles processed independently
    -- does torchscript compile, so I can get an IR or ONNX if I need it?

    """
    def test_basic(self):
        """ test if no frills attention works"""
        query = torch.randn([3, 5, 10, 32])
        content = torch.randn([3, 5, 30, 32])
        layer = Attention.MultiHeadedAttention(32, 32, 32, 4)
        attn = layer(query, content, content)
        self.assertTrue(shape_equal([3, 5, 10, 32], attn.shape))

    def test_uneven_embeddings(self):
        """ tests if attention works when the embeddings are changing dynamic_shape """
        query = torch.randn([3,5,10, 64])
        content = torch.randn([3, 5, 30, 16])
        layer = Attention.MultiHeadedAttention(64, 16, 32, 4)
        attn = layer(query, content, content)
        self.assertTrue(shape_equal([3, 5, 10, 32], attn.shape))

    def test_masked_embeddings(self):
        """ tests if attention works when a mask is involved """
        query = torch.randn([3,5,10, 64])
        content = torch.randn([3, 5, 30, 16])
        mask = torch.randn([10, 30]) > 0.5
        layer = Attention.MultiHeadedAttention(64, 16, 32, 4)
        attn = layer(query, content, content, mask)
        self.assertTrue(shape_equal([3, 5, 10, 32], attn.shape))

    def test_parallel_capable(self):
        """ Tests whether the class functions correctly when provided an ensemble"""
        query = torch.randn([3,5,10, 64])
        content = torch.randn([3, 5, 30, 16])
        mask = torch.randn([10, 30]) > 0.5
        layer = Attention.MultiHeadedAttention(64, 16, 32, 4, [3,5])
        attn = layer(query, content, content, mask)
        self.assertTrue(shape_equal([3, 5, 10, 32], attn.shape))


    def test_torchscript_compile(self):
        """ Tests that MHA works when torchscript compiled."""
        query = torch.randn([3,5,10, 64])
        content = torch.randn([3, 5, 30, 16])
        mask = torch.randn([10, 30]) > 0.5
        layer = Attention.MultiHeadedAttention(64, 16, 32, 4, 5)
        layer = torch.jit.script(layer)
        attn = layer(query, content, content, mask)


    @unittest.skipUnless(torch.cuda.is_available(), "Gpu not availble")
    def test_mha_cuda(self):
        """ tests if MHA works in a cuda environment."""
        device = torch.device("cuda")
        query = torch.randn([3,5,10, 64]).to(device)
        content = torch.randn([3, 5, 30, 16]).to(device)
        mask = (torch.randn([10, 30]) > 0.5).to(device)
        layer = Attention.MultiHeadedAttention(64, 16, 32, 4, 5).to(device)
        layer = torch.jit.script(layer)
        attn = layer(query, content, content, mask)

class test_PIMU(unittest.TestCase):
    """
    Test case for the PIMU class.
    """
    def test_basic(self):
        """ Test whether a basic PIMU instance works."""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.PIMU(32, 10, 4)
        shape = torch.tensor([3, 5, 10, 32])
        injected = layer(query)
        injected_shape = torch.tensor(injected.shape)
        self.assertTrue(torch.all(shape == injected_shape))

    def test_ensemble(self):
        """ Test whether an ensembled PIMU instance works."""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.PIMU(32, 10, 4, 5)
        injected = layer(query)
        self.assertTrue(shape_equal([3, 5, 10, 32], injected.shape))

    def test_torchscript_compile(self):
        """ Tests whether torchscript compiles on the layer"""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.PIMU(32, 10, 4, 5)
        layer = torch.jit.script(layer)
        injected = layer(query)

    @unittest.skipUnless(torch.cuda.is_available(), "no cuda environment")
    def test_cuda(self):
        """ Tests whether PIMA works in a cuda environment"""
        device = torch.device("cuda")
        query = torch.randn([3, 5, 10, 32]).to(device)
        layer = Attention.PIMU(32, 10, 4, 5).to(device)
        layer = torch.jit.script(layer)
        injected = layer(query)

class test_PISU(unittest.TestCase):
    def test_basic(self):
        """ Tests whether a basic, no frills PISU instance will work"""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.PISU(32, 16, 4, 4)
        summary = layer(query)

        summary_shape = torch.tensor(summary.shape)
        expected_shape = torch.tensor([3, 5, 4, 16])
        self.assertTrue(torch.all(summary_shape == expected_shape))

    def test_ensemble(self):
        """ Test whether the ensembled PISU instance works"""
        query = torch.randn([3,5, 10, 32])
        layer = Attention.PISU(32, 16, 20, 4, [3,5])
        summary = layer(query)
        self.assertTrue(shape_equal([3, 5, 20, 16], summary.shape))

    def test_torchscript_compile(self):
        """ Tests whether torchscript safely compiles"""
        query = torch.randn([3,5, 10, 32])
        layer = Attention.PISU(32, 16, 20, 4, 5)
        layer = torch.jit.script(layer)
        summary = layer(query)

    @unittest.skipUnless(torch.cuda.is_available(), "No gpu available")
    def test_cuda(self):
        """ Tests whether PISU works in a cuda environment"""
        device = torch.device("cuda")
        query = torch.randn([3,5, 10, 32]).to(device)
        layer = Attention.PISU(32, 16, 20, 4, 5).to(device)
        layer = torch.jit.script(layer)
        summary = layer(query)



class test_LCSA(unittest.TestCase):
    """

    Unit test for the Local Context Self Attention layer.

    """
    def test_basic(self):
        """ test whether it runs at all."""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.LCSA(32, 4, [1, 2, 5, 10])
        local_conditioning = layer(query)

        self.assertTrue(shape_equal([3, 5, 10, 32], local_conditioning.shape))
    def test_ensemble(self):
        """ test whether operating in ensemble mode causes any bugs"""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.LCSA(32, 4, [1, 2, 5, 10], parallelization=5)
        local_conditioning = layer(query)
        self.assertTrue(shape_equal([3, 5, 10, 32], local_conditioning.shape))
    def test_torchscript_compiles(self):
        """ test whether torchscript is willing to compile an initialized layer"""
        query = torch.randn([3, 5, 10, 32])
        layer = Attention.LCSA(32, 4, [1, 2, 5, 10], parallelization=5)
        layer = torch.jit.script(layer)
        local_conditioning = layer(query)
        self.assertTrue(shape_equal([3, 5, 10, 32], local_conditioning.shape))


if __name__ == '__main__':
    unittest.main()
