import unittest
import torch
from src.supertransformerlib import Attention


class TestMultiHeadedAttention(unittest.TestCase):
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
        heads = 4
        batch_shape = [3, 5]
        query = torch.randn(batch_shape + [10, 32])
        key = torch.randn(batch_shape + [10, 32])
        value = torch.randn(batch_shape + [10, 32])


        factory = Attention.MultiHeadedAttentionFactory(32, heads)
        factory = torch.jit.script(factory)
        layer = factory()

        attn = layer(query, key, value)
        self.assertTrue(attn.shape == torch.Size(batch_shape + [10, 32]))

    def test_uneven_embeddings(self):
        """ tests if attention works when the embeddings are changing dynamic_shape """
        heads = 4
        d_output = 78
        batch_shape = [3, 5]
        query = torch.randn(batch_shape + [10, 64])
        key = torch.randn(batch_shape + [30, 128])
        value = torch.randn(batch_shape + [30, 52])

        factory = Attention.MultiHeadedAttentionFactory(d_model = 64,
                                                        d_key = 128,
                                                        d_value = 52,
                                                        heads = heads,
                                                        d_output = d_output)
        factory = torch.jit.script(factory)

        layer = factory()
        attn = layer(query, key, value)
        self.assertTrue(attn.shape == torch.Size(batch_shape + [10, d_output]))

    def test_masked_embeddings(self):
        """ tests if attention works when a mask is involved """

        batch_shape = [3, 5]
        query = torch.randn(batch_shape + [10, 64])
        content = torch.randn(batch_shape + [30, 16])
        mask = torch.randn(batch_shape + [10, 30]) > 0.5

        factory = Attention.MultiHeadedAttentionFactory(64, 4,
                                                        16, 16, 32)
        factory = torch.jit.script(factory)

        layer = factory()
        attn = layer(query, content, content, mask)

        self.assertTrue(attn.shape ==  torch.Size([3, 5, 10, 32]))

    def test_parallel_capable(self):
        """ Tests whether the class functions correctly when provided an ensemble"""
        heads = 5
        query = torch.randn([3,5,10, 64])
        content = torch.randn([3, 5, 30, 16])
        mask = torch.randn([10, 30]) > 0.5

        factory = Attention.MultiHeadedAttentionFactory(64, heads,)

        layer = Attention.MultiHeadedAttention(64, 16, 32, 4, [3,5])
        attn = layer(query, content, content, mask)
        self.assertTrue(shape_equal([3, 5, 10, 32], attn.shape))



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