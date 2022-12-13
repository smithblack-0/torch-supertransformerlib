import torch
import unittest
from src.supertransformerlib.Attention import Utility


class test_dotproductattention(unittest.TestCase):
    """
    Tests the dot product attention function loose in the
    function module.
    """
    def test_basic_dotproductattn(self):
        """Tests if a random, no frills case works safely."""
        query = torch.randn([3, 5, 10, 20])
        content = torch.randn([3, 5, 30, 20])
        attn = Utility.dot_product_attention(query, content, content)

    def test_masked_dotproductattn(self):
        """ tests if a masked case functions correctly """
        query = torch.randn([3, 5, 10, 20])
        content = torch.randn([3, 5, 30, 20])
        mask = torch.randn([10, 30]) > 0.5
        attn = Utility.dot_product_attention(query, content, content, mask)

    def test_torchscript_compile(self):
        test = torch.jit.script(Utility.dot_product_attention)

    @unittest.skipUnless(torch.cuda.is_available(), "gpu test requires valid gpu install")
    def test_gpu(self):
        """ tests if the gpu runs okay """
        device = torch.device("cuda")
        query = torch.randn([3, 5, 10, 20]).to(device)
        content = torch.randn([3, 5, 30, 20]).to(device)
        mask = (torch.randn([10, 30]) > 0.5).to(device)
        attn = Utility.dot_product_attention(query, content, content, mask)

class test_CreateHead(unittest.TestCase):
    """
    Test mechanism for the class designed
    to project heads. Used in many other classe
    """
    def test_simple(self):

        test_tensor = torch.randn([10, 12, 64])
        d_model = 64
        heads = 5
        d_heads = d_model//heads

        factory = Utility.MakeHeadFactory(d_model, heads, d_heads)
        factory = torch.jit.script(factory)

        layer = factory()
        layer(test_tensor)

    def test_save_load(self):

        test_tensor = torch.randn([10, 12, 64])
        d_model = 64
        heads = 5
        d_heads = d_model//heads

        factory = Utility.MakeHeadFactory(d_model, heads, d_heads)
        factory = torch.jit.script(factory)

        path = "testfixture.data"
        torch.jit.save(factory, path)
        otherfactory = torch.jit.load(path)

        layer = otherfactory()
