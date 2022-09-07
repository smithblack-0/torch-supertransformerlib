import unittest
import torch

from src.supertransformerlib import Layers
from src.supertransformerlib.Core import Linear



class testBandedAttn(unittest.TestCase):
    def testBasic(self):
        """ Test whether the defaults are working """
        query = torch.arange(96).view(1, 4, 24).type(torch.float32)
        key = query.clone()
        value = query.clone()

        tester = Layers.BandedMultiheadedAttention(24, 3)
        tester(query, key, value)
    def testCompressive(self):
        """ Test whether or not compression and expansion abilities are functioning"""
        query = torch.randn([2,3,80,80])
        key1 = torch.randn([2, 3, 40, 80])
        key2 = torch.randn([2, 3, 160, 80])
        value1 = key1.detach()
        value2 = key2.detach()

        tester_decompress = Layers.BandedMultiheadedAttention(80, 20, compression_ratio=(1, 2))
        tester_compress = Layers.BandedMultiheadedAttention(80, 20, compression_ratio=(2, 1))

        test_decompress = tester_decompress(query, key2, value2)
        test_compress = tester_compress(query, key1, value1)
    def testGradients(self):
        """ Tests that gradients are updating when a layer is run """
        query = torch.arange(96).view(1, 4, 24).type(torch.float32)
        key = query.clone()
        value = query.clone()

        tester = Layers.BandedMultiheadedAttention(24, 3)
        test_optim = torch.optim.Adam(tester.parameters())
        params = [item.clone().detach() for item in tester.parameters()]

        for _ in range(10):
            test_optim.zero_grad()
            test_result = tester(query, key, value)
            loss = test_result.sum()**2

            loss.backward()
            test_optim.step()

        for initial, final in zip(params, tester.parameters()):
            test = torch.not_equal(initial, final)
            test = torch.any(test)
            self.assertTrue(test, "Parameters not updating")
    def test_Saveable(self):
        """ Tests whether or not saving, then loading, works properly"""

        query = torch.arange(96).view(1, 4, 24).type(torch.float32)
        key = query.clone()
        value = query.clone()

        tester = Layers.BandedMultiheadedAttention(24, 3)
        tester2 = Layers.BandedMultiheadedAttention(24, 3)

        state_dict = tester.state_dict()
        tester2.load_state_dict(state_dict)

        resulta = tester(query, key, value)
        resultb = tester2(query, key, value)
        self.assertTrue(torch.equal(resulta, resultb))

    def test_jit(self):
        """Tests whether or not the layer can be jit compiled. """
        query = torch.arange(96).view(1, 4, 24).type(torch.float32)
        key = query.clone()
        value = query.clone()

        tester = Layers.BandedMultiheadedAttention(24, 3)
        tester = torch.jit.script(tester)
        tester(query, key, value)


if __name__ == "__main__":
    unittest.main()