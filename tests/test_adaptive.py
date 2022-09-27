import torch
import unittest
from src.supertransformerlib import Adaptive

class test_AdaptiveMap(unittest.TestCase):
    """
    Test the ability to remap tensors using
    meshes.
    """
    def test_constructor(self):
        """Test the constructor makes the mesh and mapping correctly."""
        halted_probabilities = torch.tensor([[0.0, 0.0],[0.1, 1.0], [1.0, 1.0]])
        map = Adaptive.Adaptive_Map(halted_probabilities)

        expected_mapping = torch.tensor([[[0, 0], [0, 1]],[[1,0],[1,1]]])
        self.assertTrue(torch.all(map.mesh == expected_mapping))

        expected_mask = torch.tensor([[True, True], [True, False]])
        print(expected_mask.shape)
        print(map.mask.shape)
        self.assertTrue(torch.all(map.mask == expected_mask))
    def test_tensor_map_simple(self):
        """Test the map can restrict and update some simple tensor"""
        halted_probabilities = torch.tensor([[0.0, 0.0],[0.1, 1.0], [1.0, 1.0]])
        expected_restricted = torch.tensor([[0.0, 0.0],[0.1, 1.0]])
        expected_updated = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        map = Adaptive.Adaptive_Map(halted_probabilities)

        restricted = map.restrict(halted_probabilities)
        self.assertTrue(torch.all(expected_restricted == restricted))

        update = torch.zeros_like(restricted)
        updated = map.inverse(halted_probabilities, update)
        self.assertTrue(torch.all(expected_updated == updated))
    def test_tensor_map_complex(self):
        """Test that tensor map still performs when extra dimensions are involved beyond the query dim"""
        halting_probabilities = torch.clamp(2*torch.rand([10, 20, 30]), 0, 1)
        tensor = torch.randn([10, 20, 30, 5, 6])
        map = Adaptive.Adaptive_Map(halting_probabilities)

        restricted = map.restrict(tensor)
        update = map.inverse(tensor, restricted)
    def test_torchscript_compiles(self):
        halting_probabilities = torch.clamp(2*torch.rand([10, 20, 30]), 0, 1)
        map_func = torch.jit.script(Adaptive.Adaptive_Map)
        map = map_func(halting_probabilities)
        restricted = map.restrict(halting_probabilities)
        updated = map.inverse(halting_probabilities, restricted)
    def test_torchscript_metacompiles(self):

        @torch.jit.script
        def makemap(halting_probs):
            return Adaptive.Adaptive_Map(halting_probs)
        map = makemap(torch.rand([10, 10, 10]))


class test_Buffer(unittest.TestCase):
    """
    Test the ability of the buffer to function
    correctly
    """
    def test_torchscript(self):
        @torch.jit.script
        def makemap(hprobs):
            return Adaptive.Adaptive_Map(hprobs)

        cls = torch.jit.script(Adaptive.Batch_Buffer)

        @torch.jit.script
        def make_buffer(hprobs):
            return Adaptive.Batch_Buffer.start_buffer(hprobs)
        make_buffer(torch.rand([10, 10, 10]))



class test_Adaptive_Attention(unittest.TestCase):
    def test_constructor(self):
        """Test the constructor functions reasonably well"""
        Adaptive.Adaptive_Attention(10, 20, 30, 5, 5, 5)
    def test_call_simple(self):
        """Test if call works in the simplest of cases"""
        #Setup

        query = torch.randn([1, 5, 32])
        key = torch.randn([1, 5, 32])
        value = torch.randn([1, 5, 32])
        layer = Adaptive.Adaptive_Attention(32, 32, 32, 5, 6, 7)

        buffer = Adaptive.Batch_Buffer.start_buffer(query)
        accumulator = buffer.get_subaccumulator()

        #Call
        new_accumulator = layer(accumulator, query, key, value)
        self.assertTrue(torch.any(new_accumulator.Output != accumulator.Output))
    def test_call_intermediate(self):
        """Test if forward works in a mildly complicated situation"""
        query = torch.randn([10, 5, 32])
        key = torch.randn([10, 5, 48])
        value = torch.randn([10, 5, 64])
        layer = Adaptive.Adaptive_Attention(32, 48, 64, 5, 6, 7)

        buffer = Adaptive.Batch_Buffer.start_buffer(query, 64)
        accumulator = buffer.get_subaccumulator()

        new_accumulator = layer(accumulator, query, key, value)
        self.assertTrue(torch.any(new_accumulator.Output != accumulator.Output))

    def test_halted(self):
        """Test that when the probability gets full, we capture residuals and halt"""
        query = torch.randn([1, 5, 32])
        key = torch.randn([1, 5, 48])
        value = torch.randn([1, 5, 64])
        layer = Adaptive.Adaptive_Attention(32, 48, 64, 5, 6, 7)

        buffer = Adaptive.Batch_Buffer.start_buffer(query, 64)
        buffer = buffer.update(torch.ones([1, 5]))
        accumulator = buffer.get_subaccumulator()

        new_accumulator = layer(accumulator, query, key, value)
        self.assertTrue(torch.all(new_accumulator.Output == accumulator.Output))
        self.assertTrue(torch.all(new_accumulator.Residuals == accumulator.Residuals))

    def test_collects_residuals(self):
        """Test that when the probabity is fully exhausted residuals are collected"""
        query = torch.randn([10, 5, 32])
        key = torch.randn([10, 5, 48])
        value = torch.randn([10, 5, 64])
        layer = Adaptive.Adaptive_Attention(32, 48, 64, 5, 6, 7)

        buffer = Adaptive.Batch_Buffer.start_buffer(query, 64)
        buffer = buffer.update(0.9*torch.ones([10, 5]))
        accumulator = buffer.get_subaccumulator()

        new_accumulator = layer(accumulator, query, key, value)


        freshly_halted = new_accumulator.Halting_Probabilities > 1 - 0.00001
        residuals = new_accumulator.Residuals
        halted_residuals = residuals.masked_select(freshly_halted)
        expected_residuals = 0.1*torch.ones_like(halted_residuals)
        self.assertTrue(freshly_halted.sum() > 0) #With these kinds of conditions, SOMETHING must have halted\
        self.assertTrue(torch.all((expected_residuals-halted_residuals).abs() < 0.001))




class test_Adaptive_Attention_Integration(unittest.TestCase):
    """
    Tests that when put together, the mapping features and layers
    can perform adaptive attention over something like a batch
    """

    def test_mechanism_basic(self):
        """
        Test that mapping and halting may be elegantly performed.

        The logic here test ability to elegantly remap keys and
        perform attention until halted.
        """
        batch_mockup = torch.randn([20, 20, 10, 32])
        key_mockup = torch.randn([20, 20, 10, 64])
        layer_mockup = Adaptive.Adaptive_Attention(32, 64, 64, 4, 4, 4)

        buffer = Adaptive.Batch_Buffer.start_buffer(batch_mockup, 64)
        while not buffer.is_halted():
            subbatch = buffer.get_subaccumulator()
            subquery = buffer.Map.restrict(batch_mockup)
            subkey = buffer.Map.restrict(key_mockup)

            update = layer_mockup(subbatch, subquery, subkey, subkey)
            buffer.set_from_subaccumulator(update)
    @unittest.skipUnless(torch.cuda.is_available(), "gpu test requires valid gpu install")
    def test_mechanism_cuda(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        batch_mockup = torch.randn([20, 20, 10, 32]).to(device)
        key_mockup = torch.randn([20, 20, 10, 64]).to(device)
        layer_mockup = Adaptive.Adaptive_Attention(32, 64, 64, 4, 4, 4).to(device)

        buffer = Adaptive.Batch_Buffer.start_buffer(batch_mockup, 64)
        count = 0
        while not buffer.is_halted():
            count += 1
            subbatch = buffer.get_subaccumulator()
            subquery = buffer.Map.restrict(batch_mockup)
            subkey = buffer.Map.restrict(key_mockup)

            update = layer_mockup(subbatch, subquery, subkey, subkey)
            buffer.set_from_subaccumulator(update)
    def test_mechanism_as_torchscript(self):
        """Test that torchscript compilation is functional"""
        class test_torchscript(torch.nn.Module):
            def __init__(self, q_query, q_key, q_value, q_confidence, q_assembly, heads):
                super().__init__()
                self.attn = Adaptive.Adaptive_Attention(32, 64, 64, 4, 4, 4)
            def forward(self, batch_mockup, key_mockup):
                buffer = Adaptive.Batch_Buffer.start_buffer(batch_mockup, 64)
                while not buffer.is_halted():
                    subbatch = buffer.get_subaccumulator()
                    subquery = buffer.Map.restrict(batch_mockup)
                    subkey = buffer.Map.restrict(key_mockup)

                    update = self.attn(subbatch, subquery, subkey, subkey)
                    buffer.set_from_subaccumulator(update)
                return buffer

        batch_mockup = torch.randn([20, 20, 10, 32])
        key_mockup = torch.randn([20, 20, 10, 64])
        layer_mockup = test_torchscript(32, 64, 64, 4, 4, 4)
        layer_mockup = torch.jit.script(layer_mockup)
        layer_mockup(batch_mockup, key_mockup)

    def profile_cpu(self):

        from torch.profiler import profile, record_function,ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                self.test_mechanism_basic()

        print(prof.key_averages().table())
    def profile_gpu(self):
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                self.test_mechanism_cuda()

        print(prof.key_averages().table())