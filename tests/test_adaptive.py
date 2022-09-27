import torch
import unittest
from src.supertransformerlib import Adaptive
from torch.profiler import profile, record_function, ProfilerActivity


class test_AdaptiveMap(unittest.TestCase):
    """
    Test the ability to remap tensors using
    meshes.
    """
    def test_constructor(self):
        """Test the constructor makes the mesh and mapping correctly."""
        halted_probabilities = torch.tensor([[0.0, 0.0],[0.1, 1.0], [1.0, 1.0]])
        map = Adaptive.Adaptive_Map(halted_probabilities)

        expected_mapping = torch.tensor([0, 1])
        self.assertTrue(torch.all(map.index == expected_mapping))

        expected_mask = torch.tensor([[True, True], [True, False]])
        print(expected_mask)
        print(map.mask)
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
        updated = map.update(halted_probabilities, update)
        self.assertTrue(torch.all(expected_updated == updated))
    def test_tensor_map_complex(self):
        """Test that tensor map still performs when extra dimensions are involved beyond the query dim"""
        halting_probabilities = torch.clamp(2*torch.rand([10, 20, 30]), 0, 1)
        tensor = torch.randn([10, 20, 30, 5, 6])
        map = Adaptive.Adaptive_Map(halting_probabilities)


        restricted = map.restrict(tensor)
        restricted = -torch.ones_like(restricted)
        update = map.update(tensor, restricted)

        #Check that updates are not occuring when halted
        halted = halting_probabilities >= 1 - 0.001
        halted = halted.unsqueeze(-1).unsqueeze(-1)
        halted = halted.expand([-1, -1, -1, 5, 6])
        unhalted = torch.logical_not(halted)

        #Any updated with a halted probability should not have updated
        orig = tensor.masked_select(halted)
        final = update.masked_select(halted)
        self.assertTrue(torch.all(orig == final), "When halted probabilities were present, update occurred.")

        #The updates with an unhalted probability should now be -1,
        #which is not within the torch.rand [0, 1] domain.
        orig = tensor.masked_select(unhalted)
        final = update.masked_select(unhalted)
        self.assertTrue(torch.all(orig != final))
    def test_torchscript_compiles(self):
        """Test if torchscript is willing to do a proper update"""
        halting_probabilities = torch.clamp(2*torch.rand([10, 20, 30]), 0, 1)
        map_func = torch.jit.script(Adaptive.Adaptive_Map)
        map = map_func(halting_probabilities)
        restricted = map.restrict(halting_probabilities)
        updated = map.update(halting_probabilities, restricted)
    def test_torchscript_metacompiles(self):
        """Test if torchscript will do updates when indirection occurs. """
        #This test exists because torchscript was refusing to do vector
        #indexing. That is, indexing of nature tensor[index.unbind(-1)]
        #did not work.

        @torch.jit.script
        def makemap(halting_probs):
            return Adaptive.Adaptive_Map(halting_probs)
        map = makemap(torch.rand([10, 10, 10]))
        tensor = torch.randn([10, 10, 10, 30])
        restriction = map.restrict(tensor)
        update = map.update(tensor, restriction)



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

        buffer = Adaptive.Adaptive_Translator.start_buffer(query)
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

        buffer = Adaptive.Adaptive_Translator.start_buffer(query, 64)
        accumulator = buffer.get_subaccumulator()

        new_accumulator = layer(accumulator, query, key, value)
        self.assertTrue(torch.any(new_accumulator.Output != accumulator.Output))

    def test_halted(self):
        """Test that when the probability gets full, we capture residuals and halt"""
        query = torch.randn([1, 5, 32])
        key = torch.randn([1, 5, 48])
        value = torch.randn([1, 5, 64])
        layer = Adaptive.Adaptive_Attention(32, 48, 64, 5, 6, 7)

        buffer = Adaptive.Adaptive_Translator.start_buffer(query, 64)
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

        buffer = Adaptive.Adaptive_Translator.start_buffer(query, 64)
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

    def get_test_mechanism(self)->torch.nn.Module:
        class test_mechanism(torch.nn.Module):
            def __init__(self, q_query, q_key, q_value, q_confidence, q_assembly, heads):
                super().__init__()
                self.attn = Adaptive.Adaptive_Attention(32, 64, 64, 4, 4, 4)

            def forward(self, batch_mockup, key_mockup):
                buffer = Adaptive.Adaptive_Translator.start_buffer(batch_mockup, 64)
                while not buffer.is_done():
                    subbatch = buffer.get_subaccumulator()
                    subquery = buffer.g(batch_mockup)
                    subkey = buffer.get_unhalted_from_tensor(batch_mockup)

                    update = self.attn(subbatch, subquery, subkey, subkey)
                    buffer.set_from_subaccumulator(update)
                return buffer

        layer_mockup = test_mechanism(32, 64, 64, 4, 4, 4)
        return layer_mockup

    def get_test_tensors(self):
        batch_mockup = torch.randn([20, 20, 10, 32])
        key_mockup = torch.randn([20, 20, 10, 64])
        return batch_mockup, key_mockup

    def test_mechanism_cpu(self):
        """
        Test that mapping and halting may be elegantly performed.

        The logic here test ability to elegantly remap keys and
        perform attention until halted.
        """

        layer = self.get_test_mechanism()
        batch, key = self.get_test_tensors()
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("basic_cpu_profiling"):
                buffer = layer(batch, key)
        print("basic cpu profiling", prof.key_averages().table())


    @unittest.skipUnless(torch.cuda.is_available(), "gpu test requires valid gpu install")
    def test_mechanism_cuda(self):

        layer = self.get_test_mechanism().to("cuda")
        batch, key = self.get_test_tensors()
        batch = batch.to("cuda")
        key = key.to("cuda")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("basic_gpu_profiling"):
                layer(batch, key)

        print("basic gpu profiling", prof.key_averages().table())


    def test_cpu_as_torchscript(self):
        """Test that torchscript compilation is functional"""
        layer = self.get_test_mechanism()
        layer = torch.jit.script(layer)
        batch, key = self.get_test_tensors()

        with profile(activities=[], record_shapes=True) as prof:
            with record_function("cpu_torchscript"):
                buffer = layer(batch, key)

        print("torchscript cpu profiling", prof.key_averages().table())

    @unittest.skipUnless(torch.cuda.is_available(), "gpu test requires valid gpu install")
    def test_gpu_as_torchscript(self):

        layer = self.get_test_mechanism()
        layer = torch.jit.script(layer)
        batch, key = self.get_test_tensors()
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("basic_cpu_profiling"):
                buffer = layer(batch, key)
        print("basic cpu profiling", prof.key_averages().table())
