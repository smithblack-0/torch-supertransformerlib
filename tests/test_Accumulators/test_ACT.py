"""
Test mechanisms for the classes and information utilized for
adaptive computation time and related derivative products
which can be utilized to perform adaptive halting.

This includes tests for retaining residuals, tests that
halting works properly, and tests that the programming
mechanism is sane

"""

import unittest
import torch
import src.supertransformerlib.Accumulation.ACT as ACT

class test_ACT(unittest.TestCase):
    def test_halts(self):
        batch_size = 4
        shape = 64

        spec = ACT.BatchSpec(shape, batch_size)
        halting_accumulator = ACT.haltingAccumulator(spec)
        ponder_accumulator = ACT.ponderAccumulator(spec)
        while halting_accumulator.is_halted is not True:
            #calculate probability updates. Your code goes here.
            halting_probabilities = torch.rand([batch_size, shape])
            print(halting_accumulator.halting_probability)
            # Adaptive Computation code.
            clamped_halting_probabilities, remainder = halting_accumulator.clamp_probabilities(halting_probabilities)
            halting_accumulator.update(clamped_halting_probabilities)
            ponder_accumulator.update(remainder)

            # Updates due to halting probabilities. Do using clamped halting probabilities

        print(ponder_accumulator.ponder_cost)