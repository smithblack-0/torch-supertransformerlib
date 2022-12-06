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
import src.supertransformerlib.Core as Core
class test_Halting_Lattice(unittest.TestCase):
    """
    Test fixture for the halting
    lattice accumulator

    The halting accumulator needs to perform three
    important features. These are:

    * Clamp the probabilities below one when trigged
    * Spit out the remainder when clamping
    * Update its halting probability stores when called, and display
      what has been halted, and if everything is halted.
    """
    def test_noclamp_when_expected(self):
        """Test clamping returns the correct manually set values"""

        spec = ACT.BatchSpec(1, 1)
        accum = ACT.haltingLattice(spec)

        test_tensor = torch.tensor([[0.5]])
        clamped_probabilities, remainder = accum.clamp_probabilities(test_tensor)
        self.assertTrue(clamped_probabilities == test_tensor)
        self.assertTrue(remainder == 0)
    def test_clamped_when_expected(self):
        """ Test clamping occurs, and with the right amount"""
        spec = ACT.BatchSpec(1, 1)
        accum = ACT.haltingLattice(spec)

        test_tensor = torch.tensor([[1.5]]) #Probability will add up greater than one
        expected_tensor = torch.tensor([[1.0]])
        clamped_probabilities, remainder = accum.clamp_probabilities(test_tensor)
        self.assertTrue(expected_tensor == clamped_probabilities)
        self.assertTrue(remainder == expected_tensor)
    def test_clamp_due_to_accumulated_probability(self):
        """ Test clamping occurs due to probability exhaustion"""
        spec = ACT.BatchSpec(1, 1)
        accum = ACT.haltingLattice(spec)

        test_tensor = torch.tensor([0.3]) #Probability will add up greater than one
        expected_tensor = torch.tensor([0.2])
        update = torch.tensor([0.8])
        accum.halting_probability = update # Manually set halting probability. Test action only.

        clamped_probabilities, remainder = accum.clamp_probabilities(test_tensor)
        self.assertAlmostEqual(float(clamped_probabilities), float(expected_tensor))
        self.assertAlmostEqual(float(remainder), float(expected_tensor))
    def test_clamp_raises_on_bad_shape(self):
        """Test method provides useful error info"""
        def tester():
            spec = ACT.BatchSpec(1, 1)
            accum = ACT.haltingLattice(spec)
            tensor = torch.tensor([1, 1, 3])
            accum.clamp_probabilities(tensor)
        self.assertRaises(ACT.ACT_Exception, tester)
    def test_multidimensional_clamp(self):
        """Test clamping in multiple dimensions at once."""
        spec = ACT.BatchSpec([10, 20], [12, 5])
        accum = ACT.haltingLattice(spec)
        test_tensor = 2*torch.rand([10, 20, 12, 5])
        accum.clamp_probabilities(test_tensor)
    def test_unhalted_elementwork(self):
        """ Test that displays of unhalted elements and elements per batch work."""
        spec = ACT.BatchSpec(3, 2)
        accum = ACT.haltingLattice(spec)
        manual_probability_tensor = torch.tensor([[0.0, 0.0],[1.0, 0.0], [1.0, 1.0]])
        accum.halting_probability = manual_probability_tensor

        expected_elements = torch.tensor([[True, True], [False, True], [False, False]])
        expected_batch = torch.tensor([True, True, False])

        self.assertTrue(torch.all(accum.unhalted_elements == expected_elements))
        self.assertTrue(torch.all(accum.unhalted_batches == expected_batch))
    def test_simple_update(self):
        """ Test the update method is successful. Also test is_halted, unhalted_elements,
        and unhalted_batches
        """

        # Test proceeds to put 1.0 worth of probability into
        # the accumulator in a few steps.

        spec = ACT.BatchSpec(1, 1)
        accum = ACT.haltingLattice(spec)
        self.assertTrue(accum.halting_probability == 0)
        self.assertFalse(accum.is_halted)
        self.assertTrue(accum.unhalted_elements)

        update = torch.tensor([[0.3]])
        accum.update(update)
        self.assertTrue(accum.halting_probability == 0.3)
        self.assertFalse(accum.is_halted)
        self.assertTrue(accum.unhalted_elements)

        update = torch.tensor([[0.7]])
        accum.update(update)
        self.assertTrue(accum.halting_probability == 1.0)
        self.assertTrue(accum.is_halted)
        self.assertFalse(accum.unhalted_elements)
    def test_simple_halting_program(self):
        """
        Tests that when configured and fed values, it halts.

        This is a simple program which will do this.
        """
        spec = ACT.BatchSpec([10, 20], [12, 5])
        accumulator = ACT.haltingLattice(spec)

        while accumulator.is_halted is False:
            probability_updates = torch.rand(list(spec.total_shape))
            clamped_probs, _ = accumulator.clamp_probabilities(probability_updates)
            accumulator.update(clamped_probs)

        self.assertTrue(accumulator.is_halted)
    def test_efficient_halting_program(self):
        """
        Tests that when configured and fed values, an efficient halting
        program can be constructed. This is a program which uses the fact
        some batchs have halted to in turn skip over those batches when
        developing.
        """

        spec = ACT.BatchSpec([10, 20], [12, 5])
        accumulator = ACT.haltingLattice(spec)
        stateblock = torch.zeros(list(spec.total_shape)) #Hypothetical state information for some model


        while accumulator.is_halted is False:
            # Gets the data by mask indexing. Returns a flatter version of the problem
            unhalted_batch_data = stateblock[accumulator.unhalted_batches]
            # Generate probability updates for each

            restricted_probability_updates = torch.rand(unhalted_batch_data.shape)

            # Returns to original shape
            restricted_indices = Core.gen_indices_from_mask(accumulator.unhalted_batches)
            general_probability_updates = torch.sparse_coo_tensor(restricted_indices,
                                                                  restricted_probability_updates,
                                                                  size = list(spec.total_shape))
            general_probability_updates = general_probability_updates.to_dense()

            # Update
            clamped_probs, _ = accumulator.clamp_probabilities(general_probability_updates)
            accumulator.update(clamped_probs)


class test_ACT(unittest.TestCase):
    def test_halts(self):
        batch_size = 4
        shape = 64
        embeddings = 10

        spec = ACT.BatchSpec(shape, batch_size)
        halting_accumulator = ACT.haltingLattice(spec)
        ponder_accumulator = ACT.ponderAccumulator(spec)


        output_accumulator = ACT.stateAccumulator(spec, embeddings)



        while halting_accumulator.is_halted is not True:
            #calculate probability updates. Your code goes here.
            update = torch.rand([batch_size, shape, embeddings])
            halting_probabilities = torch.rand([batch_size, shape])


            # Adaptive Computation code.
            clamped_halting_probabilities, remainder = halting_accumulator.clamp_probabilities(halting_probabilities)
            halting_accumulator.update(clamped_halting_probabilities)
            ponder_accumulator.update(remainder)

            # Updates due to halting probabilities. Do using clamped halting probabilities

            update = clamped_halting_probabilities.unsqueeze(-1)*update
            output_accumulator.update(update)
