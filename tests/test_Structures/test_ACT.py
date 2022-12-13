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
import src.supertransformerlib.Structures.Accumulators as Accumulators
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

        spec = Accumulators.latticeSpec(1, 1)
        accum = Accumulators.haltingLatticeAccumulator(spec)

        test_tensor = torch.tensor([[0.5]])
        clamped_probabilities, remainder = accum.clamp_probabilities(test_tensor)
        self.assertTrue(clamped_probabilities == test_tensor)
        self.assertTrue(remainder == 0)
    def test_clamped_when_expected(self):
        """ Test clamping occurs, and with the right amount"""
        spec = Accumulators.latticeSpec(1, 1)
        accum = Accumulators.haltingLatticeAccumulator(spec)

        test_tensor = torch.tensor([[1.5]]) #Probability will add up greater than one
        expected_tensor = torch.tensor([[1.0]])
        clamped_probabilities, remainder = accum.clamp_probabilities(test_tensor)
        self.assertTrue(expected_tensor == clamped_probabilities)
        self.assertTrue(remainder == expected_tensor)
    def test_clamp_due_to_accumulated_probability(self):
        """ Test clamping occurs due to probability exhaustion"""
        spec = Accumulators.latticeSpec(1, 1)
        accum = Accumulators.haltingLatticeAccumulator(spec)

        test_tensor = torch.tensor([0.3]) #Probability will add up greater than one
        expected_tensor = torch.tensor([0.2])
        update = torch.tensor([0.8])
        accum.lattice_halting_probability = update # Manually set halting probability. Test action only.

        clamped_probabilities, remainder = accum.clamp_probabilities(test_tensor)
        self.assertAlmostEqual(float(clamped_probabilities), float(expected_tensor))
        self.assertAlmostEqual(float(remainder), float(expected_tensor))
    def test_clamp_raises_on_bad_shape(self):
        """Test method provides useful error info"""
        def tester():
            spec = Accumulators.latticeSpec(1, 1)
            accum = Accumulators.haltingLatticeAccumulator(spec)
            tensor = torch.tensor([1, 1, 3])
            accum.clamp_probabilities(tensor)
        self.assertRaises(Accumulators.Accumulator_Exception, tester)
    def test_multidimensional_clamp(self):
        """Test clamping in multiple dimensions at once."""
        spec = Accumulators.latticeSpec([10, 20], [12, 5])
        accum = Accumulators.haltingLatticeAccumulator(spec)
        test_tensor = 2*torch.rand([10, 20, 12, 5])
        accum.clamp_probabilities(test_tensor)
    def test_unhalted_elementwork(self):
        """ Test that displays of unhalted elements and elements per batch work."""
        spec = Accumulators.latticeSpec(3, 2)
        accum = Accumulators.haltingLatticeAccumulator(spec)
        manual_probability_tensor = torch.tensor([[0.0, 0.0],[1.0, 0.0], [1.0, 1.0]])
        accum.lattice_halting_probability = manual_probability_tensor

        expected_elements = torch.tensor([[True, True], [False, True], [False, False]])
        expected_batch = torch.tensor([True, True, False])

        self.assertTrue(torch.all(accum.unhalted_elements == expected_elements))
        self.assertTrue(torch.all(accum.unhalted_batches == expected_batch))

    def  test_get_penalty(self):
        """
        Test the get penalty method returns sane
        masks and penalties.
        :return:
        """

        #ACT mode
        spec = Accumulators.latticeSpec(3, 2)
        accum = Accumulators.haltingLatticeAccumulator(spec)
        manual_probability_tensor = torch.tensor([[0.0, 0.0],[1.0, 0.0], [1.0, 1.0]])
        expected_output = torch.tensor([[True, True], [False, True], [False, False]])
        accum.lattice_halting_probability = manual_probability_tensor
        self.assertTrue(torch.all(expected_output == accum.get_penalty(2)))

        #Filtration mode
        spec = Accumulators.latticeSpec(3, 2)
        accum = Accumulators.haltingLatticeAccumulator(spec,mode="Filtration")
        manual_probability_tensor = torch.tensor([[0.0, 0.0],[0.3, 0.0], [1.0, 1.0]])
        expected_output = torch.tensor([[1.0, 1.0],[0.7, 1.0], [0.0, 0.0]])
        accum.lattice_halting_probability = manual_probability_tensor
        self.assertTrue(torch.all(expected_output == accum.get_penalty(2)))

        #Extension requested
        spec = Accumulators.latticeSpec(1, 1)
        accum = Accumulators.haltingLatticeAccumulator(spec)
        manual_probability_tensor = torch.tensor([[1.0]])
        expected_output = torch.tensor([[[False]]])
        accum.lattice_halting_probability = manual_probability_tensor
        self.assertTrue(torch.all(expected_output == accum.get_penalty(3)))


    def test_simple_update(self):
        """ Test the update method is successful. Also test is_halted, unhalted_elements,
        and unhalted_batches
        """

        # Test proceeds to put 1.0 worth of probability into
        # the accumulator in a few steps.

        spec = Accumulators.latticeSpec(1, 1)
        accum = Accumulators.haltingLatticeAccumulator(spec)
        self.assertTrue(accum.lattice_halting_probability == 0)
        self.assertFalse(accum.is_halted)
        self.assertTrue(accum.unhalted_elements)

        update = torch.tensor([[0.3]])
        accum.update(update)
        self.assertTrue(accum.lattice_halting_probability == 0.3)
        self.assertFalse(accum.is_halted)
        self.assertTrue(accum.unhalted_elements)

        update = torch.tensor([[0.7]])
        accum.update(update)
        self.assertTrue(accum.lattice_halting_probability == 1.0)
        self.assertTrue(accum.is_halted)
        self.assertFalse(accum.unhalted_elements)\

    def test_map_into_sparseElements(self):
        """
        Test that when configured correctly the
        map is capable of isolating only the elements
        which require current computation correctly
        """

        spec = Accumulators.latticeSpec(2, 3)
        accum = Accumulators.haltingLatticeAccumulator(spec)

        update = torch.tensor([[1, 0, 1.0],[0, 1, 0]])
        mask = update < 0.5
        accum.update(update)

        # Develop test tensors and perform tests. Tests
        # are performed by mapping into sparseElements
        # mode with a few different tensor lengths.

        basic_test_tensor = torch.rand([2, 3])
        test_tensor_1d = torch.rand([2, 3, 4])
        test_tensor_4d = torch.rand([2, 3, 4, 5, 6, 7])

        expected_basic_tensor = basic_test_tensor[mask]
        expected_tensor_1d = test_tensor_1d[mask]
        expected_tensor_4d = test_tensor_4d[mask]

        self.assertTrue(torch.all(accum.map_tensor_into_sparseElements_space(basic_test_tensor)
                                  == expected_basic_tensor))
        self.assertTrue(torch.all(accum.map_tensor_into_sparseElements_space(test_tensor_1d)
                                  == expected_tensor_1d))
        self.assertTrue(torch.all(accum.map_tensor_into_sparseElements_space(test_tensor_4d)
                                  == expected_tensor_4d))

    def test_map_into_lattice_space(self):
        """ Test the ability of the model to map back into lattice space."""

        spec = Accumulators.latticeSpec(2, 3)
        accum = Accumulators.haltingLatticeAccumulator(spec)

        update = torch.tensor([[1, 0, 1.0],[0, 1, 0]])
        mask = update < 0.5
        num_unhalted = accum.unhalted_elements.sum()

        accum.update(update)

        # Develop test tensors and perform tests. Tests
        # are performed by mapping into sparseElements
        # mode with a few different tensor lengths.

        basic_test_tensor = torch.tensor([1., 2, 3])
        test_tensor_1d = torch.tensor([[1., 2],[3,4], [5, 6]])

        expected_basic_tensor = torch.tensor([[0, 1.0, 0],[2.0, 0, 3.0]])
        expected_tensor_1d = torch.tensor([[[0, 0], [1, 2.], [0, 0]],[[3,4],[0, 0], [5,6]]])


        self.assertTrue(torch.all(accum.map_tensor_into_lattice_space(basic_test_tensor)
                                  == expected_basic_tensor))
        self.assertTrue(torch.all(accum.map_tensor_into_lattice_space(test_tensor_1d)
                                  == expected_tensor_1d))

class test_Halting_Lattice_Integration(unittest.TestCase):
    """
    Tests that useful programs can be made using the class
    """
    def test_simple_act_halting_program(self):
        """
        Tests that when configured and fed values, it halts.

        This is a simple program which will do this.
        """


        spec = Accumulators.latticeSpec([10, 20], [12, 5])
        accumulator = Accumulators.haltingLatticeAccumulator(spec)

        while accumulator.is_halted is False:
            probability_updates = torch.rand(list(spec.total_shape))
            clamped_probs, _ = accumulator.clamp_probabilities(probability_updates)
            accumulator.update(clamped_probs)

        self.assertTrue(accumulator.is_halted)
    def test_simple_filtration_program(self):

        spec = Accumulators.latticeSpec([10, 20], [12, 5])
        accumulator = Accumulators.haltingLatticeAccumulator(spec, mode="Filtration")

        #Proxies for other things in the code
        get_synthetic_probabilities = lambda :  2*(torch.rand(list(spec.total_shape)) - 0.5)
        run_update_proxy = lambda data : data
        data_proxy = torch.randn(list(spec.total_shape) + [3, 4]) #Hypothetical state information for some model


        bias_rate = 0.3
        state = data_proxy
        for i in range(20):
            # Do stuff here
            state = run_update_proxy(state)

            # Update halting state
            probability_updates = get_synthetic_probabilities() + bias_rate
            clamped_probs, _ = accumulator.clamp_probabilities(probability_updates)
            accumulator.update(clamped_probs)

            if accumulator.is_halted:
                break

            state = data_proxy*accumulator.get_penalty(data_proxy.dim())
