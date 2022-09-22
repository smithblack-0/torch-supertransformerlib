import torch
import unittest
from src.supertransformerlib import Adaptive



class test_Calculate_Output_Update(unittest.TestCase):
    """
    Tests for the calculate update layer
    """

    def test_constructor(self):
        """Test the constructor runs at all."""
        Adaptive.Calculate_Output_Update(10, 10)
    def test_random(self):
        """Test that the call runs with legal arguments"""
        halting_probs = torch.zeros([2, 3],dtype=torch.float32)
        state = torch.randn([2, 3, 12])
        state_update = torch.randn([2, 1, 12])
        updater = Adaptive.Calculate_Output_Update(12)
        update_probs, update_res, update_output = updater(halting_probs, state, state_update)
        print(halting_probs+update_probs)
        print((halting_probs+update_probs).sum(dim=-2))
        print(update_res)
        print(update_output)
    def test_repetition(self):
        """Test the behavior upon repetition."""

        halting_probs = torch.zeros([2, 3],dtype=torch.float32)
        state = torch.randn([2, 3, 12])
        state_update = torch.randn([2, 1, 12])
        updater = Adaptive.Calculate_Output_Update(12)
        update_probs, update_res, update_output = updater(halting_probs, state, state_update)

        print(update_probs)
        print(update_res)

        halting_probs += update_probs
        update_probs, update_res, update_output = updater(halting_probs, state, state_update)

        print(update_probs)
        print(update_res)

class test_state_engine(unittest.TestCase):
    def test_get_unhalted_mesh(self):
        """Test the get unhalted feature works properly."""

        state = torch.randn([2, 2, 2, 3])
        h_prob = torch.tensor([[[0.1, 0.3], [0.4, 0.6]],[[0.2, 0.4],[0.5,0.5]]])
        t_prob = torch.randn([2,2])
        h_res = torch.zeros([2])
        t_res = torch.zeros([2])
        output =  torch.zeros([2])
        trashcan = torch.zeros([2])

        engine = Adaptive.State_Engine(
            state,
            h_prob,
            t_prob,
            h_res,
            t_res,
            output,
            trashcan
        )
        dim1, dim0 = engine.get_unhalted_mesh()
        unhalted = h_prob[dim1, dim0]
        self.assertTrue(torch.all(unhalted == torch.tensor([[0.1, 0.3],[0.2, 0.4]])))
    def test_get_details(self):
        """
        Test the ability of get details to retrieve the required features
        based on the state, filtering out irrelevant information and applying
        required masks.
        :return:
        """

        state = torch.randn([2, 2, 2, 3])
        h_prob = torch.tensor([[[0.1, 0.3], [0.4, 0.6]],[[0.2, 0.4],[0.5,0.5]]])
        t_prob = torch.tensor([[[0.4, 0.1], [0.1, 0.3]],[[0.2, 0.4],[0.5,0.5]]])
        h_res = torch.zeros([2])
        t_res = torch.zeros([2])
        output =  torch.zeros([2])
        trashcan = torch.zeros([2, 1, 3])

        engine = Adaptive.State_Engine(
            state,
            h_prob,
            t_prob,
            h_res,
            t_res,
            output,
            trashcan
        )

        state, halting_probs, trash_probs = engine.get_details()
        print(state)
        print(halting_probs)
        print(trash_probs)