import unittest
import torch
import itertools
import src.supertransformerlib.Core.SampleUtils as SampleUtils

class test_top_P_Mask(unittest.TestCase):
    """
    A test case for top p
    """

    def manually_calculate_top_p(self, tensor: torch.Tensor, probability_threshold: float)->torch.Tensor:
        """
        Manually calculates top p using clearly defined for loop logic
        Naturally, this is very slow and so not utilized in practice.

        However, it is very easy to tell if it is correct.
        """

        if tensor.dim() == 1:
            # Base case.
            #
            # Make a buffer of the same shape as tensor filled
            # with false. Then iterate over the sorted entries.
            #
            # For each entry which has not yet added up to be
            # above the probability threshold, set that unit to
            # true. Return when done.

            total_probability = 0
            output_tensor = torch.full_like(tensor, False, dtype=torch.bool)

            values, index = torch.sort(tensor, dim=-1, descending=True)
            for value, index in zip(values, index):
                output_tensor[index] = True
                total_probability += value
                if total_probability > probability_threshold:
                    break
            return output_tensor

        # Tail case. We break apart the
        # batch and process each tensor dimension
        # individually.

        layers = tensor.unbind(0)
        output = []
        for layer in layers:
            outcome = self.manually_calculate_top_p(layer, probability_threshold)
            output.append(outcome)
        return torch.stack(output, dim=0)

    def test_run_1d_cases(self):

        dims = list(range(1, 100))
        probs = [0.0, 0.4, 0.7, 1.0]
        options = itertools.product(dims, probs)

        for dim_length, probability_threshold in options:
            tensor = torch.rand([dim_length])
            tensor = torch.softmax(tensor, dim=-1)

            try:
                expected = self.manually_calculate_top_p(tensor, probability_threshold)
                got = SampleUtils.top_p_mask(tensor, probability_threshold)
                self.assertTrue(torch.all(expected == got))
            except Exception as err:
                expected = self.manually_calculate_top_p(tensor, probability_threshold)
                got = SampleUtils.top_p_mask(tensor, probability_threshold)
                self.assertTrue(torch.all(expected == got))

    def test_run_batched_cases(self):

        dim0 = [1, 5, 10]
        dim1 = [1, 5, 10]
        dim2 = [1, 5, 10]
        probs = [0.0, 0.4, 0.7, 1.0]
        options = itertools.product(dim0, dim1, dim2, probs)

        for dim0, dim1, dim2, probability_threshold in options:
            tensor = torch.randn([dim0, dim1, dim2])
            tensor = torch.softmax(tensor, dim=-1)

            try:
                expected = self.manually_calculate_top_p(tensor, probability_threshold)
                got = SampleUtils.top_p_mask(tensor, probability_threshold)
                self.assertTrue(torch.all(expected == got))
            except Exception as err:
                expected = self.manually_calculate_top_p(tensor, probability_threshold)
                got = SampleUtils.top_p_mask(tensor, probability_threshold)
                self.assertTrue(torch.all(expected == got))

