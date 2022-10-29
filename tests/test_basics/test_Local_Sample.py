"""

Test features for the local
kernel sampling functions and features

"""
import unittest
import torch
from torch.nn import functional as F
import itertools
from src.supertransformerlib.Basics import Local_Sample
from src.supertransformerlib import Core
from typing import List


# Tools

def native_stridenum_generator(test_list: torch.Tensor, start: int, end: int,  stride: int,  dilation: int, offset: int):
    iterater = range(0, test_list.shape[-1], stride)
    counter = 0
    max_output = None
    min_output = None
    for i in iterater:
        options = torch.range(start, end).to(dtype=torch.int64)*dilation + i + offset
        if torch.all(0 <= options) and torch.all(options < test_list.shape[-1]):
            if min_output is None:
                min_output = counter
            max_output = counter + 1
            t = i + offset
        counter += 1
    return min_output, max_output



def native_1d_example_generator(test_list: torch.Tensor, start: int, end: int, stride: int, dilation: int, offset: int):
    """
    Manually generates a native view equivalent, in a not-very-efficient loop.
    Used for testing and for giving an idea as to how the damn view shoud
    work.

    Works in 1d, on the last dimension.
    """

    output: List[torch.Tensor] = []
    iterater = range(0, test_list.shape[-1], stride)
    for i in iterater:

        if start <= end:
            options = torch.range(start, end).to(dtype=torch.int64) * dilation + i + offset
        else:
            options = torch.empty([0], dtype=torch.int64)
        if torch.all(0 <= options) and torch.all(options < test_list.shape[-1]):
            sample = test_list[..., options]
            output.append(sample)
    if len(output) > 0:
        return torch.stack(output, -2)
    else:
        shape = [*test_list.shape[:-1], 0, end-start + 1]
        return torch.empty(shape, dtype = test_list.dtype)

def native_nd_example_generator(test_tensor: torch.Tensor,
                                start: torch.Tensor,
                                end: torch.Tensor,
                                stride: torch.Tensor,
                                dilation: torch.Tensor,
                                offset: torch.Tensor
                                ):
    """
    Generates nd examples using vector indexing. Less efficient than primary designs.
    Primarily tests if the striding mechanisms and such are configured correctly.
    """

    interesting_length = start.shape[0]
    dim_lengths = torch.tensor(test_tensor.shape[-interesting_length:])
    locations = [torch.arange(0, dim, striding) for dim, striding in zip(dim_lengths, stride)]
    ranges = [torch.range(starting, ending).to(dtype=torch.int64)*dilating + offsetting for
              starting, ending, dilating, offsetting in zip(start, end, dilation, offset)]

    #Develop compatible draw regions.
    samplings = []
    for location, r, dim_length in zip(locations, ranges, dim_lengths):

        #Develop canadidate samples
        location = location.unsqueeze(-1)
        r = r.unsqueeze(0)
        sample = location + r

        #Exclude samples which are drawing off the end of the array
        #Exclude samples which are drawing from before the array

        satisfied_end_condition = torch.all(sample < dim_length, dim=-1)
        satisfied_start_condition = torch.all(sample >= 0, dim=-1)
        valid = torch.logical_and(satisfied_start_condition, satisfied_end_condition)
        sample = sample[valid]

        samplings.append(sample)

    # Expand the samplings properly such that the sample instructions can be
    # broadcast together
    for i, sample in enumerate(samplings):
        #Develop samplings for broadast

        sampled_dim = i
        local_dim = i + interesting_length

        for j in range(2*len(samplings)):
            if j < sampled_dim:
                sample = sample.unsqueeze(0)
            if local_dim > j > sampled_dim:
                sample = sample.unsqueeze(sampled_dim + 1)
            if j > local_dim:
                sample = sample.unsqueeze(-1)
        samplings[i] = sample

    #Perform sampling process, then restore

    output = test_tensor
    for _ in range(interesting_length):
        output = output.movedim(-1, 0)

    samplings = tuple(samplings)
    output = output[samplings]

    for _ in range(2*interesting_length):
        output = output.movedim(0, -1)

    return output

def padded_1d_example_generator(test_list: torch.Tensor, start: int, end: int, stride: int, dilation: int, offset: int):
    """
    Manually generates a native view equivalent, in a not-very-efficient loop.
    Used for testing and for giving an idea as to how the damn view shoud
    work.

    Works in 1d, on the last dimension.
    """

    output: List[torch.Tensor] = []
    iterater = range(0, test_list.shape[-1], stride)
    for i in iterater:

        if start <= end:
            options = torch.range(start, end).to(dtype=torch.int64) * dilation + i + offset
        else:
            options = torch.empty([0], dtype=torch.int64)

        # Collect zeros where needed by identifying where we will draw
        # off the end of the tensor, concatenating on a zeros tensor,
        # and drawing from that instead.

        zeros_drawpoint = torch.zeros([*test_list.shape[:-1], 1], dtype=test_list.dtype)
        test_source = torch.concat([test_list, zeros_drawpoint], dim=-1)
        needs_redirection = torch.logical_or(options < 0, options >= test_list.shape[-1])
        drawpoints = torch.where(needs_redirection, test_source.shape[-1] - 1, options)

        # Perform a sampling, while drawing zeros where needed

        sample = test_source[..., drawpoints]
        output.append(sample)
    if len(output) > 0:
        return torch.stack(output, -2)
    else:
        shape = [*test_list.shape[:-1], 0, end-start + 1]
        return torch.empty(shape, dtype = test_list.dtype)

def padded_nd_example_generator(test_tensor: torch.Tensor,
                                start: torch.Tensor,
                                end: torch.Tensor,
                                stride: torch.Tensor,
                                dilation: torch.Tensor,
                                offset: torch.Tensor
                                ):
    """
    Generates nd examples using vector indexing. Less efficient than strided
    design, but considerably less complex. Utilized to ensure primary design
    matches theoredical specifications.
    """

    # Basically, this works by jumping along the length of dimensions by stride
    # and padding out any region which does not have an entry at that location.

    interesting_length = start.shape[0]
    dim_lengths = torch.tensor(test_tensor.shape[-interesting_length:])
    locations = [torch.arange(0, dim, striding) for dim, striding in zip(dim_lengths, stride)]
    ranges = [torch.range(starting, ending).to(dtype=torch.int64)*dilating + offsetting for
              starting, ending, dilating, offsetting in zip(start, end, dilation, offset)]

    # Develop compatible draw regions. Also, develop draw tensor
    # with padded region to draw upon.

    start_padding = torch.zeros([interesting_length], dtype=torch.int64)
    end_padding = torch.ones([interesting_length], dtype=torch.int64)
    padding = torch.stack([start_padding, end_padding], dim=-1).flatten().tolist()
    buffer = F.pad(test_tensor, padding)

    samplings = []
    for location, r, dim_length in zip(locations, ranges, dim_lengths):

        #Develop canadidate samples
        location = location.unsqueeze(-1)
        r = r.unsqueeze(0)
        sample = location + r

        # Refactor sampling directives such that samples drawing off the
        # end of the array are reformatted to draw off the padded
        # region instead


        unsatisfied_end_condition = sample >= dim_length
        unsatisfied_start_condition = sample < 0
        invalid = torch.logical_or(unsatisfied_end_condition, unsatisfied_start_condition)
        sample = torch.where(invalid, dim_length, sample)
        samplings.append(sample)

    # Expand the samplings properly such that the sample instructions can be
    # broadcast together
    for i, sample in enumerate(samplings):
        #Develop samplings for broadast

        sampled_dim = i
        local_dim = i + interesting_length

        for j in range(2*len(samplings)):
            if j < sampled_dim:
                sample = sample.unsqueeze(0)
            if local_dim > j > sampled_dim:
                sample = sample.unsqueeze(sampled_dim + 1)
            if j > local_dim:
                sample = sample.unsqueeze(-1)
        samplings[i] = sample

    #Perform sampling process, then restore

    output = buffer
    for _ in range(interesting_length):
        output = output.movedim(-1, 0)

    samplings = tuple(samplings)
    output = output[samplings]

    for _ in range(2*interesting_length):
        output = output.movedim(0, -1)

    return output



# Tests



class test_gen_stridenum(unittest.TestCase):
    """
    Test that generate minimum stridenum generator works properly
    """
    tensor = torch.tensor(torch.arange(200).view(10, 20))
    dim_lengths = torch.tensor(tensor.shape[-1])
    starts = torch.tensor([-3, -2, -1, 0])
    ends = torch.tensor([0, 1, 2, 3, 4])
    strides = torch.tensor([1, 2, 3, 4])
    dilations = torch.tensor([1, 2, 3, 4])
    offsets = torch.tensor([-3, -2, -1, 0, 1, 2, 3])
    options = itertools.product(starts, ends, strides, dilations, offsets)

    def test_cases(self):
        """Test a bunch of cases to ensure logic is sound. Ignore cases where no return is found, as they will be validated layer"""
        for option in self.options:
            try:
                min_expected, max_expected = native_stridenum_generator(self.tensor, *option)
                min_got, max_got = Local_Sample.calculate_stridenums(self.dim_lengths, *option)
                if min_expected is not None:
                    self.assertTrue(min_expected == min_got)
                if max_expected is not None:
                    self.assertTrue(max_expected == max_got)
            except Exception as err:
                #Can set break points here when troubleshooting
                min_expected, max_expected = native_stridenum_generator(self.tensor, *option)
                min_got, max_got = Local_Sample.calculate_stridenums(self.dim_lengths, *option)
                if min_expected is not None:
                    self.assertTrue(min_expected == min_got)
                if max_expected is not None:
                    self.assertTrue(max_expected == max_got)

    def test_torchscript(self):
        """ Test that this very important function works under torchscript"""

        tensor = torch.tensor(torch.arange(200).view(10, 20))
        dim_length = torch.tensor(tensor.shape[-1])
        starts = torch.tensor([-1])
        ends = torch.tensor([1])
        strides = torch.tensor([2])
        dilations = torch.tensor([4])
        offsets = torch.tensor([-1])

        scripted_func = torch.jit.script(Local_Sample.calculate_stridenums)
        output = scripted_func(dim_length, starts, ends, strides, dilations, offsets)

class test_native_cases(unittest.TestCase):
    """
    Uses the native test case generator to test a variety of different cases with
    different specifications
    """
    def test_tools(self):
        """ Test that the test example generating tools agree. Particularly
        test that the verified 1d case and the nd example generator agree
        when working with 1d problems
        """

        tensor = torch.arange(200).view(10, 20)
        starts = torch.tensor([-3, -2, -1, 0])
        ends = torch.tensor([0, 1, 2, 3, 4])
        strides = torch.tensor([1, 2, 3, 4])
        dilations = torch.tensor([1, 2, 3, 4])
        offsets = torch.tensor([-3, -2, -1, 0])
        options = itertools.product(starts, ends, strides, dilations, offsets)
        for option in options:
            try:
                expected = native_1d_example_generator(tensor, *option) #This is known to be correct
                option2 = [item.unsqueeze(-1) for item in option]
                got = native_nd_example_generator(tensor, *option2)
                self.assertTrue(torch.all(expected == got))# This is believed to be correct.
            except Exception as err:
                expected = native_1d_example_generator(tensor, *option) #This is known to be correct
                option = [item.unsqueeze(-1) for item in option]
                got = native_nd_example_generator(tensor, *option)
                self.assertTrue(torch.all(expected == got))# This is believed to be correct.

    def test_1d_cases(self):
        """Tests a variety of 1d cases against the 1d example generator."""
        tensor = torch.arange(200).view(10, 20)
        starts = [-3, -2, -1, 0, -1, -2]
        ends = [-2, -1, 0, 1, 2, 3, 4]
        strides = [1, 2, 3, 4]
        dilations = [1, 2, 3, 4]
        offsets = [-3, -2, -1, 0]

        options = itertools.product(starts, ends, strides, dilations, offsets)

        for option in options:
            if option[0] >= option[1]:
                #Ignores when start node is higher then end node.
                continue

            try:
                expected = native_1d_example_generator(tensor, *option)
                got = Local_Sample.local_sample(tensor, *option, mode="native")
                self.assertTrue(torch.all(expected == got))
            except Exception as err:

                expected = native_1d_example_generator(tensor, *option)
                got = Local_Sample.local_sample(tensor, *option, mode="native")
                self.assertTrue(torch.all(expected == got))
                raise err


    def test_nd_case(self):
        """ Test we still work properly in multiple dimensions"""
        tensor = torch.arange(490).view(10, 7, 7)


        nd = 2
        starts = [torch.tensor([-3, -2, -1, 0])]*nd
        ends = [torch.tensor([0, 1, 2, 3, 4])]*nd
        strides =[torch.tensor([1, 2, 3, 4])]*nd
        dilations = [torch.tensor([1, 2, 3, 4])]*nd
        offsets = [torch.tensor([-3,-2, 0,1, 3])]*nd

        starts = torch.cartesian_prod(*starts).unbind(0)
        ends = torch.cartesian_prod(*ends).unbind(0)
        strides = torch.cartesian_prod(*strides).unbind(0)
        dilations = torch.cartesian_prod(*dilations).unbind(0)
        offsets = torch.cartesian_prod(*offsets).unbind(0)

        options = itertools.product(starts, ends, strides, dilations, offsets)
        for option in options:
            try:
                expected = native_nd_example_generator(tensor, *option)
                got = Local_Sample.local_sample(tensor, *option, mode="native")
                self.assertTrue(torch.all(expected == got))
            except Exception as err:
                #Set breakpoints here when debugging.
                expected = native_nd_example_generator(tensor, *option)
                got = Local_Sample.local_sample(tensor, *option, mode="native")
                self.assertTrue(torch.all(expected == got))

class test_padded_cases(unittest.TestCase):
    """
    Test the padded local sampling method, and verifies
    it matches the hypothetical ideal.
    """

    def test_tools(self):
        """ Test that the test example generating tools agree. Particularly
        test that the verified 1d case and the nd example generator agree
        when working with 1d problems
        """

        tensor = torch.arange(200).view(10, 20)
        starts = torch.tensor([-3, -2, -1, 0, 1, 2])
        ends = torch.tensor([-1, -2, 0, 1, 2, 3, 4])
        strides = torch.tensor([1, 2, 3, 4])
        dilations = torch.tensor([1, 2, 3, 4])
        offsets = torch.tensor([-3, -2, -1, 0, 1, 2, 3])
        options = itertools.product(starts, ends, strides, dilations, offsets)
        for option in options:
            if option[0] > option[1]:
                continue

            try:
                expected = padded_1d_example_generator(tensor, *option) #This is known to be correct
                option2 = [item.unsqueeze(-1) for item in option]
                got = padded_nd_example_generator(tensor, *option2)
                self.assertTrue(torch.all(expected == got))# This is believed to be correct.
            except Exception as err:
                expected = padded_1d_example_generator(tensor, *option) #This is known to be correct
                option = [item.unsqueeze(-1) for item in option]
                got = padded_nd_example_generator(tensor, *option)
                self.assertTrue(torch.all(expected == got))# This is believed to be correct.


    def test_1d_cases(self):
        """Tests a variety of 1d cases against the 1d example generator."""
        tensor = torch.arange(200).view(10, 20)
        starts = [-3, -2, -1, 0, -1, -2]
        ends = [-2, -1, 0, 1, 2, 3, 4]
        strides = [1, 2, 3, 4]
        dilations = [1, 2, 3, 4]
        offsets = [-3, -2, -1, 0]

        options = itertools.product(starts, ends, strides, dilations, offsets)

        for option in options:
            if option[0] >= option[1]:
                #Ignores when start node is higher then end node.
                continue

            try:
                expected = padded_1d_example_generator(tensor, *option)
                got = Local_Sample.local_sample(tensor, *option, mode="pad")
                self.assertTrue(torch.all(expected == got))
            except Exception as err:

                expected = padded_1d_example_generator(tensor, *option)
                got = Local_Sample.local_sample(tensor, *option, mode="pad")
                self.assertTrue(torch.all(expected == got))

                raise err

    def test_nd_case(self):
        """ Test we still work properly in multiple dimensions"""
        tensor = torch.arange(98).view(2, 7, 7)


        nd = 2
        starts = [torch.tensor([-2, -1, 0])]*nd
        ends = [torch.tensor([0, 1, 3, 4])]*nd
        strides =[torch.tensor([1, 2, 3, 4])]*nd
        dilations = [torch.tensor([1, 2, 3, 4])]*nd
        offsets = [torch.tensor([-3, 0,1, 3])]*nd

        starts = torch.cartesian_prod(*starts).unbind(0)
        ends = torch.cartesian_prod(*ends).unbind(0)
        strides = torch.cartesian_prod(*strides).unbind(0)
        dilations = torch.cartesian_prod(*dilations).unbind(0)
        offsets = torch.cartesian_prod(*offsets).unbind(0)

        options = list(itertools.product(starts, ends, strides, dilations, offsets))
        numcases = len(options)
        print("testing %s possibilities" % numcases)
        counter = 0
        for option in options:
            try:
                expected = padded_nd_example_generator(tensor, *option)
                got = Local_Sample.local_sample(tensor, *option, mode="pad")
                self.assertTrue(torch.all(expected == got))
            except Exception as err:
                #Set breakpoints here when debugging.
                print(tensor[0])

                expected = padded_nd_example_generator(tensor, *option)
                print(expected[0], "done")

                got = Local_Sample.local_sample(tensor, *option, mode="pad")
                self.assertTrue(torch.all(expected == got))
            counter += 1
            if counter % 1000 == 0:
                print(counter)

class test_native_errors(unittest.TestCase):
    """
    Test that native convolution throws appropriate
    errors when called.
    """
    def test_tensor_rank_insufficient(self):
        pass
    def test_tensor_dim_not_large_enough_simple(self):
        pass
    def test_tensor_dim_not_large_enough_dilated(self):
        pass
    def test_tensor_dim_not_large_enough_2d(self):
        pass