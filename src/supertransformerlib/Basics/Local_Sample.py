"""

A module for the mechanisms assocaited with convolutional sampling

Convolutional sampling is a mechanism by which the required information
for performing a particular convolution can be developed. These include
but are not exclusively restricted to parameters based around dilation,
prior elements, post elements, and offset.
"""
import torch
from torch import nn
from torch.nn import functional as F

import src.supertransformerlib.Core.Errors
import src.supertransformerlib.Core.Functions
import src.supertransformerlib.Core.StringUtil
from typing import Tuple, List, Optional


# Not actually utilized, but manually impliments the process.
# It is instead far more efficient to use as strided

class ConvolutionalError(src.supertransformerlib.Core.Errors.ValidationError):
    """
    An error to throw when convolution is going badly
    """

    def __init__(self, reason: str, task: Optional[str] = None):
        self.reason = reason
        self.task = task
        super().__init__("ConvolutionalError", reason, task)


def round_to_nearest_node(tensor: torch.Tensor,
                          node_spacing: torch.Tensor,
                          offset: Optional[torch.Tensor] = None,
                          mode: str = "floor") -> torch.Tensor:
    """
    Rounds to nearest node on an arithmetic sequence defined in
    terms of a offset and a node spacing. Has three modes,
    consisting of floor, ceil, and round

    :param tensor:
    :param node_spacing:
    :param offset:
    :param mode:
    :return:
    """

    if offset is None:
        offset = torch.zeros([1], dtype=tensor.dtype, device=tensor.device)

    temp = torch.div(tensor - offset, node_spacing)

    if mode == "floor":
        output = node_spacing * torch.floor(temp).to(dtype=tensor.dtype) + offset
    elif mode == "ceiling":
        output = node_spacing * torch.ceil(temp).to(dtype=tensor.dtype) + offset
    elif mode == "round":
        output = node_spacing * torch.round(temp).to(dtype=tensor.dtype) + offset
    else:
        raise ValueError("Did not get one of floor, ceiling, or round")
    return output


torch.jit.script(round_to_nearest_node)


def calculate_stridenums(dim_length: torch.Tensor,
                         start: torch.Tensor,
                         end: torch.Tensor,
                         stride: torch.Tensor,
                         dilation: torch.Tensor,
                         offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the minimum and maximum stride numbers

    These are the locations along the sample space in which
    it is the case that the first and last sample is taken.

    This process ignores whether the min stride number is greater
    than the max stride number.
    """

    natural_start = offset + start * dilation  # The theoredical beginning of sampling
    true_start = torch.where(natural_start < 0, 0, natural_start)  # The actual first location we can sample from
    true_start = round_to_nearest_node(true_start, stride, natural_start, "ceiling")
    min_stridenum = torch.div(true_start - natural_start, stride).to(
        dtype=torch.int64)  # The difference divided by strides is the stride num

    kernel_length = (end - start) * dilation + 1  # The length of the kernel
    natural_end = natural_start + dim_length - 1  # The natural sampling end point, traveling along the kernel
    end_by = dim_length - kernel_length  # If you do not end by here, you run out of elements
    true_end = torch.where(natural_end > end_by, end_by, natural_end)
    true_end = round_to_nearest_node(true_end, stride, natural_start, "floor")
    max_stridenum = torch.div(true_end - natural_start, stride, rounding_mode="floor")
    max_stridenum = max_stridenum + 1

    return min_stridenum, max_stridenum


torch.jit.script(calculate_stridenums)


def native_local_sample(
        tensor: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        stride: torch.Tensor,
        dilation: torch.Tensor,
        offset: torch.Tensor,
) -> torch.Tensor:
    """
    The 'Native' convolutional mode. Shrinks the output as
    would be expected when doing a convolution.

    It is the case that a kernel of the correct width
    is sampled from, starting at the zeroith
    dimension and then offset by the appropriate amount.

    Then only valid positions are sampled from.

    Understanding what the parameters stride, dilation,and offset
    do may be significantly assisted by checking out this link:

    https://github.com/vdumoulin/conv_arithmetic.


    Convolutional sampling is performed with respect to the
    tensor index with regards to the nearby neighbors.
    """

    # Calculate the new dynamic_shape. This will be found by finding the
    # difference between the minimum and maximum stride numbers,
    # which are a representation from the start of theoredical
    # sampling to the end of where we actually find sane
    # samples.
    #
    # We also calculate the local dimensions by calculating the
    # number of included kernel elements.

    local_lengths = start.shape[0]
    static_shape = torch.tensor(tensor.shape[:-local_lengths], dtype=torch.int64)
    dimensions_requiring_recalculation = torch.tensor(tensor.shape[-local_lengths:], dtype=torch.int64)

    min_stridenum, max_stridenum = calculate_stridenums(dimensions_requiring_recalculation,
                                                        start,
                                                        end,
                                                        stride,
                                                        dilation,
                                                        offset)

    update_shape = max_stridenum - min_stridenum
    update_shape = torch.where(update_shape < 0, 0, update_shape)
    local_shape = end - start + 1

    new_shape: List[int] = torch.concat(
        [static_shape, update_shape, local_shape]).tolist()  # List so torchscript is happy

    # Calculate the new strides. The strides will be utilized for both the purposes
    # of drawing information from particular dimensions, along with for drawing primary
    # information. Dimensions which have a stride modification entry will see their current
    # stride multiplied by the given one, and dilations are applied to the original strides
    # for sampling along the local dimension.

    current_strides = torch.tensor([tensor.stride(dim) for dim in range(tensor.dim())],
                                   dtype=torch.int64)  # Workaround for bad typing on Tensor.stride. Torchscript wont
                                                       # do tensor.stride()
    static_strides = current_strides[:-local_lengths]
    input_mutating_strides = current_strides[-local_lengths:]

    updated_strides = input_mutating_strides * stride
    local_strides = input_mutating_strides * dilation

    new_stride: List[int] = torch.concat(
        [static_strides, updated_strides, local_strides]).tolist()  # List so torchscript is happy

    # Calculate the required offsets for the particular final position, and clamp
    # negative entries. It is the case that a negative offset can be utilized
    # to perform sampling, but is not allowed by as strided. The behavior
    # in such a case is to shrink the sampled dimensions. We perform clamping to
    # account for it, and multipy then combine the provided offsets to figure out where in the
    # data buffer we start drawing from. Then we turn it into an int, so torchscript will
    # be happy

    natural_start = offset + start * dilation
    startpoint_offsets = round_to_nearest_node(torch.zeros_like(dimensions_requiring_recalculation),
                                               stride, natural_start, "ceiling")  # Locks on to the natural lattice.
    provided_offsets = torch.where(offset < 0, 0, offset)
    working_offsets = startpoint_offsets + provided_offsets
    working_offsets = working_offsets * input_mutating_strides
    new_offset = int(working_offsets.sum())

    # Perform the efficient strided view, and return the result

    return tensor.as_strided(new_shape, new_stride, new_offset)


torch.jit.script(native_local_sample)


def padded_local_sample(
        tensor: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        stride: torch.Tensor,
        dilation: torch.Tensor,
        offset: torch.Tensor,
) -> torch.Tensor:
    """

    The padded local sampling system. Behaves very
    similar to the native sampling system, except
    locations which do not have a native view have
    the missing parts filled in by padding. As a result
    the primary dimensions will retain the same dynamic_shape.
    """

    # We handle this problem by creating a new buffer containing a
    # padded version of the original tensor, with padding sufficient to
    # draw all requisite samples. Offsets are used to handle the fact
    # that potentially we will start partway inside the buffer.

    # Calculate the padding and offset requirements. Padding
    # will consist of whatever is needed to ensure the natural
    # offset starts cleanly, while offset requirements here may
    # be needed in order to handle positive offsets which will otherwise
    # start at zero. Once we figure out the padding, calculate the
    # buffer tensor

    local_lengths = start.shape[0]
    dim_lengths = torch.tensor(tensor.shape[-local_lengths:], dtype=torch.int64)
    natural_start = offset + start * dilation
    start_padding = torch.where(-natural_start < 0, 0, -natural_start)
    start_offsets = torch.where(natural_start > 0, natural_start, 0)

    kernel_length = (end - start + 1) * dilation
    natural_end = natural_start + dim_lengths - 1
    end_padding = natural_end - (
                dim_lengths - kernel_length - 1)  # The natural sampling end point, traveling along the kernel
    end_padding = torch.where(end_padding < 0, 0, end_padding)

    padding = torch.stack([start_padding, end_padding], dim=-1)
    padding = torch.flip(padding, dims=[0]) #Padding wants to define outputs last
    padding = padding.flatten()
    padding: List[int] = padding.tolist()
    buffer = F.pad(tensor, padding)

    # Calculate the required dynamic_shape. This involves looking at the
    # current dynamic_shape and dividing it by the stride then rounding down,
    # and including this along with the static dynamic_shape and local dims
    # enlarging the dimensions to account for the new locations.
    #
    # Note that the update dynamic_shape is the solution to
    # stride(n -1) >= length - 1, solved for n.

    static_shape = torch.tensor(tensor.shape[:-local_lengths], dtype=torch.int64)
    local_shape = end - start + 1
    update_shape = torch.div(dim_lengths - 1 + stride, stride, rounding_mode="floor")
    new_shape: List[int] = torch.concat([static_shape, update_shape, local_shape]).tolist()

    # Calculate the new strides. The strides for the
    # local dimension are influenced by the dilation, and
    # will consist of the stride of the dimension being sampled from times the
    # dilation. Meanwhile, the updated size will be multiplied by the stride
    # directive

    current_strides = torch.tensor([buffer.stride(dim) for dim in range(buffer.dim())],
                                   dtype=torch.int64)  # Workaround for bad typing on Tensor.stride. Torchscript wont do tensor.stride()
    static_strides = current_strides[:-local_lengths]
    input_mutating_strides = current_strides[-local_lengths:]

    updated_strides = input_mutating_strides * stride
    local_strides = input_mutating_strides * dilation

    new_stride: List[int] = torch.concat(
        [static_strides, updated_strides, local_strides]).tolist()  # List so torchscript is happy

    # Finish developing the offsets. The offsets are currently defined per dimension, but
    # as_strided accepts a data buffer offset, not a vector offset. As a result, we combine
    # the offset with the stride to get the integer representation instead

    new_offsets = input_mutating_strides * start_offsets
    new_offset = int(new_offsets.sum())

    # Perform the sampling. Then return the result

    return buffer.as_strided(new_shape, new_stride, new_offset)


torch.jit.script(padded_local_sample)


def local_sample(
        tensor: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        stride: torch.Tensor,
        dilation: torch.Tensor,
        offset: torch.Tensor,
        mode: str = "pad",
        task: Optional[str] = None,
) -> torch.Tensor:
    """
    Performs kernel sampling from an incoming kernel based on the indicated
    parameters. For the meaning of terms such as stride, dilation, etc see
    https://github.com/vdumoulin/conv_arithmetic.

    The kernel may be thought as beginning at location i = start*dilation + offset, then
    moving along by units stride. At each point, we look and see if we can draw a complete
    sample. If we can, we do. Otherwise, we look at the next sample point. We do this
    for tensor_dims//stride times.

    Mode "native" will only draw samples if the kernel is compatible, while mode
    "pad" will pad extra zeros to ensure there is no shrinkage due to the kernel,
    though striding may still cause shrinkage. Meanwhile, "rollaround" will do the same
    thing, but by wrapping the tensor instead.

    :param tensor: The tensor to transform
    :param start: The number of prior elements to include, along each dimension
    :param end: The number of post elements to include, along each dimension
    :param stride: The stride of the problem, along each dimension
    :param dilation: The dilation of the problem, along each dimension
    :param offset: The offset of the problem, along each dimension
    :return: A tensor of the same dynamic_shape, with one extra dimension. This extra dimension is generated
            by sampling the last N dimensions according to the provided kernel specifications, and
            concatenating all the samples together.
    """

    # Perform primary validation

    start = src.supertransformerlib.Core.Functions.standardize_shape(start, 'start', True, True, task)
    end = src.supertransformerlib.Core.Functions.standardize_shape(end, 'end', True, True, task)
    stride = src.supertransformerlib.Core.Functions.standardize_shape(stride, 'stride', False, False, task)
    dilation = src.supertransformerlib.Core.Functions.standardize_shape(dilation, 'dilation', False, False, task)
    offset = src.supertransformerlib.Core.Functions.standardize_shape(offset, 'offset', True, True, task)

    if start.shape[0] > tensor.dim():
        reason = f"""\
        A problem occurred while attempting to perform convolutional
        sampling. The passed start tensor was of rank {start.shape[0]},
        while the tensor only had rank {tensor.dim()}. This is not
        allowed.
        """
        reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
        raise ConvolutionalError(reason, task)
    if start.shape[0] != end.shape[0]:
        reason = f"""\
        Param 'start' and param 'end' did not have 
        the same rank. 'start' has rank {start.shape[0]}
        while 'end' has rank {end.shape[0]}
        """
        reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
        raise ConvolutionalError(reason, task)
    if start.shape[0] != stride.shape[0]:
        reason = f"""\
        Param 'start' and param 'stride' did not have the
        same rank. 'start' has rank {start.shape[0]} while
        'stride' has rank {stride.shape[0]}
        """
        reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
        raise ConvolutionalError(reason, task)
    if start.shape[0] != dilation.shape[0]:
        reason = f"""\
        Param 'start' and param 'dilation' did not have
        the same rank. 'start' has rank {start.shape[0]} while
        'dilation' has rank {dilation.shape[0]}
        """
        reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
        raise ConvolutionalError(reason, task)
    if start.shape[0] != offset.shape[0]:
        reason = f"""\
        Param 'offset' and param 'start' did not have the
        same rank. 'start' had rank {start.shape[0]} while offset
        had rank {offset.shape[0]}
        """
        reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
        raise ConvolutionalError(reason, task)
    if torch.any(start > end):
        reason = f"""\
        Start nodes were higher than end nodes. This is not 
        allowed. 
        """
        reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
        raise ConvolutionalError(reason, task)

    if mode == "native":
        return native_local_sample(tensor, start, end,
                                   stride, dilation, offset)
    elif mode == "pad":
        return padded_local_sample(tensor, start, end,
                                   stride, dilation, offset)


@torch.jit.script
def local(tensor: torch.Tensor,
          kernel_width: int,
          stride_rate: int,
          dilation_rate: int,
          start_offset: int = 0,
          end_offset: int = 0):
    """

    Description:

    This is a function designed to extract a series of kernels generated by standard convolutional
    keyword conventions which could, by broadcasted application of weights, be used to actually perform
    a convolution. The name "local" is due to the fact that the kernels generated are inherently a
    somewhat local phenomenon.

    When calling this function, a series of kernels with dynamic_shape determined by dilation_rate and kernel_width,
    and with number determined by stride_rate, will be generated along the last dimension of the input tensor.
    The output will be a tensor with an additional dimension on the end, with width equal to the size of
    the kernel, and the second-to-last dimension then indices these kernels.

    Note that the different between initial and final indexing dimensions is:
        compensation = (kernel_width - 1) * dilation_rate

    See 'dilocal' for a version of this which is fast when dealing with many dilations in parallel, as in banding.
    Padding by this much is guaranteed to prevent information loss.

    :param tensor: The tensor to take a local kernel out of
    :param kernel_width: How wide to make the kernel
    :param stride_rate: How fast to make the stride rate
    :param dilation_rate: How large the dilation rate should be
    :param start_offset: How long to wait before sampling from the tensor
    :param end_offset: Where to stop sampling from the tensor
    """

    # Input Validation

    assert kernel_width >= 1, "kernel_width should be greater than or equal to 1"
    assert stride_rate >= 1, "stride_rate should be greater than or equal to 1"
    assert dilation_rate >= 1, "dilation_rate should be greater than or equal to 1"
    assert start_offset >= 0
    assert end_offset >= 0

    # Construct dynamic_shape. Take into account the kernel_width, dilation rate, and stride rate.

    # The kernel width, and dilation rate, together modifies how far off the end of the
    # data buffer a naive implimentation would go, in an additive manner. Striding, meanwhile
    # is a multiplictive factor

    effective_length = tensor.shape[-1]
    effective_length = effective_length - start_offset - end_offset
    dilated_kernel_width = (kernel_width - 1) * (dilation_rate - 1) + kernel_width

    assert effective_length >= dilated_kernel_width, \
        ("With given start and end offset insufficient material remains for kernel", effective_length,
         dilated_kernel_width)

    effective_length = effective_length - dilated_kernel_width
    final_index_shape = (effective_length + stride_rate) // stride_rate  # Perform striding correction.

    static_shape = torch.tensor(tensor.shape[:-1], dtype=torch.int64)
    dynamic_shape = torch.tensor((final_index_shape, kernel_width), dtype=torch.int64)
    final_shape = torch.concat([static_shape, dynamic_shape])

    # Construct the stride. The main worry here is to ensure that the dilation striding, and primary
    # striding, now occurs at the correct rate. This is done by taking the current one, multiplying,
    # and putting this in the appropriate location.

    input_stride = [tensor.stride(dim) for dim in range(tensor.dim())]  # Workaround for bad typing on Tensor.stride

    static_stride = torch.tensor(input_stride[:-1], dtype=torch.int64)
    dynamic_stride = torch.tensor((stride_rate * input_stride[-1], dilation_rate * input_stride[-1]), dtype=torch.int64)
    final_stride = torch.concat([static_stride, dynamic_stride])

    # perform extraction. Return result

    final_shape: List[int] = final_shape.tolist()
    final_stride: List[int] = final_stride.tolist()
    return tensor[..., start_offset:-end_offset].as_strided(final_shape, final_stride)
