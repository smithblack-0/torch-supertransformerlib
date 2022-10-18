"""

A module for the mechanisms assocaited with convolutional sampling

Convolutional sampling is a mechanism by which the required information
for performing a particular convolution can be developed. These include
but are not exclusively restricted to parameters based around dilation,
prior elements, post elements, and offset.
"""
import torch
from torch import nn
from src.supertransformerlib.Core import Core
from typing import Tuple, List

# Not actually utilized, but manually impliments the process.
# It is instead far more efficient to use as strided
def example_generator(test_list: torch.Tensor, start: int, end: int, dilation: int,  stride: int, offset: int):
    output: List[torch.Tensor] = []
    for i in range(0, len(test_list), stride):
        options = torch.range(start, end).to(dtype=torch.int64)*dilation + i + offset
        if torch.all(0 <= options) and torch.all(options < len(test_list)):
            sample = test_list[options]
            output.append(sample)
    return output

def calculate_min_max_sample_locations(
        dim_length: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        stride: torch.Tensor,
        dilation: torch.Tensor,
        offset: torch.Tensor,
        )->Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the conceptual minimum sample
    location and maximum sample location. Assumes we
    are bounded only on the nearest edge.


    :param dim_length: How long the dimension is.
    :param start: The start point for the kernel
    :param end: The end point for the kernel
    :param stride: The degree of the stride.
    :param dilation: The amount of the dilation
    :param offset: The amount of the offset
    :return: A tuple of two tensors. Prior shrinkage and post shrinkage.
        These represent the degree to which shrinkage occurs from the front and
        from the back. With some cases, these might not be equal.
    """

    # The interaction between the elements above are quite annoying, but not
    # unconquerable. Based on the example from above, it is the case
    # that there will be a series of locations along the tensor which
    # are used to sample from, and then a kernel defined with respect to
    # those sampling locations.
    #
    # Let the sampling locations be i. They will be defined in terms
    # of i = range(0, dim_length, stride) + offset,
    #
    # For each of these locations, we will look at them and sample
    # only if the minimum sample location >= 0, and maximum < dim_length.
    # These corrospond to:
    #
    #  dim_length > start*dilation  + i >= 0
    # 0 <= end*dilation + i < dim_length
    #
    # from which constraints can rapidly be placed on those
    # i's for the start and end restriction respectively.
    #
    # i >= -start*dilation
    # i < dim_length - end*dilation
    # i <= dim_length - end*dilation - 1
    # We can round up or down to the nearest stride node
    # and we have a formula to beat the problem, so long as we do
    # not allow illegal values. We count the legal strides
    # and get the solution


    minimum_unstrided_sampling_location = -start * dilation
    mintemp = torch.div(minimum_unstrided_sampling_location - offset, stride)
    offset_minimum_location = stride*torch.ceil(mintemp) + offset

    maximum_unstrided_sampling_location = dim_length - end * dilation - 1
    maxtemp = torch.div(maximum_unstrided_sampling_location - offset, stride)
    offset_maximum_location = stride*torch.floor(maxtemp) + offset

    return offset_minimum_location, offset_maximum_location

def calculate_requisite_padding(
        dim_length: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        stride: torch.Tensor,
        dilation: torch.Tensor,
        offset: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates required amount of additional padding needed in order to
    ensure entire defined sample space can be sampled from.

    :param dim_length:
    :param start:
    :param end:
    :param stride:
    :param dilation:
    :param offset:
    :return:
    """


)


def convolutional_sample(
                  tensor: torch.Tensor,
                  start: torch.Tensor,
                  end: torch.Tensor,
                  stride: torch.Tensor,
                  dilation: torch.Tensor,
                  offset: torch.Tensor,
                  mode: str = "pad",
                  )->torch.Tensor:
    """
    Performs kernel sampling from an incoming kernel based on the indicated
    parameters. For the meaning of terms such as stride, dilation, etc see
    https://github.com/vdumoulin/conv_arithmetic.

    The kernel is always thought to be defined with respect to the current index, and
    always includes at least one element. Prior elements and post elements then
    defines with respect to this central element how many things to include in
    a pythagoric manner. For instance, with prior_elements being -1, and post elements
    being 1, we include one prior element, and one post element in addition to the primary
    element.


    :param tensor: The tensor to transform
    :param start: The number of prior elements to include, along each dimension
    :param end: The number of post elements to include, along each dimension
    :param stride: The stride of the problem, along each dimension
    :param dilation: The dilation of the problem, along each dimension
    :param offset: The offset of the problem, along each dimension
    :return: A tensor of the same shape, with one extra dimension. This extra dimension is generated
            by sampling the last N dimensions according to the provided kernel specifications, and
            concatenating all the samples together.
    """






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

    When calling this function, a series of kernels with shape determined by dilation_rate and kernel_width,
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

    # Construct shape. Take into account the kernel_width, dilation rate, and stride rate.

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