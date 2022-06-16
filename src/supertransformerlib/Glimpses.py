"""

This is a module for the manipulation of tensors by means of lightweight memory views and
minimal padding. It extends the native torch functions in ways that I find useful.

All items within this module are functions. They all accept a tensor and parameters, then
do something with it. They also tend to return views to allow efficient memory utilization.

"""

import torch
from torch.nn import functional as F
from typing import Union, List


def view(tensor,
         input_shape: Union[torch.Tensor, List[int], int],
         output_shape: Union[torch.Tensor, List[int], int]) -> torch.Tensor:
    """
    This will, when passed an input shape and compatible output shape, assume that said shapes
    refer to the later dimensions in a tensor, as in broadcasting, and will perform a reshape from
    input shape to output shape while keeping all other dimensions exactly the same.


    ---- parameters ---

    :param tensor:
        The tensor to be modified.
    :param input_shape:
        The expected input shape. This can be a list/tuple of ints, or an int. It should represent the shape at the end
        of the input tensor's .shape which will be matched in the tensor input
    :param output_shape:
        The expected output shape. This can be a list/tuple of ints, or an int. It should represent the final shape one
        wishes the tensor to take. It also must be the case that the total size of the input and output shape must be the same.

    ---- Examples ----


    For tensors of shape:

    a = (5,2), b=(3, 4, 5,2), c=(30, 5,2),

    For input_shape = (5,2), output_shape=10, one has

    f(a, input_shape, output_shape) = shape(10)
    f(b, input_shape, output_shape) = shape(3, 4, 10)
    f(c, input_shape, output_shape) = shape(30, 10)


    """

    # Raw Type converison. The goal here is to end up with something solely
    # in terms of tensors.

    if torch.jit.isinstance(input_shape, int):
        input_shape = [input_shape]
    if torch.jit.isinstance(output_shape, int):
        output_shape = [output_shape]

    if torch.jit.isinstance(input_shape, List[int]):
        input_shape = torch.tensor(input_shape, dtype=torch.int64)
    if torch.jit.isinstance(output_shape, List[int]):
        output_shape = torch.tensor(output_shape, dtype=torch.int64)

    torch.jit.annotate(torch.Tensor, input_shape)
    torch.jit.annotate(torch.Tensor, output_shape)

    # Basic sanity testing
    assert input_shape.prod() == output_shape.prod(), \
        "Shapes incompatible: Input shape and output shape were not compatible: "

    # Perform view action.
    slice_length: int = len(input_shape)
    static_shape: torch.Tensor = torch.tensor(tensor.shape[:-slice_length], dtype=torch.int64)

    final_shape: torch.Tensor = torch.concat([static_shape, output_shape])
    final_shape: List[int] = final_shape.tolist()

    output: torch.Tensor = tensor.reshape(final_shape)
    return output


def reshape(tensor,
            input_shape: Union[torch.Tensor, List[int], int],
            output_shape: Union[torch.Tensor, List[int], int]) -> torch.Tensor:
    """
    This will, when passed an input shape and compatible output shape, assume that said shapes
    refer to the later dimensions in a tensor, as in broadcasting, and will perform a reshape from
    input shape to output shape while keeping all other dimensions exactly the same.


    ---- parameters ---

    :param tensor:
        The tensor to be modified.
    :param input_shape:
        The expected input shape. This can be a list/tuple of ints, or an int. It should represent the shape at the end
        of the input tensor's .shape which will be matched in the tensor input
    :param output_shape:
        The expected output shape. This can be a list/tuple of ints, or an int. It should represent the final shape one
        wishes the tensor to take. It also must be the case that the total size of the input and output shape must be the same.

    ---- Examples ----


    For tensors of shape:

    a = (5,2), b=(3, 4, 5,2), c=(30, 5,2),

    For input_shape = (5,2), output_shape=10, one has

    f(a, input_shape, output_shape) = shape(10)
    f(b, input_shape, output_shape) = shape(3, 4, 10)
    f(c, input_shape, output_shape) = shape(30, 10)


    """

    # Raw Type converison. The goal here is to end up with something solely
    # in terms of tensors.

    if torch.jit.isinstance(input_shape, int):
        input_shape = [input_shape]
    if torch.jit.isinstance(output_shape, int):
        output_shape = [output_shape]

    if torch.jit.isinstance(input_shape, List[int]):
        input_shape = torch.tensor(input_shape, dtype=torch.int64)
    if torch.jit.isinstance(output_shape, List[int]):
        output_shape = torch.tensor(output_shape, dtype=torch.int64)

    torch.jit.annotate(torch.Tensor, input_shape)
    torch.jit.annotate(torch.Tensor, output_shape)

    # Basic sanity testing
    assert input_shape.prod() == output_shape.prod(), \
        "Shapes incompatible: Input shape and output shape were not compatible: "

    # Perform view action.
    slice_length: int = len(input_shape)
    static_shape: torch.Tensor = torch.tensor(tensor.shape[:-slice_length], dtype=torch.int64)

    final_shape: torch.Tensor = torch.concat([static_shape, output_shape])
    final_shape: List[int] = final_shape.tolist()

    output: torch.Tensor = tensor.reshape(final_shape)
    return output


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


@torch.jit.script
def dilocal(tensor: torch.Tensor,
            kernel_width: int,
            stride_rate: int,
            dilations: Union[List[int], torch.Tensor],
            pad_to_input: bool = True,
            pad_value: float = 0.0,
            pad_justification: str = "center") -> torch.Tensor:
    """

    Performs the local operation in parallel with a variety of different dilations. Entries
    are padded with zero if there would not be a reference in the original tensor, such as
    edge dilations.

    :param tensor: The input to localize
    :param kernel_width: The kernel to use
    :param stride_rate: The stride rate to use
    :param dilations: A list of the dilations. Each one will end up on a head dimension
    :param pad_to_input_size: Whether or not to ensure the output is as wide as the input.
    :param pad_value: What to pad with.
    :param pad_justification: May be "forward", "center", or "backward".
    :return: tensor. Has shape (dilations, items, kernel_width).
    """
    assert isinstance(kernel_width, int)
    assert isinstance(stride_rate, int)
    assert pad_justification in ("forward", "center", "backward")

    if not isinstance(dilations, torch.Tensor):
        torch.jit.annotate(List[int], dilations)
        dilations = torch.tensor(dilations, dtype=torch.int64)
    torch.jit.annotate(torch.Tensor, dilations)

    # Calculate the offsets and paddings required to create the padbuffer.

    principle_padding = (kernel_width - 1) * (dilations.max() - 1)
    if pad_to_input:
        total_padding = principle_padding + kernel_width - 1
    else:
        total_padding = principle_padding

    particular_total_offsets = principle_padding - (kernel_width - 1) * (dilations - 1)
    particular_total_offsets = particular_total_offsets.type(torch.int64)

    if pad_justification == "forward":
        post_padding: int = 0
        prior_padding: int = int(total_padding)

        start_offsets = particular_total_offsets
        end_offsets = particular_total_offsets - start_offsets
    elif pad_justification == "center":
        post_padding: int = int(total_padding // 2)
        prior_padding: int = int(total_padding - post_padding)

        end_offsets = (particular_total_offsets / 2).type(torch.int64)
        start_offsets = particular_total_offsets - end_offsets
    else:
        post_padding: int = int(total_padding)
        prior_padding: int = 0

        end_offsets = particular_total_offsets
        start_offsets = particular_total_offsets - end_offsets
    pad_op = (prior_padding, post_padding)

    # Create the buffer, then create and stack the views.

    buffer = F.pad(tensor, pad_op, value=pad_value)
    local_views = []
    for dilation, start_offset, end_offset in zip(dilations, start_offsets, end_offsets):
        view_item = local(buffer, kernel_width, stride_rate, dilation, start_offset, end_offset)
        local_views.append(view_item)
    output = torch.stack(local_views, dim=-3)
    return output


def block(tensor, number):
    """

    Descrption:

    The purpose of this function is to split up the tensor
    into number equally sized units as a view.

    Excess is simply discarded.

    :param tensor:
    :param blocks:
    :return:
    """
    pass
