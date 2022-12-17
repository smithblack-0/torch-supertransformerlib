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

import src.supertransformerlib.Core as Core
import src.supertransformerlib.Core.errors
import src.supertransformerlib.Core.Functions
import src.supertransformerlib.Core.string_util
from typing import Tuple, List, Optional


# Not actually utilized, but manually impliments the process.
# It is instead far more efficient to use as strided

class LocalError(src.supertransformerlib.Core.Errors.ValidationError):
    """
    An error to throw when convolution is going badly
    """

    def __init__(self, reason: str, task: Optional[str] = None):
        self.reason = reason
        self.task = task
        super().__init__("LocalError", reason, task)


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
    #
    # We also calculate the min and maximum stridenum. This is basically
    # the minimum and maximum iteration in which an acceptible kernel
    # may be found

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

    current_strides = Core.get_strides(tensor)
    static_strides = current_strides[:-local_lengths]
    input_mutating_strides = current_strides[-local_lengths:]

    updated_strides = input_mutating_strides * stride
    local_strides = input_mutating_strides * dilation

    new_stride: List[int] = torch.concat(
        [static_strides, updated_strides, local_strides]).tolist()  # List so torchscript is happy

    ##################################################################
    # Calculate the required offsets for the particular final position, and clamp
    # negative entries. It is the case that a negative offset can be utilized
    # to perform sampling, but is not allowed by as strided. The behavior
    # in such a case is to shrink the sampled dimensions. We perform clamping to
    # account for it, and multipy then combine the provided offsets to figure out where in the
    # data buffer we start drawing from. Then we turn it into an int, so torchscript will
    # be happy

    # Conceptually, where we start looking at options. If the natural start is greater than
    # zero, then we must start our offset there as it will be inside the current buffer.

    ######## Offsets #######
    # Calculate the required offsets.
    #
    # Natural start is the place where drawing kernels
    # conceptually starts from. If it is positive, it is also related to what the offset
    # will be. Startpoint offset is the first place where a kernel could be drawn from
    # which matches the arithmetic lattice formed by the startpoint and the stride
    # within the indicated buffer.
    #
    # If natural start is less than zero, we go ahead and use the startpoint offset. It
    # will be the case under these circumstance the shape will shrink in size as well. Else
    # we use the natural start. It will be where we first draw from.
    #
    # Regardless, we then multiply the offsets by the strides and add them together
    # with the original offset to get the new offset.




    natural_start = offset + start * dilation
    startpoint_offsets = round_to_nearest_node(torch.zeros_like(dimensions_requiring_recalculation),
                                               stride, natural_start, "ceiling")

    working_offsets = torch.where(natural_start >= 0, natural_start, startpoint_offsets)
    working_offsets = working_offsets*input_mutating_strides
    new_offset: int = int(working_offsets.sum()) + tensor.storage_offset()

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

    current_strides = Core.get_strides(buffer)
    static_strides = current_strides[:-local_lengths]
    input_mutating_strides = current_strides[-local_lengths:]

    updated_strides = input_mutating_strides * stride
    local_strides = input_mutating_strides * dilation

    new_stride: List[int] = torch.concat(
        [static_strides, updated_strides, local_strides]).tolist()  # List so torchscript is happy

    # Finish developing the offsets. The offsets are currently defined per dimension, but
    # as_strided accepts a data buffer offset, not a vector offset. As a result, we combine
    # the offset with the stride to get the integer representation instead

    current_offset = buffer.storage_offset()
    new_offsets = input_mutating_strides * start_offsets
    new_offset = int(new_offsets.sum()) + current_offset

    # Perform the sampling. Then return the result

    return buffer.as_strided(new_shape, new_stride, new_offset)


torch.jit.script(padded_local_sample)

def circular_local_sample(
        tensor: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        stride: torch.Tensor,
        dilation: torch.Tensor,
        offset: torch.Tensor,
) -> torch.Tensor:
    """
    The circular local sampling system. Behaves very
    similar to the native sampling system, except
    locations which do not have a native view are padded
    with wrapped elements.
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
    buffer = Core.pad_circular(tensor, padding)

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

    current_strides = Core.get_strides(buffer)
    static_strides = current_strides[:-local_lengths]
    input_mutating_strides = current_strides[-local_lengths:]

    updated_strides = input_mutating_strides * stride
    local_strides = input_mutating_strides * dilation

    new_stride: List[int] = torch.concat(
        [static_strides, updated_strides, local_strides]).tolist()  # List so torchscript is happy

    # Finish developing the offsets. The offsets are currently defined per dimension, but
    # as_strided accepts a data buffer offset, not a vector offset. As a result, we combine
    # the offset with the stride to get the integer representation instead

    current_offset = buffer.storage_offset()
    new_offsets = input_mutating_strides * start_offsets
    new_offset = int(new_offsets.sum()) + current_offset

    # Perform the sampling. Then return the result
    outcome = buffer.as_strided(new_shape, new_stride, new_offset)
    return outcome



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
    :param mode: One of 'native', 'pad', or 'replicate'.
        * Native mode only figures out what local kernels can be completely made using the available tensor materials
        and returns those
        * pad mode ensures the primary tensor dimensions remain the same shape, and pads the missing elements with zero
        * replicate mode creates a infinitely scrolling grid where elements are replicated in sequence to fill in any
        missing padding zones.
    :return: A tensor with the same number of specified sampled dimensions, and with that many local dimensions
        tacked on.
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
        raise LocalError(reason, task)
    if start.shape[0] != end.shape[0]:
        reason = f"""\
        Param 'start' and param 'end' did not have 
        the same rank. 'start' has rank {start.shape[0]}
        while 'end' has rank {end.shape[0]}
        """
        reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
        raise LocalError(reason, task)
    if start.shape[0] != stride.shape[0]:
        reason = f"""\
        Param 'start' and param 'stride' did not have the
        same rank. 'start' has rank {start.shape[0]} while
        'stride' has rank {stride.shape[0]}
        """
        reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
        raise LocalError(reason, task)
    if start.shape[0] != dilation.shape[0]:
        reason = f"""\
        Param 'start' and param 'dilation' did not have
        the same rank. 'start' has rank {start.shape[0]} while
        'dilation' has rank {dilation.shape[0]}
        """
        reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
        raise LocalError(reason, task)
    if start.shape[0] != offset.shape[0]:
        reason = f"""\
        Param 'offset' and param 'start' did not have the
        same rank. 'start' had rank {start.shape[0]} while offset
        had rank {offset.shape[0]}
        """
        reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
        raise LocalError(reason, task)
    if torch.any(start > end):
        reason = f"""\
        Start nodes were higher than end nodes. This is not 
        allowed. 
        """
        reason = src.supertransformerlib.Core.StringUtil.dedent(reason)
        raise LocalError(reason, task)

    if mode == "native":
        return native_local_sample(tensor, start, end,
                                   stride, dilation, offset)
    elif mode == "pad":
        return padded_local_sample(tensor, start, end,
                                   stride, dilation, offset)
    elif mode == "replicate":
        return circular_local_sample(tensor, start, end,
                                     stride, dilation, offset)
    else:
        reason = f"""\
        It was expected that the local sampling
        function would be provided with a mode among
        'native', 'pad', or 'replication'. However, instead
        received '{mode}'. This is not allowed
        """
        reason = Core.dedent(reason)
        raise LocalError(reason, task)


class Local:
    """
    A virtual layer implimenting a given local operation.

    It may be initialized with the desired local quantities,
    and will then repeatively call the local sample
    operation with those specifications.
    """
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 stride: torch.Tensor,
                 dilation: torch.Tensor,
                 offset: torch.Tensor,
                 mode: str
                 ):
        self.start = start
        self.end = end
        self.stride = stride
        self.dilation = dilation
        self.offset = offset
        self.mode = mode
    def __call__(self, tensor: torch.Tensor)->torch.Tensor:
        return local_sample(tensor,
                            self.start,
                            self.end,
                            self.stride,
                            self.dilation,
                            self.offset,
                            self.mode
                            )


class LocalFactory(nn.Module):
    """
    The factory layer for the local sample operation

    Generates local layers to perform a particular sampling
    pattern. See Basics.Local_Sample.local_sample for more
    on what each parameter does. This layer will build
    a virtual layer calling local sample with the
    specified parameters.
    """
    Type = Local
    def __init__(self,
                 start: Core.StandardShapeType,
                 end: Core.StandardShapeType,
                 stride: Core.StandardShapeType,
                 dilation: Core.StandardShapeType,
                 offset: Core.StandardShapeType,
                 mode: str = "pad",
                 ):
        """
        :param start: The number of prior elements to include, along each dimension
        :param end: The number of post elements to include, along each dimension
        :param stride: The stride of the problem, along each dimension
        :param dilation: The dilation of the problem, along each dimension
        :param offset: The offset of the problem, along each dimension
        :param mode: One of 'native', 'pad', or 'replicate'.
        * Native mode only figures out what local kernels can be completely made using the available tensor materials
        and returns those
        * pad mode ensures the primary tensor dimensions remain the same shape, and pads the missing elements with zero
        * replicate mode creates a infinitely scrolling grid where elements are replicated in sequence to fill in any
        missing padding zones.
        """
        super().__init__()


        # Basic validation

        start = Core.standardize_shape(start, "start", allow_zeros=True, allow_negatives=True)
        end = Core.standardize_shape(end, "end", allow_zeros=True, allow_negatives=True)
        stride = Core.standardize_shape(stride, "stride", allow_zeros=False, allow_negatives=False)
        dilation = Core.standardize_shape(dilation, "dilation", allow_zeros=False, allow_negatives=False)
        offset = Core.standardize_shape(offset, "offset", allow_zeros=True, allow_negatives=True)

        valid_modes = ["native", "pad", "replicate"]
        Core.validate_string_in_options(mode, "mode", valid_modes, "padding mode")

        # Broadcast, eg, stride = 1 across all dimensions.

        target = max(start.shape[0],
                     end.shape[0],
                     stride.shape[0],
                     dilation.shape[0],
                     offset.shape[0]
        )
        start = torch.broadcast_to(start, [target])
        end = torch.broadcast_to(end, [target])
        stride = torch.broadcast_to(stride, [target])
        dilation = torch.broadcast_to(dilation, [target])
        offset = torch.broadcast_to(offset, [target])





        self.start = start
        self.end = end
        self.stride = stride
        self.dilation = dilation
        self.offset = offset
        self.mode = mode

    def forward(self)->Local:
        return Local(self.start,
                     self.end,
                     self.stride,
                     self.dilation,
                     self.offset,
                     self.mode)

