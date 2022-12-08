from typing import List, Optional
import torch
import itertools
import src.supertransformerlib.Core.Errors as Errors
import src.supertransformerlib.Core.StringUtil as StringUtil



def calculate_shape_strides(shape: List[int])->torch.Tensor:
    """
    Calculate and return the strides associated
    with a particular tensor dynamic_shape assuming
    the strides are defined with the last dim
    having the smallest jump

    :param tensor: The tensor to calculate the strides of
    :return: The calculated strides
    """

    shape = list(shape)
    shape.reverse()

    cumulative_stride = 1
    strides: List[int] = []
    for dim_length in shape:
        strides.insert(0, cumulative_stride)
        cumulative_stride = cumulative_stride * dim_length

    return torch.tensor(strides)


torch.jit.script(calculate_shape_strides)


def gen_indices_from_mask(mask: torch.Tensor)->torch.Tensor:
    """
    Generates sparse tensor indices from a mask of a particular dynamic_shape.

    This can be combined with a masked select to quickly create a hybrid
    or sparse tensor

    :param mask: The boolean mask to generate from
    :return: int64 index consisting of [dims, items]
    """

    assert mask.dtype == torch.bool
    indices = torch.meshgrid([torch.arange(i) for i in mask.shape], indexing="ij")
    indices = torch.stack(indices, dim=-1)
    indices = indices[mask]
    indices = indices.transpose(0, 1)
    return indices

torch.jit.script(gen_indices_from_mask)


def convert_dense_to_hybrid(tensor: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
    """
    Converts a tensor into a hybrid sparse tensor by using a mask
    which indicates what to include. It is assumed we are
    working from the first dimension forward.

    For a tensor with shape [5, 3, 7] and a mask of
    shape [5,3], for example, you will end up with a
    sparse tensor with two sparse dimensions and one
    last dense dimension.

    :param tensor: The tensor to convert. Must be a dense tensor
    :param mask: The mask. Must be a bool tensor
    :return: The hybrid tensor to return.
    """

    index = gen_indices_from_mask(mask)
    values = tensor[mask]
    return torch.sparse_coo_tensor(index, values)

torch.jit.script(convert_dense_to_hybrid)

def masked_matrix_multiplication(tensor_a: torch.Tensor,
                                 tensor_b: torch.Tensor,
                                 mask: torch.Tensor)->torch.Tensor:
    """
    An algorithm for performing matrix multiplication in which a mask can be
    provided preventing certain elements from contributing anything. Elements
    marked as True contributes. False does not. Broadcasting is supported.

    Lets talk details. Broadcasting is completely supported across
    any number of dimensions. Tensor_a and Tensor_b work pretty much as normal. One can
    batched matrix multiply any two tensors of shapes
    [...common, n, m] and [...common, m, l] together. The mask,
    meanwhile, should be of shape [...common, n, m, l] and broadcasting
    is allowed.

    :param tensor_a: The first dense tensor to matrix multiply
    :param tensor_b: The second dense tensor to matrix multiply
    :param mask: The mask to apply. This is applied right before beginning sumation
    :return: The result
    """

    # Handle dense matrix multiplication when appropriate. When the sparsity
    # of the network is below about 90% it is faster to go ahead and
    # use dense matrix multiplication for this operation.



    # Handle broadcasting first. Because we will be doing quite a bit of mask
    # indexing trickery, we need both tensors to be exactly the same shape before
    # beginning the matrix multiplication process. This is required when mask
    # indexing.

    assert tensor_a.dim() >= 2
    assert tensor_b.dim() >= 2
    assert mask.dtype == torch.bool

    tensor_a = tensor_a.unsqueeze(-1)
    tensor_b = tensor_b.unsqueeze(-3)

    required_rank = max(tensor_a.dim(), tensor_b.dim(), mask.dim())
    while tensor_a.dim() < required_rank:
        tensor_a = tensor_a.unsqueeze(0)
    while tensor_b.dim() < required_rank:
        tensor_b = tensor_b.unsqueeze(0)
    while mask.dim() < required_rank:
        mask = mask.unsqueeze(0)

    broadcast_shape: List[int] = []
    for i, dims in enumerate(zip(tensor_a.shape, tensor_b.shape, mask.shape)):

        max_length = max(dims)
        for dim in dims:
            #Verify the tensors are all compatible
            if dim != 1 and dim != max_length:
                #Throw error.
                reason = f"""\
                         It is the case that the mask and two tensors are not
                         compatible. Once setup for expansion, tensor_a had
                         shape {tensor_a.shape}, tensor_b had shape {tensor_b.shape}
                         and the mask had shape {mask.shape}. These do not broadcast
                         together at dimension {i}.
                         """
                reason = StringUtil.dedent(reason)
                raise Errors.ValidationError("MaskedMatmulError", reason)
        broadcast_shape.append(max_length)

    tensor_a = torch.broadcast_to(tensor_b, broadcast_shape)
    tensor_b = torch.broadcast_to(tensor_a, broadcast_shape)
    mask = torch.broadcast_to(mask, broadcast_shape)

    # Perform the actual matrix multiplication. This consists of developing
    # the output buffer, performing the selection, performing the matrix multiplication
    # one element at a time, and scattering the result into the output buffer

    output = torch.zeros(tensor_a.shape, dtype = tensor_a.dtype, device = tensor_a.device)

    index = mask.nonzero().unbind(-1)
    active_tensor_a_elements = tensor_a[index]
    active_tensor_b_elements = tensor_b[index]
    unsummed_matmul = active_tensor_a_elements*active_tensor_b_elements
    output[index] = unsummed_matmul
    output = output.sum(dim=-2)
    return output

test_func = masked_matrix_multiplication #torch.jit.script(masked_matrix_multiplication)
