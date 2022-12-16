from typing import Union, List, Optional

import torch
import src.supertransformerlib.Core.StringUtil as StringUtil
import src.supertransformerlib.Core.Errors as Errors

StandardShapeType = Union[torch.Tensor, List[int], int]
def standardize_shape(input: StandardShapeType,
                      input_name: str,
                      allow_negatives: bool = False,
                      allow_zeros: bool = False,
                      task: Optional[str] = None,
                      ) -> torch.Tensor:
    """
    Converts a sequence of things representing the dynamic_shape
    of a tensor in one of several formats into a single
    standard format of a 1D tensor. Performs some
    validation as well


    :param input: One of the possible input formats
    :param input_name: The name of the thing being standardized. Used to generate helpful error messages
    :param allow_negatives: Whether negative elements are allowed in the tensor dynamic_shape
    :param allow_zeros: Whether zero elements are allowed in the tensor dynamic_shape
    :param task: The task trace, used to make nice error messages.
    :return: A 1D tensor consisting of a valid definition of a tensor's dynamic_shape.
    """
    #Turn options into tensors
    if isinstance(input, int):
        output = torch.tensor([input], dtype=torch.int64)
    elif isinstance(input, list):
        output = torch.tensor(input, dtype=torch.int64)
    elif isinstance(input, torch.Tensor):

        if input.dim() != 1:
            dims = input.dim()
            reason = f"""\
            Expected parameter '{input_name}' to receive 1d tensor representing
            dynamic_shape. Number of dimensions was actually {dims}
            """
            reason = StringUtil.dedent(reason)
            raise Errors.StandardizationError(reason, task)
        if torch.is_floating_point(input):
            tensor_type = input.dtype
            reason = f"""\
            Expected parameter '{input_name}' to receive an integer tensor type. 
            However, actually a floating point tensor of type {tensor_type} 
            """
            reason = StringUtil.dedent(reason)
            raise Errors.StandardizationError(reason, task)
        if torch.is_complex(input):
            tensor_type = input.dtype
            reason = f"""\
            Expected parameter '{input_name}' to receive an integer of an integer tensor
            type. However, actually recieved a complex type of {tensor_type}
            """
            reason = StringUtil.dedent(reason)
            raise Errors.StandardizationError(reason, task)
        output = input
    else:
        input_type = type(input)
        reason = f"""\
        Expected parameter '{input_name}' to be one of 
        int, List[int], or torch.Tensor. But instead found {input_type}
        """
        reason = StringUtil.dedent(reason)
        raise Errors.StandardizationError(reason, task)
    if not allow_negatives and torch.any(output < 0):
        reason = f"""\
        Expected parameter '{input_name}' to consist of no elements
        less than zero. This was not satisfied. 
        """
        reason = StringUtil.dedent(reason)
        raise Errors.StandardizationError(reason, task)
    if not allow_zeros and torch.any(output == 0):
        reason = f"""\
        Expected parameter '{input_name}' to consist of no elements
        equal to zero. This was not satisfied
        """
        reason = StringUtil.dedent(reason)
        raise Errors.StandardizationError(reason, task)

    return output

torch.jit.script(standardize_shape)


def validate_shape_tensor(shape_tensor: torch.Tensor):
    """
    Validates that a given tensor represents a valid dynamic_shape
    :param input: The dynamic_shape tensor
    :raises: ValueError, if the dynamic_shape tensor is invalid
    """
    if not isinstance(shape_tensor, torch.Tensor):
        raise ValueError("Shape tensor was not tensor")
    if torch.is_floating_point(shape_tensor):
        raise ValueError("Expected dynamic_shape to be an int type. Got %s" % shape_tensor.dtype )
    if shape_tensor.dim() != 1:
        raise ValueError("dynamic_shape was not 1d")
    if torch.any(shape_tensor < 0):
        raise ValueError("dynamic_shape had negative dimensions, which are not allowed")


def validate_string_in_options(string: str,
                               stringname: str,
                               options: List[str],
                               options_name: str):
    """
    Validates that a string belongs to the
    given list. Raises useful error messages

    :param string: The string to test validity of
    :param stringname: What to call the string in error messages.
    :param options: The options the string could belong to
    :param options_name: What to call the options in error messages.

    :raises: ValueError if string not str
    :raises: ValueError, if string not in options mode
    """

    if not isinstance(string, str):
        msg = "The provided %s was not a string" % stringname
        raise ValueError(msg)
    if string not in options:
        msg = "The provide %s was '%s' \n" % (stringname, string)
        msg += "However, it could only be among %s  '%s'" % (options_name, str(options))
        raise ValueError(msg)


def get_shape_as_list(shape: torch.Tensor)->List[int]:
    """
    Small function to turn shape from tensor into list format.
    Useful for torchscript, as torch.Size won't take a tensor
    under torchscript.
    """
    output: List[int] = shape.tolist()
    return output

torch.jit.script(get_shape_as_list)

def get_strides(tensor: torch.Tensor)->torch.Tensor:
    """
    Goes through a tensor and fetches the strides
    in a manner torchscript is happy with.
    """
    strides = [tensor.stride(i) for i in range(tensor.dim())]
    output = torch.tensor(strides, dtype=torch.int64, device=tensor.device)
    return output


def get_shape(tensor: torch.Tensor)->torch.Tensor:
    """
    Goes and fetches the shape of a tensor as
    a 1D tensor. Happily ensures all data type
    and shape requirements are met
    """
    shape = tensor.shape
    output = torch.tensor(shape, dtype=torch.int64, device=tensor.device)
    return output
