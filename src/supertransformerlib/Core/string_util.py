"""

There are some useful python string libraries and actions
which are not torchscript compilable, as torchscript
demands annotations on functions.

This module is for torchscript compatible alternatives.

"""
from typing import List, Dict
import torch.jit


def dedent(string: str)->str:
    """
    A torchscript compatible dedent, since textwrap's dedent
    does not work properly. Takes and eliminates common among
    the beginning of each line. Required to prevent error messages
    from looking weird.

    Quick and dirty. That is okay. This is only utilized when
    raising error messages.

    Edge cases involving tab or other such nasties are not explicitly handled,
    beware

    :param string: The string to dedent
    :return: The dedented string
    """
    lines = string.split("\n")

    #Figure out how much whitespace to removed by looking through
    #all the lines and keeping the shortest amount of whitespace.
    #
    #Then shorten all dimensions by that amount
    has_viewed_a_line_flag = False
    amount_of_whitespace_to_remove = 0
    for line in lines:
        whitespace = len(line) - len(line.lstrip())
        if not has_viewed_a_line_flag:
            amount_of_whitespace_to_remove = whitespace
            has_viewed_a_line_flag = True
        else:
            amount_of_whitespace_to_remove = min(whitespace, amount_of_whitespace_to_remove)

    output: List[str] = []
    for line in lines:
        updated_line = line[amount_of_whitespace_to_remove:]
        output.append(updated_line)

    output = "\n".join(output)
    return output


torch.jit.script(dedent)


def format(string: str, substitutions: Dict[str, str])->str:
    """
    Performs a formatting action on a string in a torchscript
    compatible manner. Does not support positional substitutions
    or escape sequences.

    :param string: The string to perform substitutions on
    :param substitutions: The substitutions to perform as would be done with str.format(keywords)
    :return: The formatted string
    """
    for key, value in substitutions.items():
        key = "{" + key +"}"
        string = string.replace(key, value)
    return string


torch.jit.script(format)
