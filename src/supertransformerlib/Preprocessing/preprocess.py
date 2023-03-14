import ast
from typing import Type
from flattening import flatten_class_inheritance
from string_scripting import exec_scriptable_string

def preprocess_dataclass(obj: Type)->Type:
    """
    A decorator designed to go around dataclasses,
    this is designed to allow dataclasses to
    be built with inheritance but still be scriptable.
    :param obj: The object to preprocess
    :return: The scriptable result
    """
    flattened_ast = flatten_class_inheritance(obj)
    source_code = ast.unparse(flattened_ast)
    new_obj = exec_scriptable_string(source_code, obj.__name__)
    return new_obj