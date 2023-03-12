"""
A place for manipulations on collections of memory tensors
to lie. Contains a significant portion of prebuilt cases,
alongside with the mechanisms needed to build your own.
"""


from typing import Dict, List, Tuple, Type, Callable, Union
import torch
import typing
import inspect
import textwrap

def make_function_private(source: str) -> str:
    """
    This function takes a source code string for a function and returns a modified
    string with the function name made private by prefixing it with an underscore.
    For example, given "def foo(whatever...)...", it returns "def _foo(whatever...)...".

    :param source: The source code string for the function.
    :return: The modified source code string with the function name made private.
    """
    # Find the index of the first occurrence of "def" to get the start of the function name
    def_start = source.find("def")
    # Find the index of the first open parenthesis to get the end of the function name
    name_end = source.find("(")
    # Extract the function name
    function_name = source[def_start + 4: name_end].strip()
    # Prepend an underscore to the function name
    private_function_name = "_" + function_name
    # Replace the original function name with the private function name in the source code
    return source.replace(f"def {function_name}", f"def {private_function_name}")

def make_function_a_method(source: str) -> str:
    """
    This function takes a source code string for a function and returns a modified
    string with the function made a method by indenting the entire function body by 4 spaces.

    :param source: The source code string for the function.
    :return: The modified source code string with the function made a method.
    """
    # Find the index of the first occurrence of "def" to get the start of the function name
    def_start = source.find("def")
    # Find the index of the first open parenthesis to get the end of the function name
    name_end = source.find("(")
    # Extract the function name
    function_name = source[def_start + 4: name_end].strip()
    # Prepend "self" to the function arguments
    method_args = source[name_end + 1: source.find(")")]
    if "->" in method_args:
        method_args = method_args[:method_args.find("->")].strip()
    new_args = "self, " + method_args if method_args else "self"
    # Indent the entire function body by 4 spaces
    function_body = source[source.find(":", source.find(")")) + 1:].strip()
    indented_body = "\n".join([" " * 4 + line for line in function_body.split("\n")])
    # Combine the modified function name, arguments, and body into a new method string
    method_str = f"def {function_name}({new_args}):\n{indented_body}"
    method_str = textwrap.indent(method_str, " "*4)
    return method_str


def get_function_params(func: callable, type_hints: bool = False) -> List[str]:
    """
    Given a live function object, returns a list of strings representing
    the parameter names and (optionally) their type hints for the function.

    :param func: A live function object.
    :param type_hints: A boolean flag indicating whether to include type hints in the result.
    :return: A list of strings representing the parameter names (and optionally their type hints).
    """
    sig = inspect.signature(func)
    params = []
    for name, param in sig.parameters.items():
        if type_hints:
            if param.annotation is inspect.Parameter.empty:
                params.append('{}: {}'.format(name, 'Any'))
            else:
                params.append('{}: {}'.format(name, param.annotation.__name__))
        else:
            params.append(name)
    return params


def generate_parallel_container(class_name: str,
                                info: List[Tuple[str, Type]],
                                fields: List[str],
                                transformations: Dict[str, Callable]) -> str:
    """
    This function is designed to make a data storage class which torchscript
    will compile and which can posses methods that will operate cleanly across
    all the memory tensor such as batch level modifications. It will return a
    string representation of this class.

    The class will consist of field level type declarations, an init
    function which accepts the various parameters, and methods which were
    defind by the transformations section.

    One thing worth mentioning is that if transformations is causing trouble,
    consider using fully qualified type hints.


    :param class_name: The name of the class.
    :param info: A specification for fields which can be of any type. These will
                show up on the generated class with their type info. Defined in
                terms of a dictionary Dict[str, Type]. All of these are optional
    :param fields: A specification for the memory tensor fields to make. These will be
                   named, and of type MemoryTensor. The constructor requires each one
                   of these to function.
    :param transformations: The most complicated part of this.
           Transformations consists of transformations which we will wish to be able
           to apply in parallel to each of the memory tensors. The section is designed
           so that a method of the same name as string will be generated, which when called
           will activate the source code found in the collable against each field feature in
           sequence. The results from this call will then be used to create a new instance
           and feed into the "field" parameter of the constructor.
    :return: A string representing the class.
    """
    # Initialize empty lists
    fields_lst = []

    init_params_lst = []
    init_args_lst = []

    passthrough_list = []
    transform_list = []
    helper_methods = []
    main_methods = []

    # Iterate over the field and info sections of the input, collecting the
    # information needed to make the fields and constructors.
    for item in info:
        name, ftype = item
        fields_lst.append(f"{name}: {ftype.__name__}")
        init_params_lst.append(f"{name}: {ftype.__name__}, ")
        init_args_lst.append(f"self.{name} = {name}")
        passthrough_list.append(f'self.{name}')
    for field_name in fields:
        fields_lst.append(f"{field_name}: MemoryTensor")
        init_params_lst.append(f"{field_name}: MemoryTensor")
        init_args_lst.append(f"self.{field_name} = {field_name}")

    # Generate the code blocks for fields, parameters, and initialization methods.
    fields_code = "\n    " + "\n    ".join(fields_lst)
    param_code = ", ".join(init_params_lst)
    args_code = "\n        " + "\n        ".join(init_args_lst)

    # Generate the initialization code

    init_template = """\
    def __init__(self, {init_params}):
        {assignments}
    """
    init_template = textwrap.dedent(init_template)
    init_template = textwrap.indent(init_template, "    ")
    init_code = init_template.format(
        init_params = param_code,
        assignments = args_code
    )


    # Create the method code. This will include
    # the helper methods and the main methods

    ##  Build the method template ##

    method_template = """\
     def {method_name}(self, {method_parameters}):
         {transform_code}
         return {class_name}({constructer_params})
     """
    method_template = textwrap.dedent(method_template)
    method_template = textwrap.indent(method_template, "    ")

    ## Fill out the pieces and make the methods.
    for method_name, method_func in transformations.items():

        # Build the helper method, which is just a carbon copy of the
        # provided function as a method
        source_code = inspect.getsource(method_func).strip()
        source_code = make_function_private(source_code)
        source_code = make_function_a_method(source_code)
        if source_code not in helper_methods:
            helper_methods.append(source_code)

        # Figure out the correct values for the
        # method name, method parameters, transform code, etc.


        method_parameters =  get_function_params(method_func, type_hints=True)
        method_parameters  = ', '.join(method_parameters)

        method_passthrough = get_function_params(method_func, type_hints=False)
        method_passthrough = [f'new_{name}' for name in method_passthrough]
        method_passthrough = ', '.join(method_passthrough)

        transform_code = [f'new_{name} = self._{method_name}(self.{name}, {method_passthrough})' for
                          name in fields]
        transform_code = '\n        ' + '\n        '.join(transform_code)

        constructer_params = passthrough_list + [f'new_{name}' for name in fields]
        constructer_params = ', '.join(constructer_params)

        # Make the method

        method_code = method_template.format(
            method_name = method_name,
            method_parameters = method_parameters,
            transform_code = transform_code,
            class_name = class_name,
            constructer_params = constructer_params,
        )
        main_methods.append(method_code)

    helper_methods_code = '\n'.join(helper_methods)
    main_method_code = '\n'.join(main_methods)

    # Create the class by filling in a template

    class_template = """\
    class {class_name}:
        '''
        A metacoded class for handling synchronous memory tensors.
        Do not edit. Will be overwritten.
        '''
        # Fields
    {field_code}
    
        #Helper methods
    {helper_method_code}

    {init_code}
    
        # Transformations
    
    {method_code}
    """
    class_template = textwrap.dedent(class_template)
    class_code = class_template.format(
        class_name = class_name,
        field_code = fields_code,
        helper_method_code = helper_methods_code,
        init_code = init_code,
        method_code = main_method_code
    )
    return class_code

def get_import_statements_from_type(typ: type) -> List[str]:
    """
    This function takes a type as input and returns a list of import statements
    that would be needed to use that type.

    :param typ: The type to generate import statements for.
    :return: A list of import statements needed for the given type.
    """

    import_statements = []

    # Get the origin of the type, which is the base type of generic types (List, Dict, Tuple, Optional)
    origin = typing.get_origin(typ)
    if origin is None:
        # If the type is not a generic type, just add the import statement for the type itself
        if typ.__module__ == "builtins":
            return import_statements
        import_statements.append(f"from {typ.__module__} import {typ.__name__}")
    else:
        # If the type is a generic type, recursively find the import statements for the base type(s)
        args = typing.get_args(typ)
        for arg in args:
            import_statements.extend(get_import_statements_from_type(arg))

        # Add the import statement for the origin type
        if origin.__module__ != "builtins":
            import_statements.append(f"from {origin.__module__} import {origin.__name__}")

    # Remove duplicates and return the list of import statements
    return list(set(import_statements))


from typing import List, Tuple, Dict, Optional, Union, Any
from torch import nn
class MyClass:
    pass

import_statements = get_import_statements_from_type(Union[int, nn.Module, List[Tuple[MyClass, torch.Tensor, Optional[float]]]])
print(import_statements)

