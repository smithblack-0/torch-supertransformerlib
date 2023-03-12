import inspect
import ast
import textwrap
from typing import Type, List, Dict, Optional, Set, Union, Tuple


def get_ast_from_object(obj: Type) -> ast.ClassDef:
    """
    Returns the AST class node for a class object.

    Args:
        obj (type): A class object.

    Returns:
        ast.ClassDef: The AST class node for the given class object.

    Raises:
        ValueError: If the class definition is not found in the AST tree.
    """
    source_lines, _ = inspect.getsourcelines(obj)
    source_code = ''.join(source_lines)
    source_code = textwrap.dedent(source_code) # If you have an inline class, it may need to be dedented.
    tree = ast.parse(source_code)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name == obj.__name__:
                return node

    raise ValueError(f"No class definition found for class {obj.__name__}")


def get_class_inheritance_chain(obj: Type) -> List[Type]:
    """
    Get the inheritance chain for a class.

    Args:
        cls (Type): The class object to get the inheritance chain for.

    Returns:
        List[Type]: A list of class objects representing the inheritance chain for the class.
    """
    inheritance_chain = []
    output = obj.mro()
    output = output[:-1] #we do not need the top level object
    return output

def get_class_docstring_node(class_node: ast.ClassDef) -> Optional[ast.Expr]:
    """
    Gets the AST node for the docstring of a class.

    Args:
        class_node (ast.ClassDef): The AST class node for the class.

    Returns:
        Optional[ast.Expr]: The AST node for the docstring, or None if no docstring is found.
    """
    for node in class_node.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
            return node
    return None


def make_comment_node(message: str, indent: int = 0) -> ast.Expr:
    """
    Returns an AST node for a comment with the given message.

    Args:
        message (str): The message to include in the comment.
        indent (int): The number of spaces to indent the comment. Default is 0.

    Returns:
        ast.Expr: An AST node representing the comment.
    """
    comment = ast.Expr(ast.Str(f"{' ' * indent}# {message}"))
    return comment


## Ast interface functions for looking up and checking crap.
def is_method(body_node: ast.AST) -> bool:
    """
    Checks if the given node is a method
    """
    return isinstance(body_node, ast.FunctionDef)

def is_property_getter(node: ast.AST) -> bool:
    """
    Checks if the given node is a getter for a property.
    """
    if isinstance(node, ast.FunctionDef) and node.decorator_list:
        decorator = node.decorator_list[0]
        if isinstance(decorator, ast.Name) and decorator.id == 'property':
            return True
    return False

def is_property_setter(node: ast.AST) -> bool:
    """
    Checks if the given node is a setter for a property.
    """
    if isinstance(node, ast.FunctionDef) and node.decorator_list:
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute) and decorator.attr == 'setter':
                return True
    return False

import ast

def is_property_deleter(node: ast.FunctionDef) -> bool:
    """
    Checks if the given node is a deleter method for a property.
    """
    if not isinstance(node, ast.FunctionDef):
        return False
    if len(node.decorator_list) == 0:
        return False
    decorator = node.decorator_list[0]
    if not isinstance(decorator, ast.Attribute):
        return False
    return decorator.attr == 'deleter'

def is_field(node: ast.AST):
    """
    Checks if the given node is a field. This could be an assign
    or an annotated assign.
    """
    return isinstance(node, (ast.AnnAssign, ast.Assign))

def is_docstring(node: ast.AST) -> bool:
    """
    Determines whether an AST node is a docstring.

    Args:
        node (ast.AST): The AST node to check.

    Returns:
        bool: True if the node is a docstring, False otherwise.
    """
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)

def get_getter_name(node: ast.FunctionDef)-> str:
    return node.name

def get_setter_name(node: ast.FunctionDef) -> str:
    return node.name

def get_deleter_name(node: ast.FunctionDef)-> str:
    return node.name

def get_method_name(node: ast.FunctionDef) -> str:
    return node.name

def get_field_name(node: Union[ast.Assign, ast.AnnAssign])->List[str]:
    # Notably, some assignments can assign more than one thing at once
    output = []
    if isinstance(node, ast.Assign):
        # Loop through the targets, fetching off of them.
        if len(node.targets) > 1:
            msg = """\
            Multiple assignments are not supported in class body. Instead of 
            doing x, y =  5, 10, break it up into two lines.
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
        elif isinstance(target, ast.Attribute):
            return target.attr

    elif isinstance(node, ast.AnnAssign):
        # Must have been an annotated assign
        return node.target.id
    else:
        raise ValueError("Illegal node given")




def flatten_class_inheritance(obj: Type)->ast.ClassDef:
    """
    Makes a revised version of a particular class which has had
    it's inheritance features removed and instead now has had
    any such methods moved by static analysis into a new, unique
    class.

    :param obj: A class object to flatten.
    :return: A ast node which has used static analysis to flatten the
             inheritance hierarchy
    """
    # We basically take apart the class heirarchy using ast, and create a stack of the proper
    # body imports in the method resolution chain. Then, we promote the docstring and the
    # decorators onto a class into a new class node, then fill out the body from the top parent
    # downward
    #
    # We do, however, make sure we only include a piece in the MRO chain if we have
    # not already seen it before.

    inheritance_chain = get_class_inheritance_chain(obj)
    base_node = get_ast_from_object(inheritance_chain[0])
    docstring_node = get_class_docstring_node(base_node)
    construction_stack: List[Tuple[str, List[ast.AST]]] = []

    known_features = []
    known_setters = []
    known_deleters = []
    for i, class_type in enumerate(inheritance_chain):
        class_ast = get_ast_from_object(class_type)

        if i == 0:
            new_body = [make_comment_node(f"Original code from given class named {class_ast.name}")]
        else:
            new_body = [make_comment_node(f"Code from class of name {class_ast.name}")]

        for node in class_ast.body:
            # The primary challenge and purpose of the
            # following code is to only transfer
            # features which have not been seen
            # before in the MRO chain.

            # Handle property getters
            if is_property_getter(node):
                # Handles property getters.
                name = get_getter_name(node)
                if name not in known_features:
                    new_body.append(node)
                    known_features.append(name)

            # Handles property setters
            elif is_property_setter(node):
                name = get_setter_name(node)
                if name not in known_setters:
                    new_body.append(node)
                    known_setters.append(name)

            # Handles property deleters
            elif is_property_deleter(node):
                name = get_deleter_name(node)
                if name not in known_deleters:
                    new_body.append(node)
                    known_deleters.append(name)

            # Handle field transfer
            elif is_field(node):
                name = get_field_name(node)
                if name not in known_features:
                    new_body.append(node)
                    known_features.append(name)

            # Handle method transfer.
            elif is_method(node):
                name = get_method_name(node)
                if name not in known_features:
                    new_body.append(node)
                    known_features.append(name)

            elif is_docstring(node):
                continue
            # Handle unknowns
            else:
                msg = """\
                Node of type {type_spec} is not compatible with
                the build engine. This unpacks to look like:
                
                {unpacked_code}
                """
                msg = textwrap.dedent(msg)
                msg = msg.format(type_spec = type(node),
                                 unpacked_code = ast.unparse(node))
                raise ValueError(msg)


        if len(new_body) > 1:
            construction_stack.append((class_ast.name, new_body))

    # Build the new class ast
    synthesis_message = """\
    APPENDED INFORMATION
    
    This class has been modified from it's original format. It
    has been modified to not use inheritance and to have no 
    parent classes. Keep in mind this may mean feature such 
    as isinstance and other bits and pieces of code may now 
    thus fail.
    """

    if docstring_node is None:
        docstring_node = ast.Expr(ast.Str(synthesis_message))
    else:
        new_docstring = f"{docstring_node.value.s}\n {synthesis_message}"
        docstring_node.value.s = new_docstring

    new_class = ast.ClassDef(
        name=base_node.name,
        body=[docstring_node],
        bases=[],
        keywords=[],
        decorator_list=base_node.decorator_list
        )


    for name, body in construction_stack:
        new_class.body.extend(body)

    return new_class
