import inspect
import ast
import textwrap
import copy
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

def is_super_node(node: ast.AST)->bool:
    """
    Detect if this node is a representitive for a
    super node
    """
    return isinstance(node, ast.Call) and\
                    isinstance(node.func, ast.Attribute) and\
                    isinstance(node.func.value, ast.Call) and\
                    isinstance(node.func.value.func, ast.Name) and\
                    node.func.value.func.id == "super"
def is_method_in_classdef(classdef_node: ast.ClassDef, method_name: str) -> bool:
    """
    Given a ClassDef node and a method name, checks if the method is defined in the class.

    Args:
        classdef_node (ast.ClassDef): The ClassDef node to search for the method.
        method_name (str): The name of the method to search for.

    Returns:
        bool: True if the method is defined in the class, False otherwise.
    """
    for node in classdef_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return True
    return False

def get_getter_name(node: ast.FunctionDef)-> str:
    return node.name

def get_setter_name(node: ast.FunctionDef) -> str:
    return node.name

def get_deleter_name(node: ast.FunctionDef)-> str:
    return node.name

def get_method_name(node: ast.FunctionDef) -> str:
    return node.name
def get_self_name_from_method(method: ast.FunctionDef) -> str:
    self_arg = method.args.args[0]
    return self_arg.arg
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

def get_super_node_info(node: ast.Call)->Tuple[str, ast.Call]:
    """
    Returns the information about the node name, and node call.
    :param node:
    :return: A tuple consisting of the name of the method, the call node,
    and the super node
    """
    method_name = node.func.attr
    call_node = node
    info = (method_name, call_node)
    return info

def resolve_method(current_class: str,
                   method_name: str,
                   inheritance_ast: Dict[str, ast.ClassDef])->\
                        Tuple[Optional[str], Optional[ast.FunctionDef]]:
    """
    Resoves a method name that is part of a class hierarchy into it's
    source code. Returns the name of the class which this resolves to,
    alongside the functiondef node containing the class code. If it
    is resolving to a builtin, returns None.

    :param current_class: The name of the current parent class
    :param method_name: The name of the method to resolve
    :param inheritance_ast: The representation of the mro as ast.
    :return: A tuple of the class we landed on, and the
             and the functdef node we tracked down. Or None, None
             if we resolve to builtins
    :raises: Value_Error, if we cannot resolve the super call
    """
    mro_position = list(inheritance_ast.keys()).index(current_class)
    if mro_position + 1 == len(inheritance_ast):
        # We are at the top of the inheritance hierarchy. Either
        # resolve this as builtin, or throw an error
        if method_name in dir(object):
            return None, None
        else:
            raise ValueError(f"Could not resolve super method of name {method_name} from class {current_class}")

    for i in range(mro_position + 1, len(inheritance_ast)):
        name = list(inheritance_ast.keys())[i]
        class_ast = list(inheritance_ast.values())[i]
        for node in class_ast.body:
            if isinstance(node, ast.FunctionDef) and get_method_name(node) == method_name:
                return name, node
    return None, None


def rename_super_call_method(node: ast.Call, new_name: str):
    if isinstance(node.func, ast.Attribute):
        node.func.attr = new_name
        return node
    raise ValueError("Was not handled valid call node")


def replace_call_with_self(node: ast.Call, self_name: str):
    """ Replaces a super call with a self call"""
    node.func.value = ast.Name(id = self_name)


def replace_super_calls(node: ast.FunctionDef,
                        current_class: str,
                        inheritance_ast: Dict[str, ast.ClassDef],
                        known_features: List[str])->Tuple[ast.FunctionDef, Dict[str, ast.FunctionDef]]:
    """
    Finds and replaces super calls relevant to the current
    ast node with helper function calls.

    Returns first the new node with the new calls, and
    second a tuple of the required addition helper functions
    needed to allow the node to properly resolve.

    Recursive.
    """
    # This handles super replacements. Basically,
    # a super call will start a subroutine that
    # looks at the super feature being pointed to
    # and will build additional helper functions if
    # needed. The super calls are then renamed to
    # point to helper functions on self.
    #
    # If a helper is already found in known_features,
    # it is assumed that function will already be available
    # for usage.

    output = {}
    node = copy.deepcopy(node)
    self_name = get_self_name_from_method(node)

    #Walk through the nodes inside the functiondef
    for subnode in ast.walk(node):

        # Act if super node is detected
        if is_super_node(subnode):
            # Fetch the name and node
            method_name, call_node = get_super_node_info(subnode)
            parent_name, parent_functdef = resolve_method(current_class,
                                                          method_name,
                                                          inheritance_ast)
            if parent_name is None:
                # Resolves to builtin. No need for any action
                continue

            alternate_method_name = f"__super_{parent_name}_{method_name}"
            if alternate_method_name not in known_features and \
                    alternate_method_name not in output:
                # A helper method is needed but has not yet been
                # built. We build it recursively by fetching the
                # required updates for a replacement, then
                # including it into an update.

                update_node, updates = replace_super_calls(
                            parent_functdef,
                            parent_name,
                            inheritance_ast,
                            known_features
                        )
                update_node.name = alternate_method_name
                updates[alternate_method_name] = update_node
                output.update(updates)

            #Tell the call node to refer to the helper function instead
            rename_super_call_method(call_node, alternate_method_name)
            replace_call_with_self(call_node, self_name)
    return node, output





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
    inheritance_ast = [get_ast_from_object(item) for item in inheritance_chain]
    inheritance_ast = {ast_case.name : ast_case for ast_case in inheritance_ast}

    base_node = get_ast_from_object(inheritance_chain[0])
    docstring_node = get_class_docstring_node(base_node)
    construction_stack: Dict[str, List[ast.Ast]] = {name : [] for name in inheritance_ast.keys()}

    # Something that ends up in one of these has already been
    # encountered before and thus should not be included in the
    # final output again.
    known_features = []
    known_setters = []
    known_deleters = []

    for i, (class_name, class_ast) in enumerate(inheritance_ast.items()):

        if i == 0:
            new_body = [make_comment_node(f"Original code from given class named {class_ast.name}")]
        else:
            new_body = [make_comment_node(f"Code from class of name {class_ast.name}")]

        for node in class_ast.body:
            # The primary challenge and purpose of the
            # following code is to only transfer
            # features which have not been seen
            # before in the MRO chain.\

            # A secondary challenge which can occur is that
            # super nodes must be properly handled, which
            # will mean replacing the super node with a
            # private function call.

            if isinstance(node, ast.FunctionDef):

                # This handles super replacements.
                node, helper_code = replace_super_calls(node,
                                                  class_name,
                                                  inheritance_ast,
                                                  known_features)
                for helper_name, helper_ast in helper_code.items():
                    new_body.append(helper_ast)
                    known_features.append(helper_name)


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
            construction_stack[class_name].extend(new_body)

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


    for name, body in construction_stack.items():
        new_class.body.extend(body)

    return new_class

