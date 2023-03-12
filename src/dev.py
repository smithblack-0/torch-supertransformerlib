import inspect
import ast
import os


def flatten_class_methods(source_path, class_name):
    # Load the source code from the file
    with open(source_path, 'r') as f:
        source_code = f.read()

    # Parse the source code into an Abstract Syntax Tree (AST)
    tree = ast.parse(source_code)

    # Find the class definition for the given class name
    class_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            class_def = node
            break

    if class_def is None:
        raise ValueError(f"No class definition found for class name {class_name}")

    # Iterate over the superclasses and move their methods to the class definition
    for base in class_def.bases:
        base_name = base.id
        base_path = os.path.join(os.path.dirname(source_path), base_name + '.py')
        if not os.path.exists(base_path):
            continue

        # Load the source code for the base class
        with open(base_path, 'r') as f:
            base_source = f.read()

        # Parse the source code for the base class
        base_tree = ast.parse(base_source)

        # Find the class definition for the base class
        base_class_def = None
        for node in ast.walk(base_tree):
            if isinstance(node, ast.ClassDef) and node.name == base_name:
                base_class_def = node
                break

        if base_class_def is None:
            continue

        # Move all methods from the base class to the target class
        for method in base_class_def.body:
            if isinstance(method, ast.FunctionDef):
                class_def.body.append(method)

        # Move the __init__ method from the base class to the target class
        for method in base_class_def.body:
            if isinstance(method, ast.FunctionDef) and method.name == '__init__':
                class_def.body.insert(0, method)

    # Generate the updated source code with the flattened class
    updated_source_lines = []
    for node in tree.body:
        if node == class_def:
            updated_source_lines.extend([ast.unparse(method_node) for method_node in node.body])
        else:
            updated_source_lines.append(ast.unparse(node))

    updated_source_code = '\n'.join(updated_source_lines)

    # Return the updated source code
    return updated_source_code

source_path = 'source.py'
class_name = 'Dog'
new_source_code = flatten_class_methods(source_path, class_name)
print(new_source_code)
