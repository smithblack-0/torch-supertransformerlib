import ast
import inspect
import unittest

import torch.nn

from supertransformerlib.Preprocessing import flattening


class test_flatten_inheritance(unittest.TestCase):

    def setUp(self):
        class A:
            field_A: int = 3

            def method1(self):
                return "A.method1"

            def method2(self):
                return "A.method2"

            def method5(self):
                return "A.method5"
            def __init__(self):
                super().__init__()

        class B:
            field_B = "test"

            def method1(self):
                return "B.method1"

            def method3(self):
                return "B.method3"
            def __init__(self):
                super().__init__()

        class C(A):
            # Testing a comment
            @property
            def redirect(self) -> int:
                return self.field_A

            @redirect.setter
            def redirect(self, item: int):
                self.field_A = item

            # testing more
            def method1(self):
                return "C.method1"

            def method4(self):
                return "C.method4"

            def __init__(self):
                super().__init__()

        self.A = A
        self.B = B
        self.C = C

    def get_instance_members(self, instance):
        members = inspect.getmembers(instance)
        fields = [f[0] for f in members if
                  not inspect.ismethod(f[1]) and
                  not inspect.ismethoddescriptor(f[1]) and
                  not f[0].startswith('__') and
                  not inspect.isfunction(f[1]) and
                  not inspect.isbuiltin(f[1])]
        methods = [m[0] for m in members if
                   not m[0].startswith('__') and
                   not inspect.ismethoddescriptor(m[1]) and
                   not inspect.isbuiltin(m[1]) and inspect.isroutine(m[1])]
        properties = [p[0] for p in members if not p[0].startswith('__') and isinstance(p[1], property)]
        return fields, methods, properties

    def test_flatten_class_inheritance(self):
        """
        Test that the flatten class inheritance method will correctly recombine
        an inheritance chain
        """

        # Setup test classe

        class D(self.C, self.B):
            """
            test
            """
            def method1(self):
                return "D.method1"
            def method2(self):
                return "D.method2"

            def method3(self):
                return "D.method3"

            def method4(self):
                return super().method1()

        # Flatten class heirarchy. No docstring involved.
        class_ast = flattening.flatten_class_inheritance(D)
        class_code = ast.unparse(class_ast)
        print(class_code)

        # Evaluate flattened code
        namespace = {}
        exec(class_code, namespace)
        new_type = namespace["D"]

        # Check that all methods behave exactly the same
        master_instance = D()
        new_instance = new_type()

        fields, methods, properties = self.get_instance_members(master_instance)

        for method in methods:
            original_result = getattr(master_instance, method)()
            new_result = getattr(new_instance, method)()
            self.assertTrue(original_result == new_result, (method, original_result, new_result))

        for field in fields:
            original_result = getattr(master_instance, field)
            new_result = getattr(new_instance, field)
            self.assertTrue(original_result == new_result)

        for property in properties:
            original_result = getattr(master_instance, property)
            new_result = getattr(new_instance, property)
            self.assertTrue(original_result == new_result)