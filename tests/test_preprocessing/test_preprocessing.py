
import unittest
import torch
from src.supertransformerlib.Preprocessing import preprocess

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

    def test_sequentiential_script(self):
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

        # Attempt to script
        D = preprocess.preprocess_dataclass(D)
        torch.jit.script(D)

    def test_decorator_script(self):
        """ Attempt to setup for scripting directly as decorator"""

        @preprocess.preprocess_dataclass
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

        torch.jit.script(D)