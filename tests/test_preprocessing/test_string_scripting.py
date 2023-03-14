import unittest
import textwrap
import torch
from src.supertransformerlib.Preprocessing import string_scripting

fixture = 4
class test_string_scripting(unittest.TestCase):

    def test_get_context(self):
        item = 3
        globals_items, locals_items = string_scripting.get_context(1)
        print(globals_items.keys())
        print(locals_items.keys())

    def test_simple_string_scripting(self):

        basic_func = """\
        def test(item: int)->int:
            return item
        """
        basic_func = textwrap.dedent(basic_func)

        func = string_scripting.exec_scriptable_string(basic_func, "test")
        self.assertTrue(func(3) == 3)
        scripted_outcome = torch.jit.script(func)
        self.assertTrue(scripted_outcome(3) == 3)

