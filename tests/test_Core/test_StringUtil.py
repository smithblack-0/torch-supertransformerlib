"""

Test feature for any string manipulation
functions which I develop
"""

import unittest
import textwrap
from src.supertransformerlib import Core

class test_functions(unittest.TestCase):
    def test_dedent(self):
        """
        Test dedent, which is used for generating
        error messages. It must be torchscript compatible
        """

        message = """\
        This is a test
        
        It has extra space due to being defined inline. 
        These need to be removed by dedent.
        """

        expected = textwrap.dedent(message)
        gotten = Core.dedent(message)
        equivalent = expected == gotten
        self.assertTrue(equivalent, "textwrap dedent and torchscript dedent did not match.")

    def test_format(self):
        """
        Test that the torchscript formatting method works as required
        """
        replace = "potato"
        replace2 = "item"
        string_to_format = "  blah blah {replace} blah {replace2}   "

        expected = string_to_format.format(
            replace=replace,
            replace2=replace2
        )
        received = Core.format(string_to_format,
                                                                  {
                                                                   "replace": replace,
                                                                   "replace2": replace2
                                                               }
                                                                  )
        self.assertTrue(expected == received)
