import unittest
from typing import Type

import torch
import torch.nn

import src.supertransformerlib.Core as Core
import src.supertransformerlib.Core.Addressbook
import src.supertransformerlib.Core.Functions
import src.supertransformerlib.Core.StringUtil


### Fixtures ###
#
# These must be located at the top level so pickle
# is happy


class test_AddressBook(unittest.TestCase):
    """

    Tests the address book. This is a memory manager built
    (ugh) in python which can be used to make pointers
    within an address space. The class is initialized
    with the space to address, which must be finite,
    and can then be passed nonnegative integers to associate
    with the address space.
    """

    def test_basics_series_constructor(self):
        """Test that the constructor works at all"""

        def test_should_pass(addresses: torch.Tensor):
            src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
        def test_should_fail(addresses: torch.Tensor, error: Type[Exception]):
            def to_fail():
                src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
            self.assertRaises(error, to_fail)

        
        #Should pass
        addresses_naive = torch.arange(120)
        addresses_shifted = torch.arange(120) + 30
        addresses_negative = torch.arange(120) - 10
        addresses_only_one = torch.tensor([0])

        #Should fail

        addresses_wrong_shape = torch.arange(1000).view(10, 10, 10)
        addresses_wrong_dtype = torch.randn(120)
        addresses_empty = torch.empty([0])

        #tests
        test_should_pass(addresses_naive)
        test_should_pass(addresses_shifted)
        test_should_pass(addresses_negative)
        test_should_pass(addresses_only_one)

        test_should_fail(addresses_wrong_shape, AssertionError)
        test_should_fail(addresses_wrong_dtype, AssertionError)
        test_should_fail(addresses_empty, AssertionError)

    def test_basics_series_malloc(self):
        """Test the memory allocator is working, ignoring any interactions with other methods."""

        addresses = torch.arange(120)
        def test_should_pass(pointer_ids: torch.Tensor):
            addressbook = src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
            addressbook.malloc(pointer_ids)

            addressbook = src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
            addressbook = torch.jit.script(addressbook)
            addressbook.malloc(pointer_ids)

        def test_should_pass_double(
                                    pointers1: torch.Tensor,
                                    pointers2: torch.Tensor):
            addressbook = src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
            addressbook.malloc(pointers1)
            addressbook.malloc(pointers2)

            addressbook = src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
            addressbook = torch.jit.script(addressbook)
            addressbook.malloc(pointers1)
            addressbook.malloc(pointers2)

        def test_should_fail(
                             pointer_ids: torch.Tensor,
                             error: Type[Exception]):

            def to_fail():
                addressbook = src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
                addressbook.malloc(pointer_ids)
            self.assertRaises(error, to_fail)

            def to_fail():
                addressbook = src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
                addressbook = torch.jit.script(addressbook)
                try:
                    addressbook.malloc(pointer_ids)
                except torch.jit.Error:
                    raise error

            self.assertRaises(error, to_fail)

        def test_should_fail_double(
                             pointers1: torch.Tensor,
                             pointers2: torch.Tensor,
                             error: Type[Exception]):
            def to_fail():
                addressbook = src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
                addressbook.malloc(pointers1)
                addressbook.malloc(pointers2)

            self.assertRaises(error, to_fail)

            def to_fail():
                addressbook = src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
                addressbook = torch.jit.script(addressbook)
                try:
                    addressbook.malloc(pointers1)
                    addressbook.malloc(pointers2)
                except torch.jit.Error:
                    raise error

            self.assertRaises(error, to_fail)

        #Should pass

        completely_full = torch.arange(120, dtype=torch.int64)
        shifted = torch.arange(120, dtype=torch.int64) + 1000
        partially_full = torch.arange(30, dtype=torch.int64)

        double_a_full, double_b_full = torch.arange(100, dtype=torch.int64), torch.arange(20, dtype=torch.int64) + 100
        partial_a_full, partial_b_full = torch.arange(20, dtype=torch.int64), torch.arange(20, dtype=torch.int64) + 20

        #Should fail

        overfull = torch.arange(130, dtype=torch.int64)
        double_overfull_a, double_overfull_b = torch.arange(20, dtype=torch.int64), torch.arange(130, dtype=torch.int64) + 20
        attempt_reassign_a, attempt_reassign_b = torch.arange(120, dtype=torch.int64), torch.arange(120, dtype=torch.int64) #Notably, should warn.
        wrong_dtype = torch.arange(120, dtype=torch.int16)

        #Tests

        test_should_pass(completely_full)
        test_should_pass(shifted)
        test_should_pass(partially_full)

        test_should_pass_double(double_a_full, double_b_full)
        test_should_pass_double(partial_a_full, partial_b_full)

        test_should_fail(overfull, RuntimeError)
        test_should_fail(wrong_dtype, AssertionError)
        test_should_fail_double(double_overfull_a, double_overfull_b, RuntimeError)
        test_should_fail_double(attempt_reassign_a, attempt_reassign_b, RuntimeError)

    def test_basic_series_dereference(self):
        """Test that dereference is working properly"""

        addresses = torch.arange(120)
        malloc_pointers = torch.arange(40)

        def test_should_succeed(pointers: torch.Tensor, expected: torch.Tensor):

            #Standard
            addressbook = src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
            addressbook.malloc(malloc_pointers)
            output = addressbook.dereference(pointers)
            self.assertTrue(torch.all(expected == output))

            #Torchscript
            addressbook = src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
            addressbook = torch.jit.script(addressbook)
            addressbook.malloc(malloc_pointers)
            output = addressbook.dereference(pointers)
            self.assertTrue(torch.all(expected == output))

        def test_should_fail(pointers: torch.Tensor, error: Type[Exception]):

            #Standard

            def to_fail():
                addressbook = src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
                addressbook.malloc(malloc_pointers)
                output = addressbook.dereference(pointers)

            self.assertRaises(error, to_fail)

            #Torchscript
            def to_fail():
                addressbook = src.supertransformerlib.Core.Addressbook.AddressSpace(addresses)
                addressbook = torch.jit.script(addressbook)
                try:
                    addressbook.malloc(malloc_pointers)
                    output = addressbook.dereference(pointers)
                except torch.jit.Error:
                    raise error("mockup", torch.tensor([3]))

            self.assertRaises(error, to_fail)

        #Should pass

        partial, partial_expected = torch.arange(20), torch.arange(20)
        full, full_expected = torch.arange(40), torch.arange(40)
        out_of_order, out_of_order_expected = torch.arange(20).flip(dims=[-1]), torch.arange(20).flip(dims=[-1])
        nd, nd_expected = torch.arange(40).view(5, 2, 4), torch.arange(40).view(5,2,4)

        #Should fail

        null_ptr = torch.arange(40) + 1
        invalid_ptr = torch.arange(40) - 1
        bad_dtype = torch.ones([10], dtype=torch.int32)

        #Tests

        test_should_succeed(partial, partial_expected)
        test_should_succeed(full, full_expected)
        test_should_succeed(out_of_order, out_of_order_expected)
        test_should_succeed(nd, nd_expected)

        test_should_fail(null_ptr, src.supertransformerlib.Core.Addressbook.NullPtr)
        test_should_fail(invalid_ptr, AssertionError)
        test_should_fail(bad_dtype, AssertionError)
