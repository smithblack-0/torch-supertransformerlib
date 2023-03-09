"""
A collection of tests for random access attention memory

"""
from typing import Tuple
import unittest
import seaborn as sns

import torch
import numpy as np
from torch import nn
from src.supertransformerlib import Attention


class TestMemAlloc(unittest.TestCase)

class TestMemSetAddress(unittest.TestCase):
    """
    A test fixture for the set address case.
    """
    def test_basic_randdata(self):
        """ Test that some random tensor flows through okay"""

        addresses = torch.randn([10, 6, 20, 5])
        values = torch.randn([10, 6, 20, 5])
        memory = Attention.MemoryTensor(addresses, values)


        update_addr = torch.randn([10, 35, 30])
        update_content = torch.randn([10, 35, 30])


        layer = Attention.MemSetAddresses(30, 6, 5, 20)
        layer = torch.jit.script(layer)

        updated_memory = layer(memory, update_addr, update_content)

    def test_parallel_randdata(self):
        """ Test that this works when used in ensemble mode"""
        addresses = torch.randn([10,7, 6, 20, 5])
        values = torch.randn([10, 7, 6, 20, 5])
        memory = Attention.MemoryTensor(addresses, values)


        update_addr = torch.randn([10, 7, 35, 30])
        update_content = torch.randn([10, 7, 35, 30])


        layer = Attention.MemSetAddresses(30, 6, 5, 20, 7)
        layer = torch.jit.script(layer)

        updated_memory = layer(memory, update_addr, update_content)

class TestMemSetContent(unittest.TestCase):
    """
    Test that the set content method is somewhat sane
    """
    def test_basic(self):
        """Test the layer works at all"""
        addresses = torch.randn([10, 6, 20, 5])
        values = torch.randn([10, 6, 20, 5])
        memory = Attention.MemoryTensor(addresses, values)

        update_addr = torch.randn([10, 35, 30])
        update_content = torch.randn([10, 35, 30])

        layer = Attention.MemSetContent(30, 6, 5, 20)
        layer = torch.jit.script(layer)

        updated_memory = layer(memory, update_addr, update_content)

    def test_parallel(self):
        """ Test the layer works decently when dealing in """
        addresses = torch.randn([10,7, 6, 20, 5])
        values = torch.randn([10, 7, 6, 20, 5])
        memory = Attention.MemoryTensor(addresses, values)


        update_addr = torch.randn([10, 7, 35, 30])
        update_content = torch.randn([10, 7, 35, 30])


        layer = Attention.MemSetContent(30, 6, 5, 20, 7)
        layer = torch.jit.script(layer)

        updated_memory = layer(memory, update_addr, update_content)


class TestMemGetContent(unittest.TestCase):
    """
    Test the ability of the memory system to get content from the memory backend
    """
    def test_basic(self):
        """Test the layer works at all"""
        addresses = torch.randn([10, 6, 20, 5])
        values = torch.randn([10, 6, 20, 5])
        memory = Attention.MemoryTensor(addresses, values)

        fetch_addr = torch.randn([10, 35, 30])

        layer = Attention.MemGetContent(30, 6, 5)
        layer = torch.jit.script(layer)

        fetched_entity = layer(memory, fetch_addr)
    def test_parallel(self):
        """ Test the layer works decently when dealing in """

        addresses = torch.randn([10, 7, 6, 20, 5])
        values = torch.randn([10,7, 6, 20, 5])
        memory = Attention.MemoryTensor(addresses, values)

        fetch_addr = torch.randn([10, 7, 35, 30])

        layer = Attention.MemGetContent(30, 6, 5, 7)
        #layer = torch.jit.script(layer)

        fetched_entity = layer(memory, fetch_addr)

