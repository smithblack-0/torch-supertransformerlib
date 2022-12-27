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

    def test_replacement_game(self):
        """
        Test that the parameters are working sanely by making the layer
        play a little game. The layer should learn, when it sees an address
        again, to replace that address with the associated content until
        the address pops up once again.
        """
        # Set game parameters
        batch_size = 1
        buffer_size = 40
        encodings = 128
        updates_per_round = 5
        loss_threshold = 0.001
        max_rounds = 50000

        # Setup helper functions

        norm = nn.LayerNorm(encodings)


        def create_instruction(memory: Attention.MemoryTensor,
                               index: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Takes in the memory buffer and the index. Generates both a
            current key to hunt down, along with what the replacement should
            be. Returns these.

            This is sort of the "do this" part.
            """

            current = memory.Addresses[:, index]
            with torch.no_grad():
                replacement = torch.randn_like(current)
                replacement = norm(replacement)
            instruction = (current, replacement)
            return instruction

        def create_expected_return(memory: Attention.MemoryTensor,
                                   index: torch.Tensor,
                                   replacement: torch.Tensor
                                   ) -> Attention.MemoryTensor:
            """
            Using the instruction and hard logic, we develop the expected
            output. It should be the case that the given addr index is replaced with
            the given replacement.
            """
            output = memory.Addresses.clone().detach()
            output[:, index] = replacement
            output = Attention.MemoryTensor(output, memory.Contents)
            return output




        # Set game tensors.

        addr_index = np.arange(0, buffer_size)
        mem_addr = torch.randn([batch_size, buffer_size, encodings])
        memory = Attention.MemoryTensor(mem_addr, None)

        # Set game layers losses and optimizers
        layer = Attention.MemSetAddresses(encodings, 1, encodings, buffer_size)
        optim = torch.optim.Adam(params = layer.parameters())

        CLEAN_BREAK = False
        for i in range(max_rounds):


            indices_to_sample = np.random.choice(addr_index, updates_per_round, replace=False)
            addr_key, replacement_addr = create_instruction(memory, indices_to_sample)


            expected_result = create_expected_return(memory, indices_to_sample, replacement_addr)
            actual_result = layer(memory, addr_key, replacement_addr)
            diff = (expected_result.Addresses - actual_result.Addresses).abs().sum(dim=-1)

            mask = torch.full([batch_size, buffer_size], False, dtype=torch.bool)
            mask[:, indices_to_sample] = True

            replaced_loss = diff[:, mask].mean()
            retained_loss = diff[:, torch.logical_not(mask)].mean()
            total_loss = (replaced_loss + retained_loss)

            loss = total_loss
            print(total_loss, replaced_loss, retained_loss)
            if loss < loss_threshold:
                CLEAN_BREAK = True
                break

            loss.backward()
            optim.step()
            memory = expected_result
            optim.zero_grad()


        if not CLEAN_BREAK:
            raise RuntimeError("Did not train properly")




class TestSetMemory(unittest.TestCase):
    """
    A test fixture for the set memory case.

    This section assumes that head breakdown
    has already occurred somewhere along the line.
    """
    def test_basic_randdata(self):

        batch = 30
        items = 17

        d_head_addr = 4
        d_head_val = 8
        heads = 4
        mem_size = 20

        update_addr = torch.randn([batch, heads, items, d_head_addr])
        update_data = torch.randn([batch, heads, items, d_head_val])
        mem_data = torch.randn([batch, heads, mem_size, d_head_val])

        set_mem = Attention.Set_Memory(d_head_addr,
                                       heads,
                                       mem_size)

        output = set_mem(mem_data, update_addr, update_data)

    def test_does_set(self):
        pass
    def test_does_not_set(self):
        pass
    def test_parallel(self):
        pass

