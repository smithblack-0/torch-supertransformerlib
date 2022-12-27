"""
A module for the complex memory and access mechanism which exists including
the plug-and-play pieces plus a few default layers.

The mechanism boils down to clever manipulation of blocks of key, value parameters. These
can be manipulated by appropriate classes to influence the contents of the key or the values,
or to get information back out of the memory bank.
"""

from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
from src.supertransformerlib.Attention import Utility
from src.supertransformerlib import Core

@torch.jit.script
@dataclass
class MemoryTensor:
    Addresses: torch.Tensor
    Contents: torch.Tensor

class CommitmentScoringAttn(nn.Module):
    """
    Description:

        Calculates the commitment score, which
        is a number between 0 and 1 for each output element in
        attention. This is used to determine things such as how strongly
        setting occurs.

    Action:
        Score as in attn.
        Do not activate the score with softmax.
        Sum up the score
        Add the bias
        Send through a sigmoid.
        Return.
    """

    def __init__(self,
                 heads: int,
                 mem_bank_size: int,
                 parallel: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        super().__init__()

        if parallel is not None:
            parallel = Core.standardize_shape(parallel, "parallel")
            parallel = parallel.tolist()
        else:
            parallel = []

        bias_shape = parallel + [heads, mem_bank_size]
        bias = torch.zeros(bias_shape, dtype=dtype, device=device)
        bias = nn.Parameter(bias)
        self.bias = bias

    def forward(self, query: torch.Tensor, key: torch.Tensor)->torch.Tensor:
        score = torch.matmul(query, key.transpose(-1, -2))
        logits = score.sum(dim=-1)
        logits = logits + self.bias
        commitment = torch.sigmoid(logits)
        commitment = commitment.unsqueeze(-1)
        return commitment


class MemSetAddresses(nn.Module):
    """
    Description:
        Destructively sets to addresses using ranked commitment logic.
        Returns the new address memory
    Action:
        Take in old addresses along with update addresses and update values
        update
        Return
    """
    def __init__(self,
                 d_model: int,
                 heads: int,
                 d_head: int,
                 mem_bank_size: int,
                 parallel: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        super().__init__()

        self.make_commitment = CommitmentScoringAttn(heads, mem_bank_size,
                                                     parallel, dtype, device)
        self.make_address_head = Utility.MakeHead(d_model, heads, d_head,
                                               parallel, dtype, device)
        self.make_content_head = Utility.MakeHead(d_model, heads, d_head,
                                                parallel, dtype, device)
        self.norm = nn.LayerNorm(d_head)

    def forward(self,
                memory: MemoryTensor,
                update_addresses: torch.Tensor,
                update_content: torch.Tensor) -> MemoryTensor:
        """
        Performs the update process. Memory_keys are the current addresses,
        update_addresses are where we intend to update, and of course update
        values the contributions
        :param memory: The current memory tensor, which consists of the address, content pair.
        :param update_addresses: The addresses we wish to update at. Used to actually find
         the locations where we dump the content
        :param update_content: The content where we wish to dump info at:
        :return: A new memory tensor of some sort.
        """

        current_addresses = memory.Addresses
        update_addresses = self.make_address_head(update_addresses)
        update_content = self.make_content_head(update_content)

        update = Utility.dot_product_attention(current_addresses,
                                               update_addresses,
                                               update_content, activation_mode="sigmoid")

        commitment = self.make_commitment(current_addresses, update_addresses)

        new_addresses = current_addresses * (1 - commitment) + self.norm(update) * commitment

        output = MemoryTensor(new_addresses, memory.Contents)
        return output

class MemAddAddress(nn.Module):
    """
    Description:
        Adds to the address portion of memory.
        Commitment logic is used to ensure addition can be zero.
    """
    def __init__(self,
                 d_model: int,
                 heads: int,
                 d_head: int,
                 mem_bank_size: int,
                 parallel: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        super().__init__()

        self.make_commitment = CommitmentScoringAttn(heads, mem_bank_size,
                                                     parallel, dtype, device)
        self.make_address_head = Utility.MakeHead(d_model, heads, d_head,
                                               parallel, dtype, device)
        self.make_content_head = Utility.MakeHead(d_model, heads, d_head,
                                                parallel, dtype, device)

    def forward(self,
                memory: MemoryTensor,
                update_addresses: torch.Tensor,
                update_content: torch.Tensor) -> MemoryTensor:
        """
        Performs the update process. memory contains the actual memory tensor
         which is taken apart and updated with the sum.
        :param memory: The current memory tensor, which consists of the address, content pair.
        :param update_addresses: The addresses we wish to update at. Used to actually find
         the locations where we dump the content
        :param update_content: The content where we wish to dump info at:
        :return: A new memory tensor of some sort.
        """

        current_addresses = memory.Addresses
        update_addresses = self.make_address_head(update_addresses)
        update_content = self.make_content_head(update_content)

        update = Utility.dot_product_attention(current_addresses, update_addresses, update_content)
        commitment = self.make_commitment(current_addresses, update_addresses)

        new_addresses = current_addresses + update * commitment

        output = MemoryTensor(new_addresses, memory.Contents)
        return output

