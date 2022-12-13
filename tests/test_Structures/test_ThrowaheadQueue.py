"""
Test the throwahead queue mechanism
"""

import torch
import src.supertransformerlib.Structures.ThrowaheadQueue as ThrowaheadQueue
from src.supertransformerlib import Structures
import unittest

class test_ThrowaheadQueue(unittest.TestCase):
    def test_constructor(self):
        length = 10
        shape = torch.tensor([3, 5, 4])
        queue = Structures.ThrowaheadQueue(length, shape)

    def test_queuing(self):
        length = 3
        shape = torch.tensor([2, 2])

        queue = Structures.ThrowaheadQueue(length, shape)
        test_data = torch.tensor([[1.0, 0], [0.0, 1]])
        queue_weights = torch.tensor([0.3, 0.7, 0.0])

        queue.enqueue(queue_weights, test_data)
        dequeue1 = queue.dequeue()
        self.assertTrue(torch.all(dequeue1 == test_data*0.3))

        dequeue2 = queue.dequeue()
        self.assertTrue(torch.all(dequeue2 == test_data*0.7))



