import torch
import unittest
from src.supertransformerlib.Attention import Throwahead


class test_Throwahead(unittest.TestCase):
   """
   Integration testing between all the various units
   involved in throwahead.
   """
   def test_throwahead(self):

       test_tensor = torch.rand([3, 4, 64])

       queue_size = 8
       batch_shape = [3, 4]
       elements_shape = 64

       factory = Throwahead.ThrowaheadFactory(queue_size, elements_shape)
       factory = torch.jit.script(factory)

       virtual_layer = factory(batch_shape)

       for i in range(10):
           weights = torch.rand([8, 3, 4])
           test_tensor = virtual_layer(test_tensor, weights)


