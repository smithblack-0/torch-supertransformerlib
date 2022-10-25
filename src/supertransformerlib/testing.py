import torch
from torch.nn import functional as F
from torch import nn
from typing import Dict, Union
from typing import List

tensor = torch.randn([10, 10])
tensor = tensor.unsqueeze(-2).unsqueeze(-1)
tensor = tensor.expand(-1, 5, -1, 2)
tensor = tensor.view(10*5, 10*2)
print(tensor.shape)
