import torch
from torch.nn import functional as F
from torch import nn
from typing import Dict, Union
from typing import List
import numpy as np

tensor = torch.arange(100).reshape(10, 10)
mask = torch.rand([10, 10]) > 0.5


viewpoint = tensor[mask]
viewpoint[:] = 1
print(tensor)
print(viewpoint)