import torch
from torch.nn import functional as F
from torch import nn
from typing import Dict, Union
from typing import List
import numpy as np





tensor = torch.randn([3, 10, 10])
array = tensor.numpy()
array = np.pad(array, [(1, 2), (1, 2)])
tensor = torch.from_numpy(array)
print(tensor.shape)
