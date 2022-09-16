from typing import Optional, Union, Dict, List, overload
import torch
from torch import nn
from torch.nn import Module


examples = []
shape = [3, 3]
for _ in range(10):
    A = torch.randn(shape)
    M = A.transpose(0, 1).matmul(A)
    print(M)
    print(M.sum())
    print(torch.det(M))
    print(torch.trace(M))