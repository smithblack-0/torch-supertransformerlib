import torch
from torch import nn
from typing import Dict, Union



def test_if_sparse(tensor: torch.Tensor):
    tensor = tensor.coalesce()
    print(tensor.indices().shape)
    return tensor.sparse_dim()
i = [[0, 1, 1],
         [2, 0, 2]]
v =  [[3, 4], [5, 6], [7, 8]]
s = torch.sparse_coo_tensor(i, v, (2, 3, 2))


tensor = torch.randn([3, 10, 20])
tensor = tensor.to_sparse_coo()
print(tensor.indices().shape)