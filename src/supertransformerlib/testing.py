import torch
from torch import nn



core = torch.arange(100).view(10, 10)
index = torch.tensor([[2, 3],[3,4]])
gathered = torch.gather(core, dim=-1, index=index)
print(core)
print(gathered)
