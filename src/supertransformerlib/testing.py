import torch


tensor=  torch.randn([5, 3]).to_sparse_coo()
tensor2 = torch.randn([7, 3, 4])
torch.sparse.mm(tensor, tensor2)