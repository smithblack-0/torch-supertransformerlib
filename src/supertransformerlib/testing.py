import torch


test = torch.randn([3, 5, 5, 10])
test2 = torch.randn([10, 15])

matmul = torch.matmul(test, test2)

test= test.unsqueeze(-1)
matmul2 = (test*test2).sum(dim=-2)

diff = matmul - matmul2
print(diff)