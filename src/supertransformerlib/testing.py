import torch

@torch.jit.script
def test_it(data: int):
    print(data)
    return data + 1

test_it(1)
test_it(2)
test_it(3)
test_it(4)
print(test_it.code)