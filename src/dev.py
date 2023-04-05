import torch

@torch.jit._overload
def fn(x: int) -> int:
    pass

@torch.jit._overload
def fn(x: str) -> str:
    pass

def fn(x):
    if isinstance(x, int):
        return x + 3
    else:
        return x

# Example usage
myfunc = torch.jit.script(fn)
t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = torch.tensor([4.0, 5.0, 6.0])
print(myfunc(t1, t2))
