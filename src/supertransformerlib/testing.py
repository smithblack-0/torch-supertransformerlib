import torch
from torch import nn


class test_kernel_magic(nn.Module):

    @torch.jit.export
    def __getitem__(self, key):
        return self._item[key]

    @torch.jit.export
    def __setitem__(self, key, value):
        self._item[key] = value

    def __init__(self):
        super().__init__()
        self._item = torch.rand([10])


instance = test_kernel_magic()
instance = torch.jit.script(instance)
print(type(instance))
method_name = "__getitem__"
method = getattr(instance, method_name)
func = getattr(method, "__func__", None)
other = getattr(torch.jit._script.RecursiveScriptModule, method_name)
boolean_test = func == other
instance[torch.tensor([1])]