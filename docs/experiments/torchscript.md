# This documents findings about torchscript

## Tracing

### Functions

Tracing is fairly straightforward for functions without 
flow control. Provide the example inputs, it executes
and spits out an output. Notably, flow control 
such as while loops can cause major problems.


## Functions

Functions can be cleanly compiled
by torchscript, both at moment of creation
and, possibly, right before usage.

Importantly, the compiled code is executed in
c/c++, which is much faster

```python

import torch

def example()->torch.Tensor:
    item = torch.randn([3, 5]) # Call this 1
    return item

func = torch.jit.script(example)
output = func()


```

It is, however, a smart idea to compile a function
in the environment it is defined in order to prevent
missing resource errors.

## Modules

Modules are the layers which make up the normal
torch architecture.

### Tracing

Modules can be traced. This shuts down non tensor 
flow control. This means flow control based off of
python state, such as if statements in loops,
does not correctly trigger.

However, flow control based off of the tensors
itself will. For example. a 'torch.while' statement
will function correctly. 

## Classes

Classes are loose bits of code which are not a subclass of nn.Module and,
indeed, not a subclass at all.

```python
import torch

class test_class:
    def __init__(self):
        self.test = 3
    def get(self):
        return self.test
    
compiled_class = torch.jit.script(test_class)
instance = compiled_class()

print(instance.get())
print(instance.get())
print(instance.get())
print(instance.get())


```

While these classes work, they have some pretty significant downsides. For instance,
the above code, when run, will actually execute in a python environment. That is, 

```python
import torch

class test_class:
    def __init__(self):
        self.test = 3 # Call this '1'
    def get(self):
        return self.test # Call this '2'
    
compiled_class = torch.jit.script(test_class)

instance = compiled_class() # Debugger will jump to 1
instance = compiled_class() # Debugger will jump to 1
instance = compiled_class() # Debugger will jump to 1
instance = compiled_class() # Debugger will jump to 1

print(instance.get()) # Debugger will jump to 2
print(instance.get()) # Debugger will jump to 2
print(instance.get()) # Debugger will jump to 2
print(instance.get()) # Debugger will jump to 2


```

This means the code is NOT executing in c/c++ like it does with function
Classes like this can, assuming dependencies. However, this is okay.

When this code is utilized in a function or outer compiled environment, 
it DOES compile



