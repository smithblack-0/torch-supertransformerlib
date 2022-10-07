"""


Dynamic module

Dynamic consists of methods for allowing the program to configure
its own architecture and data structures, generally by means of
attention based adaptive pointers.

It is designed to basically allow many different ensembles
to exist and be utilized in parallel while only running
the ensembles actually immediately in use.


# Design

Dynamic layers consist of a reservour and execution class,
and a compatible configuration tensor. It is designed to
maximize computational efficiency at the cost of memory.




Each element of the

"""

import torch
from torch import nn
from typing import List, Callable, Optional



