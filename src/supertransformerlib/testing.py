from typing import Optional, Union, Dict, List, overload
import torch
from torch import nn
from torch.nn import Module
import dataclasses


sample_tensor = torch.randn([10, 10, 10])
index = torch.tensor([[[]]])
