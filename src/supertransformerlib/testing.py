import torch
from torch import nn
from typing import Dict, Union
from typing import List

def example_generator(dim_length: int, start: int, end: int, dilation: int,  stride: int, offset: int):
    example = torch.arange(dim_length)
    output: List[torch.Tensor] = []


    minimum_unstrided_sampling_location = -start * dilation
    mintemp = torch.div(minimum_unstrided_sampling_location - offset, stride)
    offset_minimum_location = stride*torch.ceil(mintemp) + offset

    maximum_unstrided_sampling_location = dim_length - end * dilation - 1
    maxtemp = torch.div(maximum_unstrided_sampling_location - offset, stride)
    offset_maximum_location = stride*torch.floor(maxtemp) + offset

    print("min", offset_minimum_location)
    print("max", offset_maximum_location)
    print("num", (offset_maximum_location-offset_minimum_location)/stride + 1)

    for i in range(0, example.shape[0], stride):
        options = torch.range(start, end).to(dtype=torch.int64)*dilation + i + offset
        if torch.all(0 <= options) and torch.all(options < example.shape[0]):
            sample = example[options]
            output.append(sample)
    return output


print(example_generator(8, -1, 1, 2, 2, 1))