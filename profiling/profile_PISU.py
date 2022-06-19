import torch
from torch.utils import benchmark
from src.supertransformerlib import Attention



pisu = Attention.PISU(64, 64, 10, 8)
data = torch.zeros([1, 10000, 64])
sample_sizes = torch.arange(0, 10000, 1000)

benchmarks = []
times = 100
for sample_size in sample_sizes:
    timer = benchmark.Timer(
        stmt="pisu(data)",
        setup="from __main__ import pisu",
        globals={"data" : data}
    )
    benchmarks.append(timer.timeit(times))




