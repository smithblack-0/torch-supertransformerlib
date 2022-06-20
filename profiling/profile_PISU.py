import torch
from torch.utils import benchmark
from src.supertransformerlib import Attention



pisu = Attention.PISU(64, 64, 10, 8).to(torch.device("cuda"))
pisu = torch.jit.script(pisu)
data = torch.randn([1, 10000, 64]).to(torch.device("cuda"))
sample_sizes = torch.arange(1000, 10000, 1000)

benchmarks = []
times = 400
for sample_size in sample_sizes:
    timer = benchmark.Timer(
        stmt="pisu(data)",
        setup="from __main__ import pisu",
        globals={"data" : data[:, sample_size]},
        description= "pisu with s% entries" % int(sample_size)
    )
    benchmarks.append(timer.timeit(times))
    print("done with sampels of %s" % sample_size)

compare = benchmark.Compare(benchmarks)
compare.print()

