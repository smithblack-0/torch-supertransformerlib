import torch
from torch.utils import benchmark
from src.supertransformerlib import Attention
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

mha = Attention.MultiHeadedAttention(64, 10, 8).to(torch.device("cuda"))
mha = torch.jit.script(mha)
data = torch.randn([1, 30000, 64]).to(torch.device("cuda"))
sample_sizes = torch.arange(10, 1000, 10)

benchmarks = []
times = 400
for sample_size in sample_sizes:
    timer = benchmark.Timer(
        stmt="mha(data)",
        setup="from __main__ import mha",
        globals={"data" : data[:, :sample_size]},
        description= "mha with s% entries" % int(sample_size)
    )
    benchmarks.append(timer.timeit(times))
    print("done with samples of %s" % sample_size)

compare = benchmark.Compare(benchmarks)
compare.print()

y = [item.mean for item in benchmarks]
x = sample_sizes

df = pd.DataFrame({"x" : x, "y" : y})
sns.lineplot(data=df, x="x", y= 'y')
plt.show()