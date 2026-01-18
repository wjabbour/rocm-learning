# Overview

This is my first introduction into writing GPU kernels.

## Build Instructions

```bash
make clean && make run SRC=src/kernels/reduction/v0_naive_pairwise/kernel.hip.cpp
```

## Learnings

- The size of a hardware wavefront on my RDNA GPU is 32, whereas most CDNA GPUs have a wavefront size of 64.

- Memory coalescence is crucial. Most inference workloads spend their time primarily waiting for data, not calculating, because HBM is far slower than the compute throughput of a GPU. We need to increase the efficiency of our memory operations whenever possible. The memory controller of a GPU is much wider than a CPU, fetching cache lines (32B/64B/128B) at a time, depending on architecture. As threads issue loads, the memory fabric on the chip coalesces loads bound for the same cache lines into single, wide loads in order to optimize memory access.

  For this reduction kernel, I am careful to ensure that each thread in the wavefront accesses contiguous memory. This ensures that the impact of global reads on latency are amortized across all of the threads - the GPU can satisfy the memory loads for all the threads in the wavefront using a single HBM transaction.
