# Overview

This is my first introduction into writing GPU kernels. This kernel adds two arrays.

Strategy:

- Algorithm - This kernel receives a pointer to inputs A and B, each a vector of ints. Each thread calculates its global index, idx, and writes A[idx] + B[idx] to global memory, C[idx].

- Execution Model - single-pass kernel writing to global output

Bottlenecks:

- **Memory Latency:** We load two ints from global memory just to perform a single operation on them and then write the result to global memory. Our arithmetic intensity (operations per byte transferred) is 1/12 (read 8 bytes, 2 ints, write 4 bytes, 1 int) which is much lower than the theoretical compute saturation of the card. Our kernel is spending most of its time waiting for data to arrive from global memory transactions.

## Build Instructions

```bash
make clean && make run SRC=src/kernels/primitives/v0_vector_add/kernel.hip.cpp
```

## Learnings

- The size of a hardware wavefront on my RDNA GPU is 32, whereas most CDNA GPUs have a wavefront size of 64.

- Memory coalescence is crucial. Most inference workloads spend their time primarily waiting for data, not calculating, because HBM is far slower than the compute throughput of a GPU. We need to increase the efficiency of our memory operations whenever possible. The memory controller of a GPU is much wider than a CPU, fetching cache lines (32B/64B/128B) at a time, depending on architecture. As threads issue loads, the memory fabric on the chip coalesces loads bound for the same cache lines into single, wide loads in order to optimize memory access.

  For this kernel, I am careful to ensure that each thread in the wavefront accesses contiguous memory. This ensures that the impact of global reads on latency are amortized across all of the threads - the GPU can satisfy the memory loads for all the threads in the wavefront using a single HBM transaction.
