# Block-level Reduction

**GPU**: AMD Radeon RX 9070 XT

- Memory Bandwidth: Up to 640 GB/s

**CPU**: AMD Ryzen 7 9800X3D 8-Core Processor  
**Wavefront Size**: 32  
**ROCm version**: 7.1.0  
**Tools**: hipEvent, rocprofv3, sqlitebrowser  
**OS**: Ubuntu 24.04.3 LTS

## Baseline Implementation

### Performance Summary

| Implementation | Kernel Time (μs) | System Time (ms) | N    | Kernel Speedup vs v1 | System Speedup vs v1 | Bandwidth Efficiency (%) |
| -------------- | ---------------- | ---------------- | ---- | -------------------- | -------------------- | ------------------------ |
| **v1**         | 27,502           | 6,607            | 2^31 | -                    | -                    | 49%                      |

### Description

The baseline implementation involves the host copying over N elements to the device, the host launches a kernel with a thread for each element in N, each thread finds its global index in the grid and requests that index from global memory, each wavefront of each block participates in a wave-shuffle to sum all of the values for that wavefront, one thread from each wavefront is responsible for writing that wavefront's sum to LDS, all threads in the block synchronize, one thread from each block sums the wavefront sums from its respective block and writes to global output.

At this point, N has been reduced by a factor of `N / block_size`. A new kernel is launched to process this smaller input. The reduction is complete once the output size is 1.

### Profiling Observations

Still having trouble collecting PMCs on my current setup.


### Diagnosis

I was definitely expecting this to have a faster execution time than the halving kernel out of the gate since we greatly reduced the amount of writes to global memory by having our intermediate sums written and read to LDS. I suspect that there are two large pitfalls with this implementation when compared to the optimized halving approach:

1) The halving kernel uses a configurable WORK_PER_THREAD variable which determines how many elements from global memory each thread is responsible for summing. As this number increases the arithmetic intensity of the thread increases and the fixed fee of wavefront scheduling is amortized as wavefronts can stay resident on CUs for longer. This block reduction is only processing one element per thread.
2) Closely related to the first point, we are paying high costs for scheduling overhead and treating our threads as dispensable global data fetchers. Since this kernel is inherently memory bound, we are better off launching a fixed number of threads and performing all compute on those threads only, drastically reducing any scheduling overhead.

### Suggested Improvements

Let's implement a grid-stride loop. We will launch a fixed number of blocks which perform a large portion of our reduction and have the host perform a final, small reduction.

## Second Implementation

### Performance Summary

| Implementation | Kernel Time (μs) | System Time (ms) | N    | Kernel Speedup vs v1 | System Speedup vs v1 | Bandwidth Efficiency (%) |
| -------------- | ---------------- | ---------------- | ---- | -------------------- | -------------------- | ------------------------ |
| **v1**         | 27,502           | 6,607            | 2^31 | -                    | -                    | 49%                      |
| **v2**         | 15,876           | 6,725            | 2^31 | 1.73x faster         | 0.98x (slower)         | 84%                      |
