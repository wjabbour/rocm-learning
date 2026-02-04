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

- 1
- 2
- 3

### Diagnosis

- 1
- 2
- 3

### Suggested Improvements

{description}
