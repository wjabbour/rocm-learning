# Halving Aggregation Kernel

**GPU**: AMD Radeon RX 6700 XT  
**CPU**: AMD Ryzen 5 5600X 6-Core Processor  
**ROCm version**: 7.1  
**Tools**: hipEvent, rocprofv3, sqlitebrowser  
**OS**: Ubuntu 24.04.3 LTS

## Baseline Implementation

## Performance Summary

| Implementation | Kernel Time (μs) | System Time (ms) | N    | Kernel Speedup vs v1 | System Speedup vs v1 |
| -------------- | ---------------- | ---------------- | ---- | -------------------- | -------------------- |
| **v1**         | 50,518           | 8,298            | 2^30 | -                    | -                    |

### Description

This is my first time profiling a kernel so I'm going to get a few things (or everything) wrong.

Each thread is responsible for adding two elements from global input and writing one value to global output. First we launch a kernel with gridSize = N, then we launch a kernel with gridSize = N/2 and so on until N/2 = k = 1.

### Profiling Observations

The initial kernel launches (where the number of active wavefronts exceeds ~64k) account for the majority of runtime.

The profiling counters for these large-N passes show:

- MeanOccupancyPerCU ≈ 22.7 → high occupancy
- MemUnitBusy ≈ 80% → strongly memory-bound
- Linear runtime scaling with N → bandwidth-limited

There is a flat fee to launch a kernel (scheduling, allocation, register allocation). Once N/2 = k becomes sufficiently small, our kernel launches are bottlenecked by this flat fee. e.g. the kernel duration for N = 128 is roughly equal to the kernel duration for N = 64 or N = 256. This flat fee dominates the runtime of the kernel launches with small N.

Here are the same counters for the final kernel launches:

- MeanOccupancyPerCU ≈ 0.006 → extremely low occupancy
- MemUnitBusy ≈ 3% → almost no bandwidth
- Flat runtime scaling with N → hardware scheduling limited

### Diagnosis

- Kernel launches with small N are contributing a flat amount to the overall reduction runtime.
- For high N kernels our memory subsystem is active 80% of the time - nearly saturated. We spend all this time waiting on global memory fetches, but the kernel aggregation only decreases the N of the next kernel by half. Therefore, many memory-bound passes are required to perform the aggregation.

### Suggested Improvements

Let's increase the per-thread workload. Instead of reducing two elements per thread, let's have each thread compute four elements. My hypothesis:

- N shrinks more aggressively - fewer reduction passes
- Fewer small-N kernel launches where runtime is dominated by fixed launch overhead.
- Increased arithmetic intensity per wavefront = improved latency hiding. The CU scheduler is always looking for wavefronts that are ready to execute, i.e. wavefronts that are not stalled waiting on HBM. In a low-arithmetic intensity kernel, each wavefront quickly consumes its data and returns to a stalled state, making it harder for the CU to find ready waves. By increasing arithmetic intensity, each wavefront performs more ALU work for every HBM load, spending more time in a ready-to-execute state. This increases the likelihood that the CU can issue useful work while other wavefronts are stalled.

## Second Implementation

## Performance Summary

| Implementation | Kernel Time (μs) | System Time (ms) | N    | Kernel Speedup vs v1 | System Speedup vs v1 |
| -------------- | ---------------- | ---------------- | ---- | -------------------- | -------------------- |
| **v1**         | 50,518           | 8,298            | 2^30 | -                    | -                    |
| **v2**         | 20,969           | 8,151            | 2^30 | 2.4x faster          | 1.01x faster         |

### Description

- 1
- 2
- 3

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
