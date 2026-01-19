# Halving Reduction Kernel

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
| **v1**         | 42,699           | 7,120            | 2^31 | -                    | -                    | 31%                      |

### Description

This is my first time profiling a kernel so I'm going to get a few things (or everything) wrong :smirk:.

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
- For high N kernels our memory subsystem is active 80% of the time - nearly saturated. We spend all this time waiting on global memory fetches, but the kernel reduction only decreases the N of the next kernel by half. Therefore, many memory-bound passes are required to perform the reduction.

### Suggested Improvements

We need to increase the per-thread workload. Let's make that a configurable value, increase it, and see what happens.

- N shrinks more aggressively - fewer reduction passes
- Fewer small-N kernel launches where runtime is dominated by fixed launch overhead.
- Increased arithmetic intensity per wavefront = improved latency hiding. The CU scheduler is always looking for wavefronts that are ready to execute, i.e. wavefronts that are not stalled waiting on HBM. In a low-arithmetic intensity kernel, each wavefront quickly consumes its data and returns to a stalled state, making it harder for the CU to find ready waves. By increasing arithmetic intensity, each wavefront performs more ALU work for every HBM load, spending more time in a ready-to-execute state. This increases the likelihood that the CU can issue useful work while other wavefronts are stalled.

## Second Implementation

### Performance Summary

| Implementation | Kernel Time (μs) | System Time (ms) | N    | Kernel Speedup vs v1 | System Speedup vs v1 | Bandwidth Efficiency (%) |
| -------------- | ---------------- | ---------------- | ---- | -------------------- | -------------------- | ------------------------ |
| **v1**         | 42,699           | 7,120            | 2^31 | -                    | -                    | 31%                      |
| **v2**         | 18,276           | 6,825            | 2^31 | 2.34x faster         | 1.04x faster         | 73%                      |

### Description

Instead of each thread adding two elements, each thread is now responsible for adding eight elements. With this increase in per-thread work, the kernel needs to spend more time computing. This allows the scheduler to hide the latency of the memory transactions. With this change, the kernel runtime halved and the effective throughput of the kernel is approaching the theoretical limit.

### Profiling Observations

Unfortunately, I recently upgraded to an RDNA 4 card. This card does not have support for the PMCs I need to profile the kernel in-depth. However, as expected I can see that the number of kernel launches dropped by a factor of 8 which contributes to the decreased runtime.

### Suggested Improvements

There are two primary improvements I can identify:

1. Vectorized Loads - the kernel is using scalar load instructions, loading individual `int`s from memory at a time. Since we know that in most cases we wish to load multiple `int`s at once, let's try using the `int4` data type to fetch 128 bits at a time. This will reduce the load on the instruction pipeline and increase performance.
2. Memory Bandwidth - the kernel is performing an excess number of memory writes (and subsequently reads) because the reduction does not shrink N aggressively enough. We need to use LDS and coordinate all of the wavefronts in a block to reduce to a single value. Now, each kernel pass will decrease N by a factor of threadsPerBlock.

Since #2 is a different paradigm, I will perform that work in the [block-level kernel reduction](../v1_block_level/kernel.hip.cpp).

## Third Implementation

### Performance Summary

| Implementation | Kernel Time (μs) | System Time (ms) | N    | Kernel Speedup vs v1 | System Speedup vs v1 | Bandwidth Efficiency (%) |
| -------------- | ---------------- | ---------------- | ---- | -------------------- | -------------------- | ------------------------ |
| **v1**         | 42,699           | 7,120            | 2^31 | -                    | -                    | 31%                      |
| **v2**         | 18,276           | 6,825            | 2^31 | 2.34x faster         | 1.04x faster         | 73%                      |
| **v3**         | 18,428           | 6,962            | 2^31 | 0.0x faster          | 0.0x faster          | 73%                      |

### Description

I implemented vectorized loads in the kernel.

### Profiling Observations

The runtime was not substantially different. Either the memory system or the compiler must be smart enough to detect my sequential load pattern and optimize it. I could potentially confirm this by trying to extract the assembly from the compiled binary and examining the instructions that were ultimately fed to the compute units.

### Suggested Improvements

Let's move forward with the [block-level reduction](../v1_block_level/kernel.hip.cpp).
