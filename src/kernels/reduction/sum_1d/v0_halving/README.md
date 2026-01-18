# Overview

This project implements a recursive reduction algorithm.

Strategy:

- Algorithm - each thread sums `WORK_PER_THREAD` elements from global memory and writes the intermediate answer to global memory. The elements that each thread processes and the elements that the neighboring threads process are contiguous in global memory, ensuring memory coalescing.

- Execution Model - multi-pass host-side loop. Each kernel launch is acting as a synchronization point. With each launch, we reduce the input size by a factor of `WORK_PER_THREAD` until the final sum is produced.

Bottlenecks:

- Because we do not shrink N aggressively enough, we interact with global memory more frequently than desired.

## Build Instructions

```bash
make clean && make run SRC=src/kernels/reduction/v1_halving/kernel.hip.cpp
```

## Learnings

- This is my first time learning about multi-pass kernels. In the case of such a kernel, we must launch multiple kernels in order to achieve the desired result. This is required because the problem is one that requires the output of some blocks to be used as input for other stages of computation. Since there is no synchronization primitive offered for blocks in the same kernel launch, we must run the dependent blocks in separate kernel launches.

- There are at least two ways to go about a reduction: halving reduction and block-level reduction.

  For halving reduction, for input of size N, you create grid of threads of some size < N, perform the reduction such that each thread in the grid strides along N, and now you have an output of size grid. From here, you can launch kernels of grid/j, where j doubles each time until grid/j = 1.

  For block-level reduction, each block of threads is responsible for aggregating the values computed by the threads in the block. At the end of this reduction, you are left with gridDim partials (one for each block). At that point, the input size has been compressed from something arbitrarily massive, to something on the order of thousands (hopefully). We can then launch one more kernel with gridSize = 1 and blockSize ~= 256 to process the remaining input and reduce to a single value.

  Block-level reduction thus requires a constant (and small) number of kernel launches to perform the reduction, whereas halving reduction requires Log(n) kernel launches. Additionally, the work performed amongst the halving reduction kernel launches varies wildly. The first launch is responsible for N/2 work, while the final kernels are leaving SIMD lanes vacant as we are processing less than a hardware wavefront.

- The notion and importance of latency hiding was reinforced. The halving kernel is a multi-pass kernel where the wavefronts of the kernel request data from global memory. The kernel has no LDS usage, very low register pressure due to the small amount of variable declarations and operations present, which means that many wavefronts are simultaneously resident on the CU (occupancy). Even though the kernel is memory-bound, performance scales linearly for N because the wavefront executing on a SIMD is pre-empted by the CU scheduler when the scheduler detects that the SIMD is waiting on a high-latency operation (like global memory fetch).

- Consulted the [rocprofv3 docs](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) in order to understand which hardware counters I had access to on my GPU.
