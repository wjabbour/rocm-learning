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