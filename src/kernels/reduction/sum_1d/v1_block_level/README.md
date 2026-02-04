# Overview

Strategy:

- Algorithm - 

- Execution Model - one device pass, one host pass, grid-stride loop. Launch a set amount of blocks, each wave front striding along the input summing while it strides, wavefront shuffling for wavefront aggregation, writing intermediary outputs to LDS, and wavefront-representative threads which sum LDS and write to global output. Final output is summed by host.

Bottlenecks:

## Build Instructions

```bash
make clean && make run SRC=src/kernels/reduction/sum_1d/v1_block_level/kernel.hip.cpp
```

## Learnings

- Use `size_t` instead of `int` when declaring the global thread id if you plan to launch more than 2^32-1 threads, else you will encounter integer overflow which will affect the correctness of the kernel.

- HIP exposes the `warpSize` built-in device variable which detects the hardware's warp size. This can be used to increase kernel portability.