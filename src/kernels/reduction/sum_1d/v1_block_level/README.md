# Overview

Strategy:

- Algorithm - 

- Execution Model -

Bottlenecks:

## Build Instructions

```bash
make clean && make run SRC=src/kernels/reduction/sum_1d/v1_block_level/kernel.hip.cpp
```

## Learnings

- Use `size_t` instead of `int` when declaring the global thread id if you plan to launch more than 2^32-1 threads, else you will encounter integer overflow which will affect the correctness of the kernel.

