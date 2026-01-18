# Overview

Softmax is one of the most critical kernels for inference. It takes in a list of K real numbers and calculates a probability distribution of K possible outcomes. In inference, this is used to predict the most likely token that should come next in the sequence.

This is a naive version of softmax because:

1) the input is an array of floats
2) we are only performing these calculations over a single block

Strategy:

- Algorithm - This kernel performs the following steps: First, we need to find the maximum value from the input. Each wavefront uses warp shuffling to find the maximum value in the wavefront, then writes that value to LDS. Next, we use warp shuffling again - this time using the first 8 threads (we use 8 because 256 (input size) divided by 32 (RDNA wavefront size) = 8) in the first wave to find the maximum amongst all 8 values in LDS. Now that we have the maximum value from the input, we perform a very similar set of steps to find the sum of the input. Once we have both the sum and the max of the input, we can compute our probability distribution: each thread picks a value from the input, subtracts the max from the value and computes e^(val - max) to yield y, then writes y / globalSum to the output.

- Execution Model - single-pass kernel operating on a single block of 256 threads.

Bottlenecks:

## Build Instructions

```bash
make clean && make run SRC=src/kernels/reduction/softmax/v0_naive_softmax/kernel.hip.cpp
```

## Learnings

- Warp shuffling is used for efficient intra-wavefront reductions. Each wavefront uses `__shfl_down()` to find the maximum and sum values within the wavefront before writing partial results to LDS for block-level aggregation.