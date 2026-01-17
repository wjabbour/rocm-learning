# Overview

A GEMM kernel which multiplies two input matrices, A and B, and produces an output matrix C.

Strategy:

- Algorithm - using two TILE_SIZExTILE_SIZE LDS tiles (one for A and one for B), calculate blocks of C by moving these tiles "horizontally" along A and "vertically" along B.

- Execution Model - single-pass kernel. The output matrix C is divided into TILE_SIZExTILE_SIZE groups. Each thread block computes a tile of C. The threads in a block cooperatively populate LDS with the necessary elements of A and B, using `__syncthreads()` to synchronize LDS population before calculation.

Bottlenecks:

- Occupancy - LDS and register usage decreases the maximum number of wavefronts that can be scheduled on a CU.

## Build Instructions

```bash
make clean && make run SRC=src/kernels/gemm/v0_tiled_gemm/kernel.hip.cpp
```

## Learnings

- There is a very important boundary check that we must make in this kernel. If the input size is not a multiple of the block size there will be threads launched that do not map to any input element. For example, consider a block of size 64 and an input of size 2. The hardware will launch 62 threads which do not map to any input element. Therefore, each thread must consider whether it is within the input domain.

    However, this check must also consider the dimensions of the input tensor. For example, consider an input of size 6 representing a 2x3 matrix. If we launch a 1x4 block and the job of the block is to sum all of the elements of a matrix row (this is an aggregation example, but still applies to GEMM), then thread 4 with threadIdx.x = 3 must ensure that it does not simply ask "Is 3 < 6"... it must ask "Is 3 < the length of the input row (3)?" in order to determine row grouping of the input domain and avoid incorrect results.