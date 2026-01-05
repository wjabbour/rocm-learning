# Overview

A GEMM kernel which multiplies two input matrices, A and B, and produces an output matrix C.

Strategy:

- Algorithm - using a TILE_SIZExTILE_SIZE LDS tile, calculate blocks of C.

- Execution Model - single-pass kernel. For each block of indices in C, we stride a tile horizontally along input A and vertically along input B.

Bottlenecks:

## Build Instructions

```bash
make clean && make run SRC=src/kernels/gemm/v0_tiled_gemm/kernel.hip.cpp
```