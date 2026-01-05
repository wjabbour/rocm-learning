# Overview

This GEMM kernel uses LDS tiling to "sweep" a TILExTILE tile through input matrix A in the x dimension and through input matrix B in the vertical direction.

## Build Instructions

```bash
make clean && make run SRC=src/kernels/gemm/v0_naive_gemm/kernel.hip.cpp
```