## 12/18/2025 - Naive GEMM

[source](src/kernels/gemm/v0_naive_gemm/kernel.hip.cpp)

My first introduction to GEMM and matrix multiplication in general. Came to a better understanding of
LDS, banks, and bank conflicts.

## 12/10/2025 - Stable Softmax

[source](src/kernels/reduction/v3_softmax/kernel.hip.cpp)

My first introduction to Softmax, the expf function, and warp shuffling.

## 12/05/2025

[source](src/kernels/reduction/v1_halving/kernel.hip.cpp)

Wrote a simple aggregation kernel which launches log2(N) kernels to
process input of size N. Spent a lot of time profiling this kernel with
rocprofv3 to try and prove that it is memory-bound.

## 11/18/2025 - 3D 7-point stencil with LDS tiling

[source](src/kernels/stencil/v1_tiled_7_point/kernel.hip.cpp)

Learned about tiling with halos.

## 11/14/2025 - parallel add

This one took a while to wrap my head around. Learned about LDS, tiling, different grid launch patterns.

[source](src/kernels/reduction/v0_naive_pairwise/kernel.hip.cpp)

This was my first ever kernel. Mostly learning about how to launch threads, how the host and device interact, grids.
