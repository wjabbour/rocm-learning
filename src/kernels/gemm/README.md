# Overview

General Matrix Multiply (GEMM) kernels are a core piece of training and inference pipelines. A GEMM kernel is responsible for efficiently multiplying two matrices together to produce an output matrix.

# Learnings

- When requesting data from the memory system (L1, L2, HBM) it is important to ensure that the threads of a wavefront request contiguous data so that the memory controller can coalesce the requests from each thread into as few transactions as possible.

It's important to note that a motivation for this design is the fact that CDNA HBM is optimized to move data in bursts. The fact that the HBM is capable of bursts means that we can attempt to use as few bursts as possible to transfer the data to our GPU.

- LDS on the other hand is only shared by the wavefronts of a block. It is organized at the hardware level into banks where the number of banks matches the number of SIMD lanes.

When requesting data from LDS, it is important to ensure that the threads of a wavefront either:

1) all request from the same

