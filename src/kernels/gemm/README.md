# Overview

A General Matrix Multiply (GEMM) kernel is responsible for efficiently multiplying two matrices together to produce an output matrix. This is a fundamental operation in multiple machine learning algorithms, including the transformer architecture proposed by the "Attention is all you need" paper in 2017.

## Kernel Progression

| Version | Name            | Key Strategy | Bottleneck Solved                                                              | Remaining Issue                                                                          |
| :------ | :-------------- | :----------- | :----------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| **v1**  | `v1_tiled_gemm` | LDS          | **Memory Bandwidth:** cooperative LDS usage reduces global memory transactions | **Low Arithmetic Intensity:** Each thread is performing two loads from LDS per iteration |