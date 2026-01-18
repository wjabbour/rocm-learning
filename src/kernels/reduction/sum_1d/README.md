# Overview

This directory contains a kernel progression for a kernel which reduces all elements in the input to a single value.

## Kernel Progression

| Version | Name             | Key Strategy                             | Bottleneck Solved            | Remaining Issue                                                                              |
| :------ | :--------------- | :--------------------------------------- | :--------------------------- | :------------------------------------------------------------------------------------------- |
| **v0**  | `v0_halving`     | Each thread sums two contiguous elements | Correctness                  | **Low Arithmetic Intensity:** we can increase per-thread work to increase memory utilization |
| **v1**  | `v1_block_level` | Use LDS to reduce global memory access   | Reduced global memory access | -                                                                                            |
