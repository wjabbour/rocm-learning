# Overview

This kernel processes a 3D input of floats by averaging each element with its 6 neighbors (up, down, left, right, back, front).

Strategy:

- Algorithm - 

- Execution Model - single-pass kernel.

Bottlenecks:

## Build Instructions

```bash
make clean && make run SRC=src/kernels/stencil/v1_tiled_7_point/kernel.hip.cpp
```