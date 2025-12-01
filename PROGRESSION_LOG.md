## 11/18/2025 - 3D 7-point stencil with LDS tiling

[source](kernels/stencil/v1_tiled_7_point/kernel.hip.cpp)

## 11/14/2025 - parallel add

This one took a while to wrap my head around. Learned about LDS, tiling, different grid launch patterns.

[source](kernels/reduction/v0_naive_pairwise/kernel.hip.cpp)

This was my first ever kernel. Mostly learning about how to launch threads, how the host and device interact, grids.