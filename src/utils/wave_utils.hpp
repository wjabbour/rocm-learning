#pragma once
#include <hip/hip_runtime.h>

// Performs a wavefront-level reduction to compute the sum of 'val' across all threads in the wavefront
template <typename T>
__device__ __forceinline__ T waveReduceSum(T val) {
    // assumes a wavefront size of 32
    val += __shfl_down(val, 16, 32);
    val += __shfl_down(val, 8, 32);
    val += __shfl_down(val, 4, 32);
    val += __shfl_down(val, 2, 32);
    val += __shfl_down(val, 1, 32);

    return val;
}