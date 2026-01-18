#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "utils/wave_utils.hpp"
#include "utils/hip_check.hpp"

__device__ float waveReduceMax(float val) {
    float neighbor = __shfl_down(val, 16);
    val = max(neighbor, val);

    neighbor = __shfl_down(val, 8);
    val = max(neighbor, val);

    neighbor = __shfl_down(val, 4);
    val = max(neighbor, val);

    neighbor = __shfl_down(val, 2);
    val = max(neighbor, val);

    neighbor = __shfl_down(val, 1);
    val = max(neighbor, val);

    return val;
}

__global__ void softmaxKernel(float* input, float* output) {
    __shared__ float shared_sums[8];
    __shared__ float shared_max[8];

    int tid = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int wf_id = threadIdx.x / 32;
    
    float val = input[tid];

    // calculate the max for the wavefront using shuffling
    float max = waveReduceMax(val);

    // one thread from each wavefront is resonsible for writing to LDS
    if (lane_id == 0) {
        shared_max[wf_id] = max;
    }

    __syncthreads();

    float block_max = 0.0f;

    // now that all wavefronts have written their max, find the max amongst the wavefronts
    if (wf_id == 0) {
        block_max = (lane_id < 8) ? shared_max[lane_id] : -__FLT_MAX__;

        block_max = waveReduceMax(block_max);

        if (lane_id == 0) {
            shared_max[0] = block_max;
        }
    }

    __syncthreads();

    float global_max = shared_max[0];

    // the max is needed for the expf calculation
    float expf_val = expf(val - global_max);

    // calculate the sum for the wavefront using shuffling
    float sum = waveReduceSum(expf_val); 
    
    // one thread from each wavefront is resonsible for writing to LDS
    if (lane_id == 0) {
        shared_sums[wf_id] = sum;
    }
    
    __syncthreads();

    float block_sum = 0.0f;

    // now that all wavefronts have written their sum, find the sum of sums
    if (wf_id == 0) {
        /*
            ensure all threads in the wavefront participate. in general, we don't
            want to allow threads to retrieve data from non-executing threads
            
            and adding 0 to the blockSum doesnt change our answer
        */
        block_sum = (lane_id < 8) ? shared_sums[lane_id] : 0.0f;

        block_sum = waveReduceSum(block_sum);

        if (lane_id == 0) {
            shared_sums[0] = block_sum;
        }
    }

    __syncthreads();

    float global_sum = shared_sums[0];

    // the answer for each thread
    output[tid] = expf_val / global_sum;
}

int main() {
    int n = 256;
    size_t bytes = n * sizeof(float);

    std::vector<float> h_in(n, 1.0f); // Input: all 1s
    std::vector<float> h_out(n);

    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(softmaxKernel, dim3(1), dim3(256), 0, 0, d_in, d_out);
    HIP_KERNEL_CHECK();

    HIP_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));

    std::cout << "result: " << h_out[0] << std::endl;

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}