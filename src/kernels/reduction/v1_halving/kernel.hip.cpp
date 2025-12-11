#include <hip/hip_runtime.h>
#include <iostream>
#include "utils/random_int.hpp"

__global__ void pairwise_reduction(int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * 16;

    if (i < N) {
        int sum = 0;
        for (int j = 15; j >= 0; j--) {
            if (i + j < N) {
                for (int k = j; k >= 0; k--) {
                    sum += in[i + k];
                }
                break;
            }
        }

        out[tid] = sum;
    }
}

int main() {
    hipEvent_t sys_start, sys_stop;
    hipEventCreate(&sys_start);
    hipEventCreate(&sys_stop);

    hipEventRecord(sys_start, 0); 

    int N = 1 << 30;
    size_t size = N * sizeof(int);

    int blockSize = 64;

    int *in_d, *out_d;

    int *in_h = (int*)malloc(size);
    int *out_h = (int*)malloc(size);

    hipMalloc(&in_d, size);
    hipMalloc(&out_d, size);

    for (int i = 0; i < N; i++) {
        in_h[i] = Utils::Random::int_in_range(1, 5);
        out_h[i] = 0;
    }

    hipMemcpy(in_d, in_h, size, hipMemcpyHostToDevice);
    hipMemcpy(out_d, out_h, size, hipMemcpyHostToDevice);

    int currentN = N;

    hipEvent_t k_start, k_stop;
    hipEventCreate(&k_start);
    hipEventCreate(&k_stop);

    float kernel_total_μs = 0.0f;

    while (currentN > 1) {
        // lazily, launching too many threads. I only need currentN / 4
        int outputSize = (currentN + 15) / 16;
        int blockCount = (outputSize + blockSize - 1) / blockSize;

        hipEventRecord(k_start, 0);

        hipLaunchKernelGGL(
            pairwise_reduction,
            dim3(blockCount),
            dim3(blockSize), 
            0,
            0,
            in_d,
            out_d,
            currentN
        );

        hipEventRecord(k_stop, 0);
        hipEventSynchronize(k_stop);

        float k_μs = 0.0f;
        hipEventElapsedTime(&k_μs , k_start, k_stop);
        kernel_total_μs += k_μs;

        hipDeviceSynchronize();

        /*
            ping pong buffering

            we alloc two fixed size buffers of data that we use throughout the 
            entirety of the kernel
        */
        std::swap(in_d, out_d);
        currentN = outputSize;
    }

    /*
        since we swap the pointers after every kernel launch, in_d will
        always hold the most recent output data. I swap once more here because
        it just feels "right" to memcpy from device output pointer :) 
    */
    std::swap(in_d, out_d);
    hipMemcpy(out_h, out_d, sizeof(int), hipMemcpyDeviceToHost);

    // the answer!
    std::cout << out_h[0] << "\n";

    hipFree(in_d);
    hipFree(out_d);
    free(in_h);
    free(out_h);

    hipEventRecord(sys_stop, 0);
    hipEventSynchronize(sys_stop);

    float sys_ms = 0.0f;
    hipEventElapsedTime(&sys_ms, sys_start, sys_stop);

    std::cout << "Kernel-only time = " << kernel_total_μs * 1000 << " μs\n";
    std::cout << "System time (host+device) = " << sys_ms << " ms\n";

    return 0;
}
