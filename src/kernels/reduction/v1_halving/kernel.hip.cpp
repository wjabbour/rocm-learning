#include <hip/hip_runtime.h>
#include "utils/random_int.hpp"

#define WORK_PER_THREAD 2

__global__ void pairwise_reduction(int* in, int* out, size_t N) {
    // we are launching a 1D grid of threads, so the index of this thread in the x-dimension is its global id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    /*
        with each kernel launch, we launch {current_value_of_n} / WORK_PER_THREAD threads
        since we are launching less threads than N, each thread needs to be responsible for 
        summing WORK_PER_THREAD units of N
    */
    int i = tid * WORK_PER_THREAD;

    /*
        our thread must be within N, else we will pull garbage data
    */
    if (i < N) {
        int sum = 0;
        /*
            once we know that we are on a thread that is within N, we need to find
            how many indices to the right (up to WORK_PER_THREAD) are also within N

            we start at the rightmost index, walking left until we find that the index is
            within N and use the inner loop to sum the indices
        */
        for (int j = WORK_PER_THREAD - 1; j >= 0; j--) {
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

    size_t N = 1ULL << 31;
    size_t size = N * sizeof(int);

    int *in_d, *out_d;

    int *in_h = (int*)malloc(size);
    if (in_h == NULL) {
        printf("Input array host memory allocation failed\n");
        exit(1);
    }

    int *out_h = (int*)malloc(size);
    if (in_h == NULL) {
        printf("Output array host memory allocation failed\n");
        exit(1);
    }

    printf("Allocated input and output arrays\n");

    hipMalloc(&in_d, size);
    hipMalloc(&out_d, size);

    for (size_t i = 0; i < N; i++) {
        in_h[i] = Utils::Random::int_in_range(1, 10);
        out_h[i] = 0;
    }

    printf("Populated input and output arrays\n");

    hipMemcpy(in_d, in_h, size, hipMemcpyHostToDevice);
    hipMemcpy(out_d, out_h, size, hipMemcpyHostToDevice);
    
    size_t currentN = N;

    hipEvent_t k_start, k_stop;
    hipEventCreate(&k_start);
    hipEventCreate(&k_stop);

    float kernel_total_μs = 0.0f;

    int blockSize = 64;
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    while (currentN > 1) {
        int outputSize = (currentN + WORK_PER_THREAD - 1) / WORK_PER_THREAD;
        int blockCount = (outputSize + blockSize - 1) / blockSize;

        if (size_t(blockCount * blockSize) > (size_t)prop.maxGridSize[0]) {
            printf("CRITICAL ERROR: Needed %lu blocks, but GPU Max is %d\n", blockCount, prop.maxGridSize[0]);
            exit(1);
        }

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
    printf("Result: %i\n", out_h[0]);

    hipFree(in_d);
    hipFree(out_d);
    free(in_h);
    free(out_h);

    hipEventRecord(sys_stop, 0);
    hipEventSynchronize(sys_stop);

    float sys_ms = 0.0f;
    hipEventElapsedTime(&sys_ms, sys_start, sys_stop);

    printf("Kernel-only time = %f μs\n", kernel_total_μs * 1000);
    printf("System time (host+device) = %f ms\n", sys_ms);

    return 0;
}
