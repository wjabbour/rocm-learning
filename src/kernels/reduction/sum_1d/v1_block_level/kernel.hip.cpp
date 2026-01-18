#include <hip/hip_runtime.h>
#include "utils/random_int.hpp"
#include "utils/hip_check.hpp"

# define BLOCK_SIZE 512

__global__ void block_reduction(int* in, int* out, size_t N) {
    // we may launch more than 2^32 threads, so we need to use size_t for our global thread ID
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    /*
        our thread must be within N, else we will pull garbage data
    */
    if (i < N) {
        int sum = 0;
        /*
            once we know that we are on a thread that is within N, we need to find
            how many indices to the right (up to WORK_PER_THREAD) are also within N

            we start at the rightmost index (i + WORK_PER_THREAD), walking left until we find that the index is
            within N and use the inner loop to sum the indices from there to i
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
    HIP_CHECK(hipEventCreate(&sys_start));
    HIP_CHECK(hipEventCreate(&sys_stop));

    HIP_CHECK(hipEventRecord(sys_start, 0));

    size_t N = 1ULL << 31;
    size_t inputBytes = N * sizeof(int);

    // each block will produce a single output value, so our initial output size is equal to the number of blocks launched
    size_t outputElements = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t outputBytes = outputElements * sizeof(int);
    
    // allocate host memory
    int *in_d, *out_d;
    int *in_h = (int*)malloc(inputBytes);
    if (in_h == NULL) {
        printf("Input array host memory allocation failed\n");
        exit(1);
    }

    int* out_h = (int*)calloc(outputElements, sizeof(int));
    if (out_h == NULL) {
        printf("Output array host memory allocation failed\n");
        exit(1);
    }

    printf("Allocated input and output arrays\n");

    // allocate device memory
    HIP_CHECK(hipMalloc(&in_d, inputBytes));
    HIP_CHECK(hipMalloc(&out_d, outputBytes));

    printf("Allocated device arrays\n");

    // initialize host memory
    for (size_t i = 0; i < N; i++) {
        in_h[i] = Utils::Random::int_in_range(1, 1);
    }

    printf("Populated input array\n");

    // host -> device transfer
    HIP_CHECK(hipMemcpy(in_d, in_h, inputBytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(out_d, out_h, outputBytes, hipMemcpyHostToDevice));

    size_t currentN = N;

    hipEvent_t k_start, k_stop;
    HIP_CHECK(hipEventCreate(&k_start));
    HIP_CHECK(hipEventCreate(&k_stop));

    float kernel_total_ms = 0.0f;

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    while (currentN > 1) {
        int outputSize = (currentN + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blockCount = (outputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if (size_t(blockCount * BLOCK_SIZE) > (size_t)prop.maxGridSize[0]) {
            printf("CRITICAL ERROR: Needed %lu blocks, but GPU Max is %d\n", blockCount, prop.maxGridSize[0]);
            exit(1);
        }

        HIP_CHECK(hipEventRecord(k_start, 0));

        hipLaunchKernelGGL(
            block_reduction,
            dim3(blockCount),
            dim3(BLOCK_SIZE), 
            0,
            0,
            in_d,
            out_d,
            currentN
        );
        HIP_KERNEL_CHECK();

        HIP_CHECK(hipEventRecord(k_stop, 0));
        HIP_CHECK(hipEventSynchronize(k_stop));

        float k_μs = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&k_μs , k_start, k_stop));
        kernel_total_ms += k_μs;

        HIP_CHECK(hipDeviceSynchronize());

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

    HIP_CHECK(hipEventRecord(sys_stop, 0));
    HIP_CHECK(hipEventSynchronize(sys_stop));

    float sys_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&sys_ms, sys_start, sys_stop));

    printf("Kernel-only time = %f μs\n", kernel_total_ms * 1000);
    printf("System time (host+device) = %f ms\n", sys_ms);

    return 0;
}
