#include <hip/hip_runtime.h>
#include "utils/random_int.hpp"
#include "utils/hip_check.hpp"

#define WORK_PER_THREAD 8

__global__ void halvingReduction(int* in, int* out, size_t n) {
    // we may launch more than 2^32 threads, so we need to use size_t for our global thread ID
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    /*
        with each kernel launch, we launch {current_value_of_n} / WORK_PER_THREAD threads
        since we are launching less threads than n, each thread needs to be responsible for 
        summing WORK_PER_THREAD units of n
    */
    size_t i = tid * WORK_PER_THREAD;

    /*
        our thread must be within n, else we will pull garbage data
    */
    if (i < n) {
        int sum = 0;
        /*
            once we know that we are on a thread that is within n, we need to find
            how many indices to the right (up to WORK_PER_THREAD) are also within n

            we start at the rightmost index (i + WORK_PER_THREAD), walking left until we find that the index is
            within n and use the inner loop to sum the indices from there to i
        */
        for (int j = WORK_PER_THREAD - 1; j >= 0; j--) {
            if (i + j < n) {
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

    size_t n = 1ULL << 31;
    size_t input_bytes = n * sizeof(int);

    // the output buffer is smaller than n since we aggregate the data in the kernel
    size_t output_elements = (n + WORK_PER_THREAD - 1) / WORK_PER_THREAD;
    size_t output_bytes = output_elements * sizeof(int);
    int *d_in, *d_out;

    int *h_in = (int*)malloc(input_bytes);
    if (h_in == NULL) {
        printf("Input array host memory allocation failed\n");
        exit(1);
    }

    int* h_out = (int*)calloc(output_elements, sizeof(int));
    if (h_out == NULL) {
        printf("Output array host memory allocation failed\n");
        exit(1);
    }

    printf("Allocated input and output arrays\n");

    HIP_CHECK(hipMalloc(&d_in, input_bytes));
    HIP_CHECK(hipMalloc(&d_out, output_bytes));

    printf("Allocated device arrays\n");

    for (size_t i = 0; i < n; i++) {
        h_in[i] = Utils::Random::int_in_range(1, 10);
    }

    printf("Populated input and output arrays\n");

    HIP_CHECK(hipMemcpy(d_in, h_in, input_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out, h_out, output_bytes, hipMemcpyHostToDevice));

    size_t current_n = n;

    hipEvent_t k_start, k_stop;
    HIP_CHECK(hipEventCreate(&k_start));
    HIP_CHECK(hipEventCreate(&k_stop));

    float kernel_total_ms = 0.0f;

    int block_size = 64;
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    while (current_n > 1) {
        int output_size = (current_n + WORK_PER_THREAD - 1) / WORK_PER_THREAD;
        int block_count = (output_size + block_size - 1) / block_size;

        if (size_t(block_count * block_size) > (size_t)prop.maxGridSize[0]) {
            printf("CRITICAL ERROR: Needed %lu blocks, but GPU Max is %d\n", block_count, prop.maxGridSize[0]);
            exit(1);
        }

        HIP_CHECK(hipEventRecord(k_start, 0));

        hipLaunchKernelGGL(
            halvingReduction,
            dim3(block_count),
            dim3(block_size), 
            0,
            0,
            d_in,
            d_out,
            current_n
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
        std::swap(d_in, d_out);
        current_n = output_size;
    }

    /*
        since we swap the pointers after every kernel launch, d_in will
        always hold the most recent output data. I swap once more here because
        it just feels "right" to memcpy from device output pointer :) 
    */
    std::swap(d_in, d_out);
    HIP_CHECK(hipMemcpy(h_out, d_out, sizeof(int), hipMemcpyDeviceToHost));

    // the answer!
    printf("Result: %i\n", h_out[0]);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    free(h_in);
    free(h_out);

    HIP_CHECK(hipEventRecord(sys_stop, 0));
    HIP_CHECK(hipEventSynchronize(sys_stop));

    float sys_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&sys_ms, sys_start, sys_stop));

    printf("Kernel-only time = %f μs\n", kernel_total_ms * 1000);
    printf("System time (host+device) = %f ms\n", sys_ms);

    return 0;
}
