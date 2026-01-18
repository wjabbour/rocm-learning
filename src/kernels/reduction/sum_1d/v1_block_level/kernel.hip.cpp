#include <hip/hip_runtime.h>
#include "utils/random_int.hpp"
#include "utils/hip_check.hpp"

#define BLOCK_SIZE 512

__global__ void blockReduction(int* in, int* out, size_t n) {
    __shared__ float wavefront[8];

    // we may launch more than 2^32 threads, so we need to use size_t for our global thread ID
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    /*
        the first wavefront of the block will write the final value to global memory. This 
        choice is arbitrary: any wavefront is capable of this step.
    */ 
    int wf_id = blockIdx.x / 32;
    // the thread at lane 0 will hold the final value from our wave-shuffle reduction
    int lane_id = threadIdx.x % 32;

    // load data from global memory, contiguous threads access contiguous memory
    int data = in[tid];

    // sum all values in the wavefront
    size_t wave_sum = waveReduceSum(data);
}

int main() {
    hipEvent_t sys_start, sys_stop;
    HIP_CHECK(hipEventCreate(&sys_start));
    HIP_CHECK(hipEventCreate(&sys_stop));

    HIP_CHECK(hipEventRecord(sys_start, 0));

    size_t n = 1ULL << 31;
    size_t input_bytes = n * sizeof(int);

    // each block will produce a single output value, so our initial output size is equal to the number of blocks launched
    size_t output_elements = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t output_bytes = output_elements * sizeof(int);
    
    // allocate host memory
    int *in_d, *out_d;
    int *in_h = (int*)malloc(input_bytes);
    if (in_h == NULL) {
        printf("Input array host memory allocation failed\n");
        exit(1);
    }

    int* out_h = (int*)calloc(output_elements, sizeof(int));
    if (out_h == NULL) {
        printf("Output array host memory allocation failed\n");
        exit(1);
    }

    printf("Allocated input and output arrays\n");

    // allocate device memory
    HIP_CHECK(hipMalloc(&in_d, input_bytes));
    HIP_CHECK(hipMalloc(&out_d, output_bytes));

    printf("Allocated device arrays\n");

    // initialize host memory
    for (size_t i = 0; i < n; i++) {
        in_h[i] = Utils::Random::int_in_range(1, 1);
    }

    printf("Populated input array\n");

    // host -> device transfer
    HIP_CHECK(hipMemcpy(in_d, in_h, input_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(out_d, out_h, output_bytes, hipMemcpyHostToDevice));

    size_t current_n = n;

    hipEvent_t k_start, k_stop;
    HIP_CHECK(hipEventCreate(&k_start));
    HIP_CHECK(hipEventCreate(&k_stop));

    float kernel_total_ms = 0.0f;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    while (current_n > 1) {
        int output_size = (current_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int block_count = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if (size_t(block_count * BLOCK_SIZE) > (size_t)prop.maxGridSize[0]) {
            printf("CRITICAL ERROR: Needed %lu blocks, but GPU Max is %d\n", block_count, prop.maxGridSize[0]);
            exit(1);
        }

        HIP_CHECK(hipEventRecord(k_start, 0));

        hipLaunchKernelGGL(
            blockReduction,
            dim3(block_count),
            dim3(BLOCK_SIZE), 
            0,
            0,
            in_d,
            out_d,
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
        std::swap(in_d, out_d);
        current_n = output_size;
    }

    /*
        since we swap the pointers after every kernel launch, in_d will
        always hold the most recent output data. I swap once more here because
        it just feels "right" to memcpy from device output pointer :) 
    */
    std::swap(in_d, out_d);
    HIP_CHECK(hipMemcpy(out_h, out_d, sizeof(int), hipMemcpyDeviceToHost));

    // the answer!
    printf("Result: %i\n", out_h[0]);

    HIP_CHECK(hipFree(in_d));
    HIP_CHECK(hipFree(out_d));
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
