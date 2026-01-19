#include <hip/hip_runtime.h>
#include "utils/random_int.hpp"
#include "utils/hip_check.hpp"
#include "utils/wave_utils.hpp"

#define BLOCK_SIZE 512

__global__ void blockReduction(int* in, int* out, size_t n) {
    /*
        each wavefront will write one element to LDS
        
        we cannot use warpSize here since variables cannot be used for memory allocation.
        However, the code is still functionally correct on systems with wavefronts of size 64,
        we will just be over allocating LDS.
    */
    __shared__ float wavefront_sums[BLOCK_SIZE / 32];

    // we may launch more than 2^32 threads, so we need to use size_t for our global thread ID
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    /*
        the first wavefront of the block will write the final value to global memory. This 
        choice is arbitrary: any wavefront is capable of this step.
    */ 
    int wf_id = threadIdx.x / warpSize;
    // the thread at lane 0 will hold the final value from our wave-shuffle reduction
    int lane_id = threadIdx.x % warpSize;

    // load data from global memory, contiguous threads access contiguous memory
    int data = in[tid];

    // sum all values in the wavefront
    size_t wave_sum = waveReduceSum(data);

    // one thread from each wavefront writes the wavefront sum to LDS
    if (lane_id == 0) {
        wavefront_sums[wf_id] = wave_sum;
    }

    // ensure all wavefronts have written their values to LDS before proceeding
    __syncthreads();

    /*
        now we need a thread to sum the sums in LDS and write to global memory

        any thread could do this, but I'm choosing the first thread in the grid
    */
    if (wf_id == 0 && lane_id == 0) {
        int block_sum = 0;
        // when warpSize is 64, ensure we don't attempt to read uninitialized LDS entries
        int wavefront_count = BLOCK_SIZE / warpSize;
        for (int i = 0; i < wavefront_count; i++) {
            block_sum += wavefront_sums[i];
        }

        out[blockIdx.x] = block_sum;
    }
}

int main() {
    hipEvent_t sys_start, sys_stop;
    HIP_CHECK(hipEventCreate(&sys_start));
    HIP_CHECK(hipEventCreate(&sys_stop));

    HIP_CHECK(hipEventRecord(sys_start, 0));

    //size_t n = 1ULL << 31;
    size_t n = 1<<10;
    size_t input_bytes = n * sizeof(int);

    // each block will produce a single output value, so our initial output size is equal to the number of blocks launched
    size_t output_elements = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t output_bytes = output_elements * sizeof(int);
    
    // allocate host memory
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

    // allocate device memory
    HIP_CHECK(hipMalloc(&d_in, input_bytes));
    HIP_CHECK(hipMalloc(&d_out, output_bytes));

    printf("Allocated device arrays\n");

    // initialize host memory
    for (size_t i = 0; i < n; i++) {
        h_in[i] = Utils::Random::int_in_range(1, 1);
    }

    printf("Populated input array\n");

    // host -> device transfer
    HIP_CHECK(hipMemcpy(d_in, h_in, input_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out, h_out, output_bytes, hipMemcpyHostToDevice));

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
