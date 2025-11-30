#include <hip/hip_runtime.h>
#include <iostream>

__global__ void pairwise_reduction(int* in, int* out, int N) {
    int sum = 0;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * 2;

    /*
        consider N = 5, tid 0 computes 0 + 1, tid 1 computes 2 + 3,
        index 4's partner of 5 is OOB, so we just need to pull 
        index 4 forward.
    */
    if (i < N) {
        if (i + 1 < N) {
            out[tid] = in[i] + in[i + 1];
        } else {
            out[tid] = in[i];
        }
    }
}

int main() {
    hipEvent_t start, stop;

    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start, 0);

    int N = 1 << 21;
    size_t size = N * sizeof(int);

    int blockSize = 64;

    int *in_d, *out_d;

    int *in_h = (int*)malloc(size);
    int *out_h = (int*)malloc(size);

    hipMalloc(&in_d, size);
    hipMalloc(&out_d, size);

    for (int i = 0; i < N; i++) {
        in_h[i] = i + 1;
        out_h[i] = 0;
    }

    hipMemcpy(in_d, in_h, size, hipMemcpyHostToDevice);
    hipMemcpy(out_d, out_h, size, hipMemcpyHostToDevice);

    int currentN = N;

    while (currentN > 1) {
        // lazily, launching too many threads. I only need currentN / 2
        int blockCount = (currentN + blockSize - 1) / blockSize;

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

        hipDeviceSynchronize();

        /*
            ping pong buffering

            we alloc a two fixed size buffers of data that we use throughout the 
            entirety of the kernel
        */
        std::swap(in_d, out_d);
        currentN = (currentN + 1) / 2;
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

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);

    std::cout << "Kernel time = " << ms << " ms\n";

    return 0;
}