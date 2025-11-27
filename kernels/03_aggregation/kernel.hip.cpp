#include <hip/hip_runtime.h>
#include <iostream>

__global__ void partial_reduction(int* in, int* out, int N) {
    int sum = 0;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
        sum += in[i];
    }

    out[tid] = sum;
}

int main() {
    const int N = 10;
    size_t size = N * sizeof(int);

    int blockSize = 64;
    int blockCount = (N + blockSize - 1) / blockSize;

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

    hipLaunchKernelGGL(
        partial_reduction,
        dim3(blockCount),
        dim3(blockSize), 
        0,
        0,
        in_d,
        out_d,
        N
    );

    hipDeviceSynchronize();

    
    hipLaunchKernelGGL(
        partial_reduction,
        dim3(blockCount),
        dim3(blockSize / 2), 
        0,
        0,
        in_d,
        out_d,
        blockCount * blockSize
    );

    hipDeviceSynchronize();


    hipMemcpy(out_h, out_d, size, hipMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << out_h[i] << "\n";
    }

    hipFree(in_d);
    hipFree(out_d);
    free(in_h);
    free(out_h);

    return 0;
}
