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
    const int N = 1 << 5;
    size_t size = N * sizeof(int);

    int blockSize = 64;
    int blockCount = (N + blockSize - 1) / blockSize;

    int *A_d, *B_d;

    int *A_h = (int*)malloc(size);
    int *B_h = (int*)malloc(size);

    hipMalloc(&A_d, size);
    hipMalloc(&B_d, size);

    for (int i = 0; i < N; i++) {
        A_h[i] = i + 1;
        B_h[i] = 0;
    }

    hipMemcpy(A_d, A_h, size, hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h, size, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(
        partial_reduction,
        dim3(blockCount),
        dim3(blockSize), 
        0,
        0,
        A_d,
        B_d,
        N
    );

    hipDeviceSynchronize();

    hipMemcpy(B_h, B_d, size, hipMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << B_h[i] << "\n";
    }

    hipFree(A_d);
    hipFree(B_d);
    free(A_h);
    free(B_h);

    return 0;
}
