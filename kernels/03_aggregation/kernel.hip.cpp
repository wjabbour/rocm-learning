#include <hip/hip_runtime.h>
#include <iostream>

__global__ void partial_reduction(int* in, int* out, int N) {
    float sum 0.0f;

    int tid = gridDim.x * blockDim.x + threadIdx.x;

    for (let i = tid; i < N; i += gridDim.x * blockDim.x) {
        sum += in[i];
    }

    out[tid] = sum;
}

int main() {
    const int N = 10
    size_t size = N * sizeof(int);

    int blockSize = 2;
    int blockCount = (N + blockSize - 1) / blockSize;

    int *A_d, *B_d;

    int *A_h = (int*)malloc(size);
    int *B_h = (int*)malloc(size);

    hipMalloc(&A_d, size);
    hipMalloc(&B_d, size);

    for (int i = 0; i < N; i++) {
        A_h[i] = i + 1;
    }

    hipMemcpy(A_d, A_h, size, hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h, size, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(
        add_kernel,
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
