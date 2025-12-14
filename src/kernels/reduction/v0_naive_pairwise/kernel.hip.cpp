#include <hip/hip_runtime.h>
#include <iostream>

/*
    My very first kernel

    Receives a pointer to input A, input B, and output C.

    A and B both represent 1D input arrays and C represents a 1D output array.

    We launch N threads to process input of size N, where N = A.length = B.length

    Thus, each thread computes its global index in the grid and access A[idx] and B[idx],
    performs some calculation (in this case, addition) and writes to C[idx]
*/
__global__ void add_kernel(int* A, int* B, int* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > N) return;

    C[idx] = A[idx] + B[idx];
}

int main() {
    const int N = 1 << 20;
    size_t size = N * sizeof(int);

    int blockSize = 256;
    int blockCount = (N + blockSize - 1) / blockSize;

    int *A_d, *B_d, *C_d;

    int *A_h = (int*)malloc(size);
    int *B_h = (int*)malloc(size);
    int *C_h = (int*)malloc(size);

    hipMalloc(&A_d, size);
    hipMalloc(&B_d, size);
    hipMalloc(&C_d, size);

    for (int i = 0; i < N; i++) {
        A_h[i] = i;
        B_h[i] = i*2;
    }

    hipMemcpy(A_d, A_h, size, hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h, size, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(add_kernel, dim3(blockCount), dim3(blockSize), 0, 0, A_d, B_d, C_d, N);
    hipDeviceSynchronize();

    hipMemcpy(C_h, C_d, size, hipMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << C_h[i] << "\n";
    }

    hipFree(A_h);
    hipFree(B_h);
    hipFree(C_h);
    free(A_d);
    free(B_d);
    free(C_d);

    return 0;
}
