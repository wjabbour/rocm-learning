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
    size_t bytes = N * sizeof(int);

    // initialize host memory
    std::vector<int> A_h(N, 0);
    std::vector<int> B_h(N, 0);
    std::vector<int> C_h(N, 0);

    // allocate device memory
    int *A_d, *B_d, *C_d;
    hipMalloc(&A_d, bytes);
    hipMalloc(&B_d, bytes);
    hipMalloc(&C_d, bytes);

    for (int i = 0; i < N; i++) {
        A_h[i] = i;
        B_h[i] = i*2;
    }

    hipMemcpy(A_d, A_h.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h.data(), bytes, hipMemcpyHostToDevice);
    // we dont need to copy the output buffer to the device because we are going to overwrite it's contents

    int blockSize = 256;
    int blockCount = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(add_kernel, dim3(blockCount), dim3(blockSize), 0, 0, A_d, B_d, C_d, N);
    hipDeviceSynchronize();

    hipMemcpy(C_h.data(), C_d, bytes, hipMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << C_h[i] << "\n";
    }

    free(A_d);
    free(B_d);
    free(C_d);

    // vectors use the RAII principle, calling their destructors when the function scope ends

    return 0;
}
