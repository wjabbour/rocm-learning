#include <hip/hip_runtime.h>
#include <iostream>
#include "utils/random_int.hpp"

// helper function to verify the result of the kernel
bool verify_result(const std::vector<int>& A_h, const std::vector<int>& B_h, const std::vector<int>& C_h, int N) {
    for (int i = 0; i < N; i++) {
        if (C_h[i] != A_h[i] + B_h[i]) {
            return false;
        }
    }
    return true;
}

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

    if (idx >= N) return;

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
        A_h[i] = Utils::Random::int_in_range(1, 10);
        B_h[i] = Utils::Random::int_in_range(1, 10);
    }

    // host -> device transfer
    hipMemcpy(A_d, A_h.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h.data(), bytes, hipMemcpyHostToDevice);
    // we dont need to copy the output buffer to the device because we are going to overwrite it's contents

    int blockSize = 256;
    int blockCount = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(add_kernel, dim3(blockCount), dim3(blockSize), 0, 0, A_d, B_d, C_d, N);
    hipDeviceSynchronize();

    hipMemcpy(C_h.data(), C_d, bytes, hipMemcpyDeviceToHost);

    hipFree(A_d);
    hipFree(B_d);
    hipFree(C_d);

    // vectors use the RAII principle, calling their destructors when the function scope ends

    if (verify_result(A_h, B_h, C_h, N)) {
        printf("Input processed correctly\n");
    } else {
        printf("Input processed incorrectly\n");
    }

    return 0;
}