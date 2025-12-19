#include <hip/hip_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void matmul(float* A, float* B, float* C, int width) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    float sum = 0.0f;
    // i is 0 through N - 1
    for (int k = 0; k < width; k += TILE_SIZE) {
        int aIdx = (row * width) + tx + k;
        int bIdx = (k + ty) * width + col;

        tile_A[ty][tx] = A[aIdx];
        tile_B[ty][tx] = B[bIdx];

        __syncthreads();


        // j is 0 through 15
        for (int j = 0; j < TILE_SIZE; j++) {
           sum += tile_A[ty][j] * tile_B[j][tx];
        }

        __syncthreads();
    }

    C[row * width + col] = sum;
}

int main() {
    const int N = 4;
    size_t size = N * sizeof(float);

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
