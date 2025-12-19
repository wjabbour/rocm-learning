#include <hip/hip_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void matmul(float* A, float* B, float* C, int width) {
    // a TILE_SIZE by TILE_SIZE LDS tile we will use to store the contents of input matrix A that our block will operate on
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    // a TILE_SIZE by TILE_SIZE LDS tile we will use to store the contents of input matrix B that our block will operate on
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // coordinates of the thread within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // coordinates of the thread within the grid
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    float sum = 0.0f;
    /*
        each block of threads walks horizontally along input matrix A and vertically along input matrix B

         input matrices A and B are given as 1D contiguous memory blocks stored in row-major order
    */
    for (int k = 0; k < width; k += TILE_SIZE) {
        // all threads in the wavefront request contiguous memory addresses from input matrix A
        int aIdx = (row * width) + tx + k;
        /*
            since the input is stored in row major order, we cannot fetch a column of data
            from B without executing N memory transactions (where N = size of column) per wavefront. To improve
            memory coalescense, threads in the wavefront collaboratively load in a row of length TILE_SIZE from B, all
            wavefronts in the block working together to load the full tile from B across TILE_SIZE memory transactions.
        */
        int bIdx = (k + ty) * width + col;

        // now that we've done the mental gymnastics to identify which piece of data to pull from A and B...
        tile_A[ty][tx] = A[aIdx];
        tile_B[ty][tx] = B[bIdx];

        // before any wavefront can work with the data in LDS, we must ensure that all wavefronts have finished writing to LDS
        __syncthreads();

        /*
            although the threads of the block work together to collaboratively load LDS, once it's time to calculate, each thread
            simply needs to walk horizontally along A and vertically along B to perform the matrix multiplication.

            Notice that we are now able to walk along the column in our B tile. We don't have to worry about coalescing our memory
            reads any longer. Access to LDS is extremely fast for each thread.
        */
        for (int j = 0; j < TILE_SIZE; j++) {
           sum += tile_A[ty][j] * tile_B[j][tx];
        }

        // ensure that wavefronts in our block do not begin altering LDS data until all of the wavefronts have completed their calculations
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
