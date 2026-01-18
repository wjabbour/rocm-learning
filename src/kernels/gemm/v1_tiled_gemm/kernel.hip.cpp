#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define TILE_SIZE 16

__global__ void matrix_multiply(float* A, float* B, float* C, int M, int N, int K) {
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
    for (int k = 0; k < K; k += TILE_SIZE) {
        /*
            as we iterate k across the shared dimension K, we walk horizontally along A and vertically along B

            we hold the thread's global row constant and use k + tx to determine our column in A
            we hold the thread's global col constant and use k + ty to determine our row in B
        */
        int aCol = k + tx;
        int bRow = k + ty;

        // all threads in the wavefront request contiguous memory addresses from input matrix A
        int aIdx = row * K + aCol;
        /*
            since the input is stored in row major order, we cannot fetch a column of data
            from B without executing N memory transactions (where N = size of column) per wavefront. To improve
            memory coalescense, threads in the wavefront collaboratively load in a row of length TILE_SIZE from B, all
            wavefronts in the block working together to load the full tile from B across TILE_SIZE memory transactions.
        */
        int bIdx = bRow * N + col;

        // now that we've done the mental gymnastics to identify which piece of data to pull from A and B...

        /*
            we must load the data into LDS, but we need to be careful in the cases that our grid size, input size, and LDS size
            don't share a common divisor.

            For example, if we launch a 16x16 block, but A is a 1x2 matrix, then only 2 out of 256 threads will be able to read from A. 
            We ensure that our thread is within the dimensions of A, and if not we pad LDS with 0s.
        */
        if (row < M && aCol < K) {
            tile_A[ty][tx] = A[aIdx];
        } else {
            tile_A[ty][tx] = 0.0f;
        }

        // same idea here, except the dimensions of B are KxN
        if (col < N && bRow < K) {
            tile_B[ty][tx] = B[bIdx];
        } else {
            tile_B[ty][tx] = 0.0f;
        }

        // before any wavefront can work with the data in LDS, we must ensure that all wavefronts have finished writing to LDS
        __syncthreads();

        /*
            although the threads of the block work together to collaboratively load LDS, once it's time to calculate, each thread
            simply needs to walk horizontally along tile A and vertically along tile B to perform the matrix multiplication.

            Notice that we are now able to walk along the column in our B tile. We don't have to worry about coalescing our memory
            reads any longer. Access to LDS is extremely fast for each thread.
        */
        for (int j = 0; j < TILE_SIZE; j++) {
            sum += tile_A[ty][j] * tile_B[j][tx];
        }

        // ensure that wavefronts in our block do not begin altering LDS data until all of the wavefronts have completed their calculations
        __syncthreads();
    }

    // write to output in row major format
    C[row * N + col] = sum;
}

int main() {
    /*
        we need to specify the dimensions of our matrices so that the kernel
        is generalizable to matrices of arbitrary dimensions
    */
    int M = 2;
    int N = 3;
    int K = 2;

    // given by the definition of matrix multiplication
    const size_t bytesA = M * K * sizeof(float);
    const size_t bytesB = K * N * sizeof(float);
    const size_t bytesC = M * N * sizeof(float);

    // initialize host memory
    std::vector<float> A_h = {1.0f, 2.0f, 2.0f, 4.0f};
    std::vector<float> B_h = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> C_h(M * N, 0.0f);
 
    // allocate device memory
    float *A_d, *B_d, *C_d;
    hipMalloc(&A_d, bytesA);
    hipMalloc(&B_d, bytesB);
    hipMalloc(&C_d, bytesC);

    // host -> device transfer
    hipMemcpy(A_d, A_h.data(), bytesA, hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h.data(), bytesB, hipMemcpyHostToDevice);
    // we dont need to copy the output buffer to the device because we are going to overwrite it's contents

    /*
        with this implementation, each thread in the block collaboritvely loads a piece of data
        into LDS, therefore the block dimensions exactly match the LDS data dimensions
    */
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    /*
        Matrix A is MxK, Matrix B is KxN, thus the output is MxN (M rows, N columns)

        The dim3 struct positional arguments are x, y, then z. Since x and N both correspond
        with horizontal movement and y and M both correspond with vertical movement, we must
        be careful to initialize the grid as NxM threads to unify threads to matrix indices.
    */
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    hipLaunchKernelGGL(matrix_multiply, dim3(numBlocks), dim3(blockSize), 0, 0, A_d, B_d, C_d, M, N, K);
    hipDeviceSynchronize();

    hipMemcpy(C_h.data(), C_d, bytesC, hipMemcpyDeviceToHost);

    // TODO: replace this with a smaller verification once we increase the input matrix sizes
    for (int i = 0; i < M * N; i++) {
        std::cout << C_h[i] << "\n";
    }

    hipFree(A_d);
    hipFree(B_d);
    hipFree(C_d);

    // vectors use the RAII principle, calling their destructors when the function scope ends

    return 0;
}
