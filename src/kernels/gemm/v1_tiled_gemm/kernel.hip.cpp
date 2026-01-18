#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include "utils/hip_check.hpp"

#define TILE_SIZE 16

__global__ void matrixMultiply(float* a, float* b, float* c, int m, int n, int k) {
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
    for (int k_idx = 0; k_idx < k; k_idx += TILE_SIZE) {
        /*
            as we iterate k across the shared dimension k, we walk horizontally along A and vertically along B

            we hold the thread's global row constant and use k + tx to determine our column in A
            we hold the thread's global col constant and use k + ty to determine our row in B
        */
        int a_col = k_idx + tx;
        int b_row = k_idx + ty;

        // all threads in the wavefront request contiguous memory addresses from input matrix A
        int a_idx = row * k + a_col;
        /*
            since the input is stored in row major order, we cannot fetch a column of data
            from B without executing n memory transactions (where n = size of column) per wavefront. To improve
            memory coalescense, threads in the wavefront collaboratively load in a row of length TILE_SIZE from B, all
            wavefronts in the block working together to load the full tile from B across TILE_SIZE memory transactions.
        */
        int b_idx = b_row * n + col;

        // now that we've done the mental gymnastics to identify which piece of data to pull from A and B...

        /*
            we must load the data into LDS, but we need to be careful in the cases that our grid size, input size, and LDS size
            don't share a common divisor.

            For example, if we launch a 16x16 block, but A is a 1x2 matrix, then only 2 out of 256 threads will be able to read from A. 
            We ensure that our thread is within the dimensions of A, and if not we pad LDS with 0s.
        */
        if (row < m && a_col < k) {
            tile_A[ty][tx] = a[a_idx];
        } else {
            tile_A[ty][tx] = 0.0f;
        }

        // same idea here, except the dimensions of B are kxn
        if (col < n && b_row < k) {
            tile_B[ty][tx] = b[b_idx];
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
    c[row * n + col] = sum;
}

int main() {
    /*
        we need to specify the dimensions of our matrices so that the kernel
        is generalizable to matrices of arbitrary dimensions
    */
    int m = 2;
    int n = 3;
    int k = 2;

    // given by the definition of matrix multiplication
    const size_t bytes_a = m * k * sizeof(float);
    const size_t bytes_b = k * n * sizeof(float);
    const size_t bytes_c = m * n * sizeof(float);

    // initialize host memory
    std::vector<float> h_a = {1.0f, 2.0f, 2.0f, 4.0f};
    std::vector<float> h_b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> h_c(m * n, 0.0f);
 
    // allocate device memory
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes_a));
    HIP_CHECK(hipMalloc(&d_b, bytes_b));
    HIP_CHECK(hipMalloc(&d_c, bytes_c));

    // host -> device transfer
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes_a, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes_b, hipMemcpyHostToDevice));
    // we dont need to copy the output buffer to the device because we are going to overwrite it's contents

    /*
        with this implementation, each thread in the block collaboritvely loads a piece of data
        into LDS, therefore the block dimensions exactly match the LDS data dimensions
    */
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    /*
        Matrix A is mxk, Matrix B is kxn, thus the output is mxn (m rows, n columns)

        The dim3 struct positional arguments are x, y, then z. Since x and n both correspond
        with horizontal movement and y and m both correspond with vertical movement, we must
        be careful to initialize the grid as nxm threads to unify threads to matrix indices.
    */
    dim3 num_blocks((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    hipLaunchKernelGGL(matrixMultiply, dim3(num_blocks), dim3(block_size), 0, 0, d_a, d_b, d_c, m, n, k);
    HIP_KERNEL_CHECK();
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes_c, hipMemcpyDeviceToHost));

    // TODO: replace this with a smaller verification once we increase the input matrix sizes
    for (int i = 0; i < m * n; i++) {
        std::cout << h_c[i] << "\n";
    }

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    // vectors use the RAII principle, calling their destructors when the function scope ends

    return 0;
}
