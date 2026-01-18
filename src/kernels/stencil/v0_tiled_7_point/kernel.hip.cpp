#include <hip/hip_runtime.h>
#include "utils/hip_check.hpp"

#define HALO 1

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 8

#define TILE_X  (BLOCK_X + 2*HALO)
#define TILE_Y  (BLOCK_Y + 2*HALO)
#define TILE_Z  (BLOCK_Z + 2*HALO)

// TODO: add more thorough comments
__global__
void stencil3d7pt(const float* __restrict__ in,
                   float* __restrict__ out,
                   int nx, int ny, int nz)
{
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // Offset indicies within the tile
    int lx = threadIdx.x + HALO;
    int ly = threadIdx.y + HALO;
    int lz = threadIdx.z + HALO;

    // Block dimensions
    int bx = blockDim.x;
    int by = blockDim.y;
    int bz = blockDim.z;

    // Global indices
    int gx = blockIdx.x * bx + threadIdx.x;
    int gy = blockIdx.y * by + threadIdx.y;
    int gz = blockIdx.z * bz + threadIdx.z;

    // LDS tile with halo
    __shared__ float tile[TILE_Z][TILE_Y][TILE_X];

    /*
        this function is called by threads at the boundary of their respective
        blocks to load the respective halo (depending on which boundary we
        are at: x, y, z)
    */
    auto load_halo = [&] __device__ (int offset, char dim) {

        float val = 0.0f;

        // which index from the input should this halo load from?
        int target_idx = 0;
        if (dim == 'X')      target_idx = gx + offset;
        else if (dim == 'Y') target_idx = gy + offset;
        else                 target_idx = gz + offset;

        bool valid = false;
        int idx = -1;

        if (dim == 'X') {
            if (target_idx >= 0 && target_idx < nx) {
                idx = (gz * ny + gy) * nx + target_idx;
                valid = true;
            }
        }
        else if (dim == 'Y') {
            if (target_idx >= 0 && target_idx < ny) {
                idx = (gz * ny + target_idx) * nx + gx;
                valid = true;
            }
        }
        else { // Z
            if (target_idx >= 0 && target_idx < nz) {
                idx = (target_idx * ny + gy) * nx + gx;
                valid = true;
            }
        }

        // if the current thread is outside the input domain fallback to 0.0f
        val = valid ? in[idx] : 0.0f;

        // store
        if (dim == 'X')
            tile[lz][ly][lx + offset] = val;
        else if (dim == 'Y')
            tile[lz][ly + offset][lx] = val;
        else
            tile[lz + offset][ly][lx] = val;
    };

    bool in_bounds = (gx < nx) && (gy < ny) && (gz < nz);

    // every thread load its central neighbor into LDS
    tile[lz][ly][lx] = in_bounds ? in[(gz * ny + gy) * nx + gx] : 0.0f;

    // if we are at some boundary, load respective halo
    if (tx == 0) {
        for (int i = 1; i <= HALO; i++)
            load_halo(-i, 'X');
    }

    if (tx == blockDim.x - 1) {
        for (int i = 1; i <= HALO; i++)
            load_halo(+i, 'X');
    }

    if (ty == 0) {
        for (int i = 1; i <= HALO; i++)
            load_halo(-i, 'Y');
    }

    if (ty == blockDim.y - 1) {
        for (int i = 1; i <= HALO; i++)
            load_halo(+i, 'Y');
    }
    
    if (tz == 0) {
        for (int i = 1; i <= HALO; i++)
            load_halo(-i, 'Z');
    }

    if (tz == blockDim.z - 1) {
        for (int i = 1; i <= HALO; i++)
            load_halo(+i, 'Z');
    }

    __syncthreads();

    // OOB threads should only perform loads, not compute
    if (!in_bounds) return;

    out[(gz * ny + gy) * nx + gx] = 
        0.5f * tile[lz][ly][lx] +
        0.1f * (tile[lz][ly][lx - 1] +
                tile[lz][ly][lx + 1] +
                tile[lz][ly - 1][lx] +
                tile[lz][ly + 1][lx] +
                tile[lz - 1][ly][lx] +
                tile[lz + 1][ly][lx]);
}

int main() {
    const int nx = 128;
    const int ny = 128;
    const int nz = 128;

    const int n = nz * ny * nx;
    size_t size = n * sizeof(float);

    float *d_in, *d_out;

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    HIP_CHECK(hipMalloc(&d_in, size));
    HIP_CHECK(hipMalloc(&d_out, size));

    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                h_in[(z * ny + y) * nx + x] = static_cast<float>(x + y + z);
            }
        }
    }

    HIP_CHECK(hipMemcpy(d_in, h_in, size, hipMemcpyHostToDevice));

    dim3 block_dim(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid_dim((nx + block_dim.x - 1) / block_dim.x,
                 (ny + block_dim.y - 1) / block_dim.y,
                 (nz + block_dim.z - 1) / block_dim.z);

    hipLaunchKernelGGL(stencil3d7pt, 
                        grid_dim,
                        block_dim,
                        0,
                        0,
                        d_in,
                        d_out,
                        nx,
                        ny,
                        nz
                        );
    HIP_KERNEL_CHECK();

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out, d_out, size, hipMemcpyDeviceToHost));

    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                printf("%f\n", h_out[(z * ny + y) * nx + x]);
            }
        }
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    free(h_in);
    free(h_out);

    return 0;
}
