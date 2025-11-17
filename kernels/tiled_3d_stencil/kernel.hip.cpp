#include <hip/hip_runtime.h>

#define HALO 1

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 8

#define TILE_X  (BLOCK_X + 2*HALO)
#define TILE_Y  (BLOCK_Y + 2*HALO)
#define TILE_Z  (BLOCK_Z + 2*HALO)

__global__
void stencil3d_7pt(const float* __restrict__ in,
                   float* __restrict__ out,
                   int NX, int NY, int NZ)
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

    bool in_bounds = (gx < NX) && (gy < NY) && (gz < NZ);

    if (in_bounds) {
        // Load central region
        tile[lz][ly][lx] = in[(gz * NY + gy) * NX + gx];

        // need to load halo regions as well

        // if we are on the left x-boundary of the block then we need to load the left halo
        if (tx == 0) {
            // if we are not on the global left boundary we can load from global memory
            if (gx > 0)
                tile[lz][ly][lx -1] = in[(gz * NY + gy) * NX + gx - 1];
            // else we are on the global left boundary so we would exceed bounds, set to 0
            else
                tile[lz][ly][lx -1] = 0.0f;
        }

        // if we are on the right x-boundary of the block then we need to load the right halo
        if (tx == bx - 1) {
            // if we are not on the global right boundary we can load from global memory
            if (gx < NX - 1)
                tile[lz][ly][lx + 1] = in[(gz * NY + gy) * NX + gx + 1];
            // else we are on the global right boundary so we would exceed bounds, set to 0
            else
                tile[lz][ly][lx + 1] = 0.0f;
        }

        // if we are on the bottom y-boundary of the block then we need to load the bottom halo
        if (ty == 0) {
            // if we are not on the global bottom boundary we can load from global memory
            if (gy > 0)
                tile[lz][ly - 1][lx] = in[(gz * NY + gy - 1) * NX + gx];
            // else we are on the global left boundary so we would exceed bounds, set to 0
            else
                tile[lz][ly - 1][lx] = 0.0f;
        }

        // if we are on the top y-boundary of the block then we need to load the top halo
        if (ty == by - 1) {
            // if we are not on the global top boundary we can load from global memory
            if (gy < NY - 1)
                tile[lz][ly + 1][lx] = in[(gz * NY + gy + 1) * NX + gx];
            // else we are on the global left boundary so we would exceed bounds, set to 0
            else
                tile[lz][ly + 1][lx] = 0.0f;
        }
        
        // if we are on the back z-boundary of the block then we need to load the back halo
        if (tz == 0) {
            // if we are not on the global back boundary we can load from global memory
            if (gz > 0)
                tile[lz - 1][ly][lx] = in[((gz - 1) * NY + gy) * NX + gx];
            // else we are on the global back boundary so we would exceed bounds, set to 0
            else
                tile[lz - 1][ly][lx] = 0.0f;
        }
        
        // if we are on the front z-boundary of the block then we need to load the front halo
        if (tz == bz - 1) {
            // if we are not on the global front boundary we can load from global memory
            if (gz < NZ - 1)
                tile[lz + 1][ly][lx] = in[((gz + 1) * NY + gy) * NX + gx];
            else
                tile[lz + 1][ly][lx] = 0.0f;
        }
    } else {
        // out of bounds threads set their tile values to 0
        tile[lz][ly][lx] = 0.0f;
    }

    __syncthreads();

    if (gx >= NX || gy >= NY || gz >= NZ) return;

    out[(gz * NY + gy) * NX + gx] = 
        0.5f * tile[lz][ly][lx] +
        0.1f * (tile[lz][ly][lx - 1] +
                tile[lz][ly][lx + 1] +
                tile[lz][ly - 1][lx] +
                tile[lz][ly + 1][lx] +
                tile[lz - 1][ly][lx] +
                tile[lz + 1][ly][lx]);
}

int main() {
    const int NX = 128;
    const int NY = 128;
    const int NZ = 128;

    const int N = NZ * NY * NX;
    size_t size = N * sizeof(float);

    float *in_d, *out_d;

    float *in_h = (float*)malloc(size);
    float *out_h = (float*)malloc(size);

    hipMalloc(&in_d, size);
    hipMalloc(&out_d, size);

    for (int z = 0; z < NZ; z++) {
        for (int y = 0; y < NY; y++) {
            for (int x = 0; x < NX; x++) {
                in_h[(z * NY + y) * NX + x] = static_cast<float>(x + y + z);
            }
        }
    }

    hipMemcpy(in_d, in_h, size, hipMemcpyHostToDevice);

    dim3 blockDim(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 gridDim((NX + blockDim.x - 1) / blockDim.x,
                 (NY + blockDim.y - 1) / blockDim.y,
                 (NZ + blockDim.z - 1) / blockDim.z);

    hipLaunchKernelGGL(stencil3d_7pt, 
                        gridDim,
                        blockDim, 
                        0,
                        0,
                        in_d,
                        out_d,
                        NX,
                        NY,
                        NZ
                        );

    hipDeviceSynchronize();
    hipMemcpy(out_h, out_d, size, hipMemcpyDeviceToHost);

    for (int z = 0; z < NZ; z++) {
        for (int y = 0; y < NY; y++) {
            for (int x = 0; x < NX; x++) {
                printf("%f\n", out_h[(z * NY + y) * NX + x]);
            }
        }
    }

    hipFree(in_d);
    hipFree(out_d);
    free(in_h);
    free(out_h);

    return 0;
}
