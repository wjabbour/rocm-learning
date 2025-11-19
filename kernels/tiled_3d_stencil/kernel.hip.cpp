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

    auto load_halo = [&] __device__ (int offset, char dim) {

        float val = 0.0f;

        // Compute neighbor global coordinate in chosen dimension
        int ng = 0;
        if (dim == 'X')      ng = gx + offset;
        else if (dim == 'Y') ng = gy + offset;
        else                 ng = gz + offset;

        bool valid = false;
        int idx = -1;

        // Compute correct global memory index
        if (dim == 'X') {
            if (ng >= 0 && ng < NX) {
                idx = (gz * NY + gy) * NX + ng;
                valid = true;
            }
        }
        else if (dim == 'Y') {
            if (ng >= 0 && ng < NY) {
                idx = (gz * NY + ng) * NX + gx;
                valid = true;
            }
        }
        else { // Z
            if (ng >= 0 && ng < NZ) {
                idx = (ng * NY + gy) * NX + gx;
                valid = true;
            }
        }

        val = valid ? in[idx] : 0.0f;

        // Store into LDS tile
        if (dim == 'X')
            tile[lz][ly][lx + offset] = val;
        else if (dim == 'Y')
            tile[lz][ly + offset][lx] = val;
        else
            tile[lz + offset][ly][lx] = val;
    };

    bool in_bounds = (gx < NX) && (gy < NY) && (gz < NZ);

    if (in_bounds) {
        tile[lz][ly][lx] = in[(gz * NY + gy) * NX + gx];

        if (tx == 0) {
            for (int i = 1; i <= HALO; i++)
                load_halo(-i, 'X');
        }

        /*
        
            the issue i have is with block overhang..

            to support halo greater than 1 i need to address the fact that

            the tx may not be equal to bx - 1 but tx + HALO > NX

            bx = 8, tx = 6 halo = 2

        */
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

    } else {
        // out of bounds threads set their tile values to 0
        tile[lz][ly][lx] = 0.0f;
    }

    __syncthreads();

    if (!in_bounds) return;

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
