#include <hip/hip_runtime.h>
#include <iostream>

__global__ void pairwise_reduction(int* in, int* out, int N) {
    int sum = 0;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * 2;

    /*
        consider N = 5, tid 0 computes 0 + 1, tid 1 computes 2 + 3,
        index 4's partner of 5 is OOB, so we just need to pull 
        index 4 forward.
    */
    if (i < N) {
        if (i + 1 < N) {
            out[tid] = in[2*tid] + in[2*tid + 1];
        } else {
            out[tid] = in[2*tid];
        }
    }
}

int main() {
    int N = 10;
    size_t size = N * sizeof(int);

    int blockSize = 64;

    int *in_d, *out_d;

    int *in_h = (int*)malloc(size);
    int *out_h = (int*)malloc(size);

    hipMalloc(&in_d, size);
    hipMalloc(&out_d, size);

    for (int i = 0; i < N; i++) {
        in_h[i] = i + 1;
        out_h[i] = 0;
    }

    hipMemcpy(in_d, in_h, size, hipMemcpyHostToDevice);
    hipMemcpy(out_d, out_h, size, hipMemcpyHostToDevice);

    int currentN = N;

    while (currentN > 1) {
        int blockCount = (currentN + blockSize - 1) / blockSize;

        hipLaunchKernelGGL(
            pairwise_reduction,
            dim3(blockCount),
            dim3(blockSize), 
            0,
            0,
            in_d,
            out_d,
            currentN
        );

        hipDeviceSynchronize();

        int* tmp = in_d;
        in_d = out_d;
        out_d = tmp;

        currentN /= 2;
    }

    hipMemcpy(out_h, out_d, size, hipMemcpyDeviceToHost);

    std::cout << out_h[0] << "\n";
    std::cout << out_h[1] << "\n";
    std::cout << out_h[2] << "\n";
    std::cout << out_h[3] << "\n";
    std::cout << out_h[4] << "\n";

    hipFree(in_d);
    hipFree(out_d);
    free(in_h);
    free(out_h);

    return 0;
}