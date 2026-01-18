#include <hip/hip_runtime.h>
#include <iostream>
#include "utils/random_int.hpp"
#include "utils/hip_check.hpp"

// helper function to verify the result of the kernel
bool verifyResult(const std::vector<int>& a_h, const std::vector<int>& b_h, const std::vector<int>& c_h, int n) {
    for (int i = 0; i < n; i++) {
        if (c_h[i] != a_h[i] + b_h[i]) {
            return false;
        }
    }
    return true;
}

__global__ void addKernel(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    c[idx] = a[idx] + b[idx];
}

int main() {
    hipEvent_t sys_start, sys_stop;
    HIP_CHECK(hipEventCreate(&sys_start));
    HIP_CHECK(hipEventCreate(&sys_stop));

    HIP_CHECK(hipEventRecord(sys_start, 0));
    const int n = 1 << 20;
    size_t bytes = n * sizeof(int);

    // initialize host memory
    std::vector<int> a_h(n, 0);
    std::vector<int> b_h(n, 0);
    std::vector<int> c_h(n, 0);

    // allocate device memory
    int *a_d, *b_d, *c_d;
    HIP_CHECK(hipMalloc(&a_d, bytes));
    HIP_CHECK(hipMalloc(&b_d, bytes));
    HIP_CHECK(hipMalloc(&c_d, bytes));

    for (int i = 0; i < n; i++) {
        a_h[i] = Utils::Random::int_in_range(1, 10);
        b_h[i] = Utils::Random::int_in_range(1, 10);
    }

    // host -> device transfer
    HIP_CHECK(hipMemcpy(a_d, a_h.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_d, b_h.data(), bytes, hipMemcpyHostToDevice));
    // we dont need to copy the output buffer to the device because we are going to overwrite its contents

    hipEvent_t k_start, k_stop;
    HIP_CHECK(hipEventCreate(&k_start));
    HIP_CHECK(hipEventCreate(&k_stop));

    float kernel_total_ms = 0.0f;

    int block_size = 256;
    int block_count = (n + block_size - 1) / block_size;

    HIP_CHECK(hipEventRecord(k_start, 0));

    // launch kernel
    hipLaunchKernelGGL(addKernel, dim3(block_count), dim3(block_size), 0, 0, a_d, b_d, c_d, n);
    HIP_KERNEL_CHECK();
    HIP_CHECK(hipEventRecord(k_stop, 0));

    HIP_CHECK(hipEventSynchronize(k_stop));

    float k_μs = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&k_μs , k_start, k_stop));
    kernel_total_ms += k_μs;
    HIP_CHECK(hipDeviceSynchronize());

    // device -> host transfer
    HIP_CHECK(hipMemcpy(c_h.data(), c_d, bytes, hipMemcpyDeviceToHost));

    // free device memory immediately, verify after
    HIP_CHECK(hipFree(a_d));
    HIP_CHECK(hipFree(b_d));
    HIP_CHECK(hipFree(c_d));

    // vectors use the RAII principle, calling their destructors when the function scope ends

    HIP_CHECK(hipEventRecord(sys_stop, 0));
    HIP_CHECK(hipEventSynchronize(sys_stop));

    float sys_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&sys_ms, sys_start, sys_stop));

    printf("Kernel-only time = %f μs\n", kernel_total_ms * 1000);
    printf("System time (host+device) = %f ms\n", sys_ms);

    if (verifyResult(a_h, b_h, c_h, n)) {
        printf("Input processed correctly\n");
    } else {
        printf("Input processed incorrectly\n");
    }

    return 0;
}