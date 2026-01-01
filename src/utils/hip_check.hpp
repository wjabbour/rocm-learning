#pragma once

#include <hip/hip_runtime.h>

#define HIP_CHECK(call) check_hip_error((call), #call, __FILE__, __LINE__)

inline void check_hip_error(hipError_t error, const char* func_name, const char* file, int line) {
    if (error != hipSuccess) {
        fprintf(stderr, "\n[HIP ERROR] ------------------------------------------------\n");
        fprintf(stderr, "File: %s:%d\n", file, line);
        fprintf(stderr, "Code: %s (%d)\n", hipGetErrorName(error), (int)error);
        fprintf(stderr, "Msg:  %s\n", hipGetErrorString(error));
        fprintf(stderr, "Func: %s\n", func_name);
        fprintf(stderr, "------------------------------------------------------------\n");
        exit(EXIT_FAILURE);
    }
}

#define HIP_KERNEL_CHECK() check_kernel_launch(__FILE__, __LINE__)

inline void check_kernel_launch(const char* file, int line) {
    // Check for invalid launch arguments (grid dims, shared mem size, etc.)
    hipError_t err = hipGetLastError();
    
    if (err != hipSuccess) {
        fprintf(stderr, "\n[KERNEL LAUNCH ERROR] --------------------------------------\n");
        fprintf(stderr, "File: %s:%d\n", file, line);
        fprintf(stderr, "Code: %s (%d)\n", hipGetErrorName(err), (int)err);
        fprintf(stderr, "Msg:  %s\n", hipGetErrorString(err));
        fprintf(stderr, "------------------------------------------------------------\n");
        exit(EXIT_FAILURE);
    }
}