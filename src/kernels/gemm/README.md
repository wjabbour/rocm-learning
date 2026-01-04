# Overview

General Matrix Multiply (GEMM) kernels are a core piece of training and inference pipelines. A GEMM kernel is responsible for efficiently multiplying two matrices together to produce an output matrix.

## Build Instructions

```bash
make clean && make run SRC=src/kernels/gemm/v0_naive_gemm/kernel.hip.cpp
```

# Learnings

- When requesting data from the memory system (L1, L2, HBM) it is important to ensure that the threads of a wavefront request contiguous data so that the memory controller can coalesce the requests from each thread into as few transactions as possible.

It's important to note that a motivation for this design is the fact that GPU memory is optimized to move data in large chunks, called bursts. These bursts are aligned to the size of the cache lines, typically 128 bytes on RDNA GPUs. The fact that the memory system is capable of bursts means that we can attempt to use as few bursts as possible to transfer the data to our GPU.

- LDS on the other hand is only shared by the wavefronts of a block. It is organized at the hardware level into banks where the number of banks matches the number of SIMD lanes.

When requesting data from LDS, it is important to ensure that the threads of a wavefront either:

1) all request the same memory address. This executes a special hardware-enabled operation which broadcasts the value of that memory address to all lanes in the SIMD.
2) all request from a distinct bank, i.e. two threads do not attempt to access the same LDS bank within the same clock cycle.

If either of these rules are violated, the bank access between the N colliding threads will be serialized across clock cycles since this shared resource (the bank) cannot be accessed by more than one thread simultaneously. This directly degrades performance.

The next logical question is then: "When a SIMD lane requests memory from the LDS, how does the memory system map that request to a specific LDS bank?"

The answer is simple: Each SIMD lane requests a memory address from the LDS and a modulo operation is applied to each memory address to map it to a specific LDS bank.

Importantly, since the memory address decides which bank the data is stored on, changes in the data type of the LDS tile can change the mapping of threads to banks. 

e.g. consider a system with 32 SIMD lanes, 32 LDS banks, each LDS bank is 32 bits wide, and the data type we are storing in LDS is float (4 bytes, 32 bits).

If the SIMD lanes are fetching contiguous float memory addresses then lane 0 requests from bank 0, lane 1 requests from bank 1, etc. Perfectly parallel LDS access.

Now consider that instead we are storing doubles in LDS (8 bytes, 64 bits). Each double is stored across two banks. Lane 0 requests from bank 0 and 1, lane 1 requests from bank 2 and 3... lane 16 requests from bank 0 and 1.

PROBLEM: We now have two lanes simultaneously accessing the same bank.
