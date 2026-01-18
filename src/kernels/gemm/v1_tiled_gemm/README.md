# Overview

A GEMM kernel which multiplies two input matrices, A and B, and produces an output matrix C.

Strategy:

- Algorithm - using two TILE_SIZExTILE_SIZE LDS tiles (one for A and one for B), calculate blocks of C by moving these tiles "horizontally" along A and "vertically" along B.

- Execution Model - single-pass kernel. The output matrix C is divided into TILE_SIZExTILE_SIZE groups. Each thread block computes a tile of C. The threads in a block cooperatively populate LDS with the necessary elements of A and B, using `__syncthreads()` to synchronize LDS population before calculation.

Bottlenecks:

- Occupancy - LDS and register usage decreases the maximum number of wavefronts that can be scheduled on a CU.

## Build Instructions

```bash
make clean && make run SRC=src/kernels/gemm/v1_tiled_gemm/kernel.hip.cpp
```

## Learnings

- There is a very important boundary check that we must make in this kernel. If the input size is not a multiple of the block size there will be threads launched that do not map to any input element. For example, consider a block of size 64 and an input of size 2. The hardware will launch 62 threads which do not map to any input element. Therefore, each thread must consider whether it is within the input domain.

    However, this check must also consider the dimensions of the input tensor. For example, consider an input of size 6 representing a 2x3 matrix. If we launch a 1x4 block and the job of the block is to sum all of the elements of a matrix row (this is an aggregation example, but still applies to GEMM), then thread 4 with threadIdx.x = 3 must ensure that it does not simply ask "Is 3 < 6"... it must ask "Is 3 < the length of the input row (3)?" in order to determine row grouping of the input domain and avoid incorrect results.

- When requesting data from the memory system (L1, L2, HBM) it is important to ensure that the threads of a wavefront request contiguous data so that the memory controller can coalesce the requests from each thread into as few transactions as possible.

  It's important to note that a motivation for this design is the fact that GPU memory is optimized to move data in large chunks, called bursts. These bursts are aligned to the size of the cache lines, typically 128 bytes on RDNA GPUs. The fact that the memory system is capable of bursts means that we can attempt to use as few bursts as possible to transfer the data to our GPU.

  LDS on the other hand is only shared by the wavefronts of a block. It is organized at the hardware level into banks where the number of banks matches the number of SIMD lanes.

- When requesting data from LDS, it is important to ensure that the threads of a wavefront either:

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

- In order to calculate the value of Cᵢⱼ we need row i from matrix A and column j from matrix B. As the dimensions of the input matrices increase, more and more data is necessary to calculate each element of C.

  The kernel is going to need to make many requests from global memory. Therein lies a potential pitfall: although we think of A and B as matrices, they are stored as 1D contiguous chunks of memory in a format called "row major".

  Row major means that the contents of each row are stored contiguously and adjacent rows are stored contiguously.

  e.g.

  rowMajorArray = [1, 2, 3, 4, 5, 6]

  If we declare that this represents a 3x2 matrix, then that matrix looks like this:

  ```
  1 2
  3 4
  5 6
  ```

  Here you can see that as you traverse down a column, you are actually traversing N steps at a time (Here N is 2, 1 appears at index 0 and 3 appears at index 2). As N grows, the steps you take as you traverse the columns grows.

  Tying this back to the hardware: when the threads of a wavefront request the contents of memory addresses from the Vector Memory Unit, the VMU attempts to coalesce these requests into as few requests as possible. It does this by using its knowledge of the platform's cache line size, and determining which memory addresses fit into that size.

  If we were to tell a thread to load all of the data from a column of B, if N * lds_data_type_bytes > cache_line_size (very likely) then every single data load would need a separate request to global memory. Our kernel would immediately become memory-bound.

  We solve this by breaking the problem into "tiles". We fetch TILE_SIZExTILE_SIZE tiles of data from A and B. This solves two problems with our data access:

  1) instead of each thread creating an uncoalesced memory transaction, the threads work together to bring wide chunks of data back from global memory which optimizes our use of the available memory bandwidth.
  2) we reuse the data TILE_SIZE times once loaded into LDS which amortizes the cost of loading from global memory.