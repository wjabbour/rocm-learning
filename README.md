# About Me

Hi, I’m Turner Jabbour. I’ve been a software engineer for ~6 years, primarily working in Node.js and React. Over the last year I’ve become deeply interested in GPU programming, ROCm, and the broader world of low-level performance engineering.

This repository is my space to learn in public as I transition from front-end/back-end development into GPU kernel engineering and inference systems work. Each directory contains:

- a kernel or systems-level experiment,

- a short write-up explaining the purpose of the exercise,

- what I learned,

- and how it relates to real-world inference workloads.

My long-term goal is to build strong competency in HIP, Triton, and AMD’s GPU software stack, with a focus on high-performance inference.

# Currently Working On

I’m currently studying reduction-style patterns, including:

- register-level accumulators

- strided memory access

- block-level aggregation

- reduction via LDS and warp shuffles

I recently completed a 3D 7-point stencil kernel with configurable halo sizes, using tiling into LDS to minimize global memory traffic. This taught me a lot about:

- shared memory layout

- halo regions

- coalesced reads/writes

- occupancy and register pressure

# Scheduled Learning

### RCCL

AMD’s collectives library for multi-GPU communication (AllReduce, AllGather, ReduceScatter, etc.) used heavily in distributed inference.

### Triton

A higher-level DSL for writing high-performance kernels, increasingly used in modern inference work (FlashAttention, fused ops, reductions).

### GPU Architecture

Wavefronts, SIMDs, LDS, VGPRs, vectorized memory access, latency hiding, and the AMDGCN compiler toolchain.

### Model Serving at Scale

PagedAttention, KV-cache management, continuous batching, speculative decoding, and multi-GPU parallelism.

### Profiling & Debugging

rocprof, occupancy analysis, register pressure, DMA behavior, and memory bandwidth tuning.

# Contact

doubleujabbour@gmail.com <br>
[LinkedIn](https://linkedin.com/in/doubleujabbour)
