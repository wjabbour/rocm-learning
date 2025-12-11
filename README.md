# About Me

Hi, I’m Turner Jabbour. I’ve been a software engineer for ~6 years, primarily working in Node.js and React. Over the two months I’ve become deeply interested in GPU programming, ROCm, and the broader world of low-level performance engineering.

This repository is my space to learn in public as I delve into GPU kernel engineering and inference systems work.

There are three important directories:

- [kernels](src/kernels) - I explore different kernels and include a write up of what I learned and how it relates to inference.
- [papers](papers)  - I summarize and discuss different papers.
- [topics](topics) - I dive deep into some specific topic.

My long-term goal is to build strong competency in HIP, Triton, and AMD’s GPU software stack, with a focus on high-performance inference.

# Currently Working On

I’m currently studying [reduction-style patterns](src/kernels/reduction), focusing on GEMM.

I just finished working on a [Softmax kernel](kernels/reduction/v3_softmax/kernel.hip.cpp) which was a great
introduction to the kernels at the heart of inference.

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

rocprof, occupancy analysis, register pressure.

# Contact

doubleujabbour@gmail.com  
[LinkedIn](https://linkedin.com/in/doubleujabbour)
