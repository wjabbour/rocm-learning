# About Me

Hi, I’m Turner Jabbour. I’ve been a software engineer for ~6 years, primarily working in Node and React. Around September 2025, I became deeply interested in GPU programming, ROCm, and the broader world of low-level performance engineering.

This repository is my space to learn in public as I delve into GPU kernel engineering and inference systems work.

There are three important directories:

- [kernels](src/kernels) - I explore different kernels and include a write up of what I learned and how it relates to inference.
- [papers](papers) - I summarize and discuss different papers.
- [topics](topics) - I dive deep into some specific topic.

My long-term goal is to build strong competency in HIP, Triton, and AMD’s GPU software stack, with a focus on high-performance inference.

# Currently Working On

I’m currently re-visiting my previous kernels, profiling them, and creating profiling writeups - currently working on my [block-level reduction](src/kernels/reduction/v2_block_level/kernel.hip.cpp).

I just finished the writeup for my [halving reduction](src/kernels/reduction/v1_halving/profiling.md).

[Here](PROGRESSION_LOG.md) is everything I've worked on so far, ordered and dated.

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
