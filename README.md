# Currently Working On

[Here](PROGRESSION_LOG.md) is everything I've worked on so far, ordered and dated.

I recently achieved 84.4% Memory Bandwidth Efficiency (~541 GB/s) on my Radeon 9070XT (RDNA 4) using wave shuffles, LDS, and a grid-stride loop. You can see the code and profiling writeup for the kernel [here](src/kernels/reduction/sum_1d/v1_block_level).

My primary current focus is on learning about vLLM and landing some commits in the project which push forward vLLM on RDNA.
My secondary current focus is on further digging into ROCm and HIP, focusing on vectorization and quantization.

# About Me

Hi, I'm Turner Jabbour. I've been a software engineer for ~7 years, primarily working in Node and React. Around September 2025, I became deeply interested in GPU programming, ROCm, and the broader world of low-level performance engineering.

This repository is my space to learn in public as I delve into GPU kernel engineering and inference systems work.

There are three important directories:

- [kernels](src/kernels) - I explore different kernels and include a write up of what I learned and how it relates to inference.
- [papers](papers) - I summarize and discuss different papers.
- [topics](topics) - I dive deep into some specific topic.

My long-term goal is to build strong competency in HIP, Triton, and AMD's GPU software stack, with a focus on high-performance inference.

# Contact

doubleujabbour@gmail.com  
[LinkedIn](https://linkedin.com/in/doubleujabbour)
