# Currently Working On

[Here](PROGRESSION_LOG.md) is everything I've worked on so far, ordered and dated.

I recently achieved 84.4% Memory Bandwidth Efficiency (~541 GB/s) on my Radeon 9070XT (RDNA 4) using wave shuffles, LDS, and a grid-stride loop. You can see the code and profiling writeup for the kernel [here](src/kernels/reduction/sum_1d/v1_block_level).

My primary focus is contributing to vLLM's ROCm inference stack. My secondary focus is further digging into ROCm and HIP, focusing on vectorization and quantization.

## Upstream Contributions

| Project | PR | Description | Status |
|---|---|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | [#40827](https://github.com/vllm-project/vllm/pull/40827) | Rename `LLMM1` → `vecMatMul`, refactor, fix two RDNA4 bugs | Open |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | [#41187](https://github.com/vllm-project/vllm/pull/41187) | Fix `reduction_smem` layout (stacks on #40827) | Open |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | [#35173](https://github.com/vllm-project/vllm/pull/35173) | Move `TORCH_CHECK` assertions in `wvSplitK` to fire before values are consumed | Open |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | [#35672](https://github.com/vllm-project/vllm/pull/35672) | Move test utility to test file, remove dead production code | Merged |
| [foundation-model-stack/fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) | [#67](https://github.com/foundation-model-stack/fastsafetensors/pull/67) | Remove `hipify-perl` build dependency, enable ROCm `manylinux` wheel builds | Merged |

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
