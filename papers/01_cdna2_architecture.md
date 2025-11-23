# Paper

[link](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf)

# Summary


# Thoughts & Learnings

The paper begins by explaining that GPUs have become increasingly performant, efficient, and programmable over time. Programmability seems especially relevant. If GPUs can expose additional knobs that kernel authors can twist and turn, then radically different problem spaces can all benefit.

Next, the paper goes on to describe the architectural improvements AMD has made to the matrix core technology, which is the specialized hardware responsible for executing AI and HPC's most common operations. Infinity Fabric is a high-bandwidth memory solution used to connect GPU chiplets and allow higher compute density by making space for more chiplets on the physical substrate. Heterogenous cache coherency is available on CDNA2 GPUs, allowing the GPU to operate and CPU to work more closely together, avoiding the need for expensive memory-bound operations. All of this is seamlessly enabled by the ROCm stack which allows developers to easily take advantage of these capabilities when writing kernels (which also includes porting over CUDA kernels by writing in HIP).