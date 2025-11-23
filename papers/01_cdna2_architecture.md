# Paper

[link](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf)

# Summary
Compute DNA (CDNA) is AMD's AI + HPC GPU architecture. AMD split their GPU architecture
into Radeon DNA (RDNA) and CDNA because of the drastically different use cases for GPUs.
CDNA focuses entirely on compute - there are no graphics pipelines or rasterization hardware. CDNA focuses completely on high-precision floating point, matrix cores specifically built for matmul operations used in AI workloads, advanced packaging technologies to combine heterogenous architectures to tackle domain-specific workloads, and Infinity Fabric - the high-bandwidth memory that connects everything together.


# Thoughts & Learnings

The paper begins by explaining that GPUs have become increasingly performant, efficient, and programmable over time. Programmability seems especially relevant. If GPUs can expose additional knobs that kernel authors can twist and turn, then radically different problem spaces can all benefit.

Next, the paper goes on to describe the architectural improvements AMD has made to the matrix core technology, which is the specialized hardware responsible for executing AI and HPC's most common operations. Infinity Fabric is a high-bandwidth memory solution used to connect GPU chiplets and allow higher compute density by making space for more chiplets on the physical substrate. Heterogenous cache coherency is available on CDNA2 GPUs, allowing the GPU and CPU to work more closely together, avoiding the need for expensive memory-bound operations. All of this is seamlessly enabled by the ROCm stack which allows developers to easily take advantage of these capabilities when writing kernels (which also includes porting over CUDA kernels by writing in HIP).

Next, there is an architectural review of a Graphics Compute Die (GCD) which I understand to be a misnomer, given that CDNA cards contain no graphics processing capability and are designed for compute only.

The GCD contains the compute, memory and cache controllers, and Infinity Facbric/HBM connections necessary to build exascale computing systems. The GCD is comprised of four compute engines, each controlling their own set of compute units which contain the SIMD pipelines which are responsible for executing instructions and performing computation.
Each compute unit sits physically near a large, shared L2 cache as well as near memory controllers for accessing global memory.

An MI250 and MI250X is an OAM form factor device with two GCDs on the package. The MI210 is a PCIe form factor device with a single GCD.