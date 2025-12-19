# Paper

[AMD CDNA 2 Architecture](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf)

## Preface

The `Summary and Relation to Inference` section is the main takeaway of this writeup. The remaining sections are summarizations of the sections of the white paper with the same name.

## Summary and Relation to Inference

GPUs were originally created to solve a highly parallel problem: draw a million pixels on the screen. That highly parallel problem created a computer architecture that was amenable to solving tasks in wildly different domains. As AMD and Nvidia realized that domains outside of computer graphics benefitted from this parallel GPU architecture, both companies moved to create separate hardware architectures for their consumer (graphics) and enterprise (AI + HPC) workloads.

Compute DNA (CDNA) is AMD's AI + HPC GPU architecture and Radeon DNA (RDNA) is AMD's consumer architecture.

With each generation of CDNA, the hardware of the chip is modified and improved to reflect the inherently different needs of these enterprise workloads. With CDNA 2, we see a few notable updates:

- matrix cores work natively on bfloat16 (an important inference data type) and the ISA is expanded to allow kernels to use intrinsics to improve efficiency of register access during matmul ops.

- Infinity Fabric allows cache coherency between heterogenous systems (CPU and GPU). This directly improves workloads which require frequent memory transfers between hosts and devices.

- These new hardware capabilities are exposed to developers through AMD's ROCM ecosystem.

### Introduction

GPUs have become increasingly performant, efficient, and programmable over time. That programmability, enabled by both hardware and software, is what allows GPUs to be used successfully in very different domains.

CDNA 2 is broadly made up of the following developments:

- improved matrix core technology. Matrix cores are the specialized hardware responsible for executing AI and HPC's most common operations.
- leveraging IF to connect GPU chiplets together to improve compute density with better communication and scaling.
- heterogenous cache coherency between select GPUs and CPUs, bypassing the need for expensive memcpy operations.
- improvements across the ROCm software stack to allow porting (from CUDA via HIP) and writing of kernels.

### AMD CDNA 2 Architecture Overview

The CDNA 2 architecture revolves around the Graphics Compute Die (GCD). An aside - I understand that the Graphics portion of the name is a misnomer given that CDNA cards contain no graphics processing capability.

Each GCD houses the compute engines, L2 cache slices, memory controllers, and IF router nodes that connect to adjacent HBM stacks on the interposer. The GCD is comprised of four Asynchronous Compute Engines (ACEs), each controlling their own set of compute units which contain the SIMD pipelines responsible for executing instructions and performing computation. Each compute unit sits physically near the distributed L2 cache as well as near memory controllers for accessing global memory.

Some background is that there is a standard GPU technology called Network On-chip (NoC) which is necessary to connect the hardware of a GPU die: L2, compute units, memory accessors, command processors and schedulers. IF extends this connection between physical GPU dies, enabling coherent + unified + high-bandwidth memory that allows two physical dies to be treated as a single logical GPU.

The matrix cores of the compute units were generationally improved and extended to support additional data types which are important in HPC (FP64) and AI training workloads (lower precision typically). Peak theoretical throughput of a CDNA 2 accelerator is 47.9 TFLOPS/s of double-precision. This is a 4.2x increase over the previous generation.

There are also physical hardware related to video codecs which can be used for AI training which use photo and video data.

### Scaling the Memory Hierarchy

The compute-side of the dies have improved generationally, and what becomes increasingly important is ensuring that the system can feed these computational elements via extremely fast, wide memory buses. The memory controllers which sit on the GCD are built to access large portions of global memory and load into L2 cache to amplify CU bandwidth.

There are multiple memory controllers on the GCD, each memory controller has its own physical slice of L2 cache. In this way, memory is distributed across the die. All resources on the GCD (compute engines, compute units, matrix cores, DMA engines) can access any slice of L2 cache. The L2 is 8MB and 16-way associative. The bandwidth for each L2 slice has doubled to 128 bytes per cycle, increasing the throughput of L2->CU to 6.96 TB/s in aggregate per die. Lastly, the scheduling of reads and writes to this distributed cache has been improved so that bandwidth is increased.

The synchronization capabilites of L2 cache have dramatically increased, allowing atomic FP64 operations like addition, min, max.

Memory capacity has doubled (still much orders of magnitude lower than compute capcacity).

GPU + CPU memory coherency is now usable for select models, enabling petabytes of shared + coherent memory between GPUs and CPUs. This is extremely impactful for the simplicity of runtime performance of HPC and AI workloads which require memcpy between the host and device.

### AMD CDNA 2 Architecture Shader Array

There are four Asynchronous Compute Engines (ACEs) which receive compute tasks from a command processor and dispatch them to compute units.

Each compute unit is composed of 4 16-wide SIMDs (for a total of 64 "shader cores"), registers and pipelines optimized for scalar, vector, and matrix instructions, 64KB of LDS, L1 and instruction caches, and a scheduler. 

A CDNA GPU has about 100 compute units.

CDNA is a departure from the single-precision (FP32) focus of graphics processors. The compute units have been carefully optimized to support FP64, commonly used in HPC, as a first-class citizen. These results are showcased in the following section, showing the 200-800% speedup of FP64 ops over the CDNA 1 architecture.

### AMD CDNA 2 Matrix Core Technology

There is a new FP64 matrix multiply instruction set which can be used to greatly increase the efficiency and throughput of matrix multiplication, better taking advantage of the improved matrix cores. The instructions accomplish this by reusing data loaded onto registers more optimally, requiring less register file access. The drawback is that the instructions only work on specifically-sized matrices (16x16x4 and 4x4x4).

Additionally, bfloat16 (a specialized inference data type) matrix core workloads have been improved so that they are on par with FP16 workloads.

### AMD CDNA 2 Packed FP32

The ISA has been extended with the capability of doubling FP32 operations per clock cycle by performing two FP32 operations simultaneously. To enable this, the operands must be on adjacent registers.

### AMD ROCm Open Software Platform Enables AMD CDNA 2

The ROCm software platform is an open-source ecosystem which allows developers to write performant and portable kernel code. As these libraries are integrated into 3rd party applications, these applications will become more performant, faster, and more efficient when running on AMD hardware. 

For example, when Pytorch wants to multiply a matrix, it relies on ROCm's `rocBLAS` library. `rocBLAS` has been updated to leverage the matrix core ISA available on CDNA 2, which means that a Pytorch developer is going to get their work done much faster than on CDNA 1.

To enable our systems to run the workloads of the future, the CPUs and GPUs must be peers which share access to a set of computing resources. With CDNA 2, cache coherency between nodes offers a new paradigm... enabling workloads that were not previously possible while opening the door to a future of unrealized opportunities.