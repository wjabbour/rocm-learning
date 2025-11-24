![WIP](https://img.shields.io/badge/status-WIP-yellow)

# Paper

[link](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf)

# Summary
Compute DNA (CDNA) is AMD's AI + HPC GPU architecture. AMD split their GPU architecture into Radeon DNA (RDNA) and CDNA because of the drastically different use cases for GPUs. CDNA focuses entirely on compute - there are no graphics pipelines or rasterization hardware. CDNA focuses completely on high-precision floating point, matrix cores specifically built for matmul operations used in AI workloads, advanced packaging technologies to combine heterogeneous architectures to tackle domain-specific workloads, and Infinity Fabric (IF) — the high-bandwidth interconnect that binds the system together.


# Thoughts & Learnings

The paper begins by explaining that GPUs have become increasingly performant, efficient, and programmable over time. Programmability seems especially relevant. If GPUs can expose additional knobs that kernel authors can twist and turn, then radically different problem spaces can all benefit.

CDNA 2 is broadly made up of the following developments:

- improved matrix core technology. Matrix cores are the specialized hardware responsible for executing AI and HPC's most common operations.
- leveraging IF to connect GPU chiplets together to improve compute density with better communication and scaling.
- heterogenous cache coherency between select GPUs and CPUs, bypassing the need for expensive memcpy operations.
- improvements across the ROCm software stack to allow porting (from CUDA via HIP) and writing of kernels.

Next, there is an architectural review of a Graphics Compute Die (GCD) which I understand to be a misnomer, given that CDNA cards contain no graphics processing capability and are designed for compute only.

Each GCD houses the compute engines, L2 cache slices, memory controllers, and IF router nodes that connect to adjacent HBM stacks on the interposer. The GCD is comprised of four compute engines, each controlling their own set of compute units which contain the SIMD pipelines which are responsible for executing instructions and performing computation. Each compute unit sits physically near the distributed L2 cache as well as near memory controllers for accessing global memory.

An MI250 and MI250X is an OAM form factor device with two GCDs on the package. The MI210 is a PCIe form factor device with a single GCD.

Now the paper goes on to talk more in depth about IF. 

Some background is that there is a standard GPU technology called Network On-chip (NoC) which is necessary to connect the hardware of a GPU die: L2, compute units, memory accessors, command processors and schedulers. IF extends this connection between physical GPU dies, enabling coherent + unified + high-bandwidth memory that allows two physical dies to be treated as a single logical GPU.

Additionally, the matrix cores of the compute units were generationally improved and extended to support additional data types which are important in HPC (FP64) and AI training workloads (lower precision typically). Peak theoretical throughput of a CDNA 2 accelerator is 47.9 TFLOPS/s of double-precision. This is a 4.2x increase over the previous generation.

There are also physical hardware related to video codecs which can be used for AI training which use photo and video data.

Next, the paper talks about how the memory hierarchy is scaled.

The compute-side of the dies have improved generationally, and what becomes increasingly important is ensuring that the system can feed these computational elements via extremely fast, wide memory buses. The memory controlles which sit on the GCD are built to access large portions of global memory and load into L2 cache to amplify CU bandwidth. This relates to inference because training models and various other workloads can be improved if they are bottle necked by data, requiring large datasets to be held in memory.

There are multiple memory controllers on the GCD, each memory controller has its own physical slice of L2 cache. In this way, memory is distributed across the die. All resources on the GCD (compute engines, compute units, matrix cores, DMA engines) can access any slice of L2 cache. The L2 is 8MB and 16-way associative. The bandwidth for each L2 slice has doubled to 128 byes per cycle, increasing the throughput of L2->CU to 6.96 TB/s in aggregate per die. Lastly, the scheduling of reads and writes to this distributed cache has been improved so that various types of workloads can benefit.

The synchronization capabilites of L2 cache have dramatically increased, allowing atomic FP64 operations like addition, min, max.