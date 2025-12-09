![WIP](https://img.shields.io/badge/status-WIP-yellow)

# High Bandwidth Memory

HBM is a specialized form of DRAM which sits physically inside the GPU package, acting as the RAM of the GPU. It's design is fundamentally different from CPU RAM due to the difference in how CPUs and GPUs access memory.

## CPU vs GPU

A physical CPU core executes instructions serially, so the memory system only needs to provide each core with enough data per clock cycle to feed a single instruction. The memory channel literally only needs to be 64 or 128 bits wide.

The SIMD lanes in a GPU's compute unit execute instructions in parallel. A SIMD might have 32 or 64 lanes, a compute unit might have 2-4 SIMDs, and a GPU might have 40+ compute units. Each of these 2560 (at least) lanes loads up to 64 bits every clock cycle.

This requires orders of magnitude higher memory bandwidth.

## Can DIMM Scale?

DIMM = Dual Inline Memory Module

If you've ever built or owned a gaming PC, or a PC in general, you are probably familiar with the physical appearance of DIMMs. A DIMM is a long, narrow printed circuit board (PCB) with several DRAM chips soldered onto it. The DIMM plugs into the motherboard, typically a few inches from the CPU. The DIMM communicates with the CPU using copper wiring (called traces) on the motherboard.

The physical design of DIMM has a few limitations which makes it unsuitable for the high-bandwidth requirement of a GPU:

- Traces are physically large (relatively)

  Every bit transferred between the DIMM and the CPU requires its own copper trace. The DIMM must also transfer additional control, clock, and address information which also require their own traces. This means that for a DIMM to transfer 64 bits per clock cycle, there must be **well over 100 copper traces between the DIMM slot and the CPU.**
