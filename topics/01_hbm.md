![WIP](https://img.shields.io/badge/status-WIP-yellow)

# High Bandwidth Memory

HBM is a specialized form of DRAM which sits physically inside the GPU package, acting as the RAM of the GPU. It's design is fundamentally different from CPU RAM due to the difference in how CPUs and GPUs access memory.

## CPU vs GPU

A physical CPU core executes instructions serially, so the memory system only needs to provide each core with enough data per clock cycle to feed a single instruction. The memory channel only needs to be 64 or 128 bits wide.

The SIMD lanes in a GPU's compute unit execute instructions in parallel. A SIMD might have 32 or 64 lanes, a compute unit might have 2-4 SIMDs, and a GPU might have 40+ compute units. Each of these 2560 (at least) lanes loads up to 64 bits every clock cycle.

This requires orders of magnitude higher memory bandwidth.

## Can DIMM Scale?

DIMM = Dual Inline Memory Module

If you've ever built or owned a gaming PC, or a PC in general, you are probably familiar with the physical appearance of DIMMs. A DIMM is a long, narrow printed circuit board (PCB) with several DRAM chips soldered onto it. The DIMM plugs into the motherboard, typically a few centimeters from the CPU. The DIMM communicates with the CPU using copper wiring (called traces) on the motherboard.

The physical design of DIMM has a few limitations which makes it unsuitable for the high-bandwidth requirement of a GPU:

- Traces are physically large (relatively)

  Every bit transferred between the DIMM and the CPU requires its own copper trace. The DIMM must also transfer additional control, clock, and address information which also require their own traces. This means that for a DIMM to transfer 64 bits per clock cycle, there must be **well over 100 copper traces between the DIMM slot and the CPU.** We cannot decrease the width of these traces past a certain bound because:

  1) it's difficult to engineer
  2) as the cross-sectional area of a conductor decreases, the resistance increases and the voltage decreases. We would need to "push" the electrons much harder in order for the receiver to see a "1" or "0".

- The "Power Wall"

As you increase the frequency (clock rate) of a computing chip, the power consumption and heat dissipation of the chip increases polynomially. We cannot increase DIMM bandwidth by simply making it transfer data more frequently because it would get too hot and require too much power.

- Signal Integrity

As you increase the frequency of an electrical signal, the copper trace begins to function as an antenna. The back-and-forth of electrons in the physical medium produces electromagnetic waves which interfere with neighboring traces carrying data. This effect is called crosstalk.

The core issue is that DIMMs are designed to move relatively small amounts of data at a reasonably high frequency, but due to the laws of physics we cannot continue increasing frequency in order to increase data transfer per second.

## A New Hope

There are two primary problems to solve:

1. We need to decrease the physical distance that electrons need to travel.

    This decreases resistance which decreases the voltage required to carry the electrons to the receiver. Less voltage requires less overall power.

2. We need to increase the amount of data that can be delivered simultaneously.

### Distance

### Width

## The Inference Connection
