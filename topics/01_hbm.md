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

  Every bit transferred between the DIMM and the CPU requires its own copper trace. The DIMM must also transfer additional control, clock, and address information each requiring their own traces. This means that for a DIMM to transfer 64 bits per clock cycle, there must be **well over 100 copper traces between the DIMM slot and the CPU.** We cannot decrease the width of these traces past a certain bound because:

  1. it's difficult to engineer
  2. as the cross-sectional area of a conductor decreases, the resistance increases which directly decreases voltage measured at the receiver. We would need to "push" the electrons much harder in order for the receiver to see a "1" or "0".

- The "Power Wall"

  As you increase the frequency (clock rate) of a computing chip, the power consumption and heat dissipation of the chip increases polynomially. We cannot increase DIMM bandwidth by simply making it transfer data more frequently because it would get too hot and require too much power.

- Crosstalk

  As you increase the frequency of an electrical signal, the copper trace begins to function as an antenna. The back-and-forth of electrons in the physical medium produces electromagnetic waves which interfere with neighboring traces carrying data.

- Signal Integrity

  The farther electrons have to travel, the harder they need to be "pushed" along the physical medium to produce waves with enough amplitude differential between "0" and "1" to be easily discernible by the receiver.
  
  Additionally, every physical medium has a property called capacitance, which measures that medium's ability to store electrical charge. In the context of signal integrity, a medium with high capacitance causes two large problems:

  1. it increases the latency of the electrical signal transfer between the sender and receiver
  2. the voltage at the receiver drains more slowly, putting a hard cap on the frequency that signals can be sent across the medium. If a signal is sent before the previous signal has drained to a sufficient level, the receiver's voltage graph won't clearly disambiguate the end of the previous signal and the start of the current one.

The core issue is that DIMMs are designed to move relatively small amounts of data at a high frequency, but due to the laws of physics we cannot continue increasing frequency in order to increase data transfer per second.

## A New Hope

There are two primary problems to solve:

1. We need to decrease the physical distance that electrons need to travel.

    A decrease in the length of the medium directly decreases the resistance of the medium which decreases the voltage required to push electrons through the medium, reducing the power usage of the overall system.

2. We need to increase the amount of data that can be delivered simultaneously.

    Since we cannot increase the frequency that we deliver data, we need to deliver more data with each clock cycle.

### Distance

Instead of placing the memory on the other side of a PCB, let's place the memory directly next to the GPU die. Since the electrical signal only needs to travel a few millimeters, we don't need long copper traces anymore; we can use microscopic, silicon interposers. This directly reduces capacitance and resistance, improving the power consumption and signal integrity of the system.

Additionally, traditional DIMMs are laid out horizontally. The farthest memory cell from the centralized memory controller is on the order of centimeters. To decrease the total distance that the signal needs to travel, we should attempt to reduce this within-memory distance as well.

Let's stack the DRAM chips on top of one another and connect them with microscopic copper poles, called Through-Silicon Vias (TSVs). This greatly reduces the physical distance that the signal must travel from the farthest memory cell to the memory controller.

### Width

DIMMs are built to transfer 64-bits per clock cycle at a high frequency. Instead, let's transfer 512 or 1024 bits per clock cycle at a lower frequency, but still high enough to have a much higher theoretical bandwidth than a traditional DIMM.

## The Inference Connection

Workloads are broadly classified into "Compute Bound" and "Memory Bound". The decoding portion of the inference pipeline, the part where the model generates the tokens of the response, is memory bound. This is due to the auto-regressive nature of text generation: token N must use tokens 0 through (N-1) as input. For each token generated, the entire models weights must be loaded from VRAM onto the compute units.

Therefore, the theoretical maximum memory bandwidth of the GPU puts a hard cap on the amount of tokens that can be generated per second during the decoding phase.

## What's Next?

To improve inference speed, we need to improve the rate at which the data we need can be moved to the place we need it. For this broad goal we have at least a few levers:

- Compression: decrease the amount of data we need to move across the memory bus, perform computation to inflate.
- Quantization: like compression, decrease the amount of data we need to move because we don't need high accuracy for all workloads.
- Hardware/Software Integration: minimize the duplication of weight data movement across token generation steps.
