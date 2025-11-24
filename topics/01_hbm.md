![WIP](https://img.shields.io/badge/status-WIP-yellow)

# High Bandwidth Memory

HBM is a special kind of DRAM which is physically part of the GPU package. It acts as the RAM of the GPU.

the key thing is that cpus and gpus are different, a cpu core is serially executing instructions, so buses don't need to be as wide (just need to be wide enough to deliver the word size of the architecture 32 bit or 64 bit) whereas GPUs are executing thousands of threads in parallel - the bus lanes need to be huge, memory needs to be as close as possible to compute,

DIMM is dual in module memory, this is CPU memory, removable + upgradable, relatively far from CPU (few inches). DIMM use a printed circuit board (PCB) with copper wiring called traces. The traces can be used to carry data, power, and clock or control signals. Because the traces are so far from the CPU, timing skew can be an issue and signal integrity can be an issue. There is a theoretical maximum bandwidth for memory designed in this fashion that is too low to support the needs of the GPU.

HBM is architected without copper traces entirely. The HBM sits on the same physical package as the GPU, connected by microscopic interposer wiring which enables wide lanes that don't need as high frequency to match the bandwidth of DIMM. This far exceeds the maximum bandwidth of DIMM.

DIMMs and HBM are constructed completely differently. a DIMM is multiple DRAM chips soldered onto a PCB, transferring data via copper traces. This is a horizontal layout with a small bus, at the benefit of cheap storage with very high frequency. HBM is a series of DRAM chips, vertically layered directly onto each other, all connected to each other via through-silicon vias which are copper pilars acting as elevators.