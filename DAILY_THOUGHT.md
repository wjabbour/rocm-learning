# Daily Thought

## 12/6/2025

GPU architecture enables highly parallel computation, but at the expense of scheduling freedom. For example, waves of the same kernel are typically bound to the same set of CUs by the scheduler because the cost of switching between kernels with different memory and register footprints is expensive. CPUs operate in an inherently different manner, relying on fast context-switching and pre-emption to make slices of progress across potentially thousands of processes.

There must be a compromise.

## 12/3/2025

An if statement in a kernel can negatively impact perforamnce via branch divergence.

## 11/28/2025

PCIe is a multi-layer protocol for asynchronously (no shared clock) transmitting data between the host system and a device. Since a PCIe bus has its own internal clock, it is abstracted from the timing and other hardware aspects of the host system. These two facts enable a high degree of interopability between PCIe devices and host systems.

## 11/25/2025

Bits don't exist. Only electricity exists.

## 11/24/2025

The MI250 has 362 TFLOPs of half-precision performance with a peak HBM bandwidth of 3.2 TB/s. A TFLOP is one trillion floating point operations. The GPU can pull in, at most, (3.2 * 10^12 bytes) / 2 = 1.6 trillion FP16 values per second, but can perform 362 trillion operations per second.
