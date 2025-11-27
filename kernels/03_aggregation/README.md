# Overview

An aggregation kernel is responsible for converting a set of values to a smaller set of values, typically in the form of an operation: sum, average, min.

Aggregation kernels are very important to inference workloads - LayerNorm and Softmax to name two critical ones. These algorithms are composed of multiple aggregation kernels.

# Learnings

- Memory coalescence is crucial. Most inference workloads spend their time primarily waiting for data, not calculating. We need to increase the efficiency of our memory operations whenever possible. The memory controller of a GPU is much wider than a CPU, fetching some hardware-based number of bits at a time. This memory controller is listening to requests from the threads running across the CUs. I conceptualize this controller as a server, listening to requests from its clients. It is batching the client requests to the upstream system (memory) and combining requests if they are requesting memory that is within the fetch bounds (the hardware-based number).

    This is why strided memory access (depending on the offset, it must be within the hardware-based fetch size for coalescence) is powerful. If we can construct work groups of threads which access contiguous memory then we can improve the memory performance of our kernel (however, we are hopelessly memory-bound due to hardware constraints).

- This is my first time learning about multi-pass kernels. In the case of such a kernel, we must launch multiple kernels in order to finish the job. This required me to experiment a little more with the communication between host and device.

- There are at least two ways to go about an aggregation: halving reduction and block-level reduction.

    For halving reduction, for input of size N, you create grid of threads of some size < N, perform the aggregation such that each thread in the grid strides along N, and now you have an output of size grid. From here, you can launch kernels of grid/j, where j doubles each time until grid/j = 1. 

    For block-level reduction, each block of threads is responsible for aggregating the values computed by the threads in the block. At the end of this reduction, you are left with gridDim partials (one for each block). At that point, the input size has been compressed from something arbitrarily massive, to something on the order of thousands (hopefully). We can then launch one more kernel with gridSize = 1 and blockSize ~= 256 to process the remaining input and reduce to a single value.

    Block-level reduction thus requires a constant (and small) number of kernel launches to perform the aggregation, whereas halving reduction requires Log(n) kernel launches. Additionally, the work performed amongst the halving reduction kernel launches varies wildly. The first launch is responsible for N/2 work, while the final kernels are leaving SIMD lanes vacant as we are processing less than a hardware wavefront.


