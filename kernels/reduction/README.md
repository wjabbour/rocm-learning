# Overview

An aggregation kernel is responsible for converting a set of values to a smaller set of values, typically in the form of an operation: sum, average, min.

Aggregation kernels are very important for inference workloads - LayerNorm and Softmax to name two critical ones. These algorithms are composed of multiple aggregation kernels.

# Learnings

- Memory coalescence is crucial. Most inference workloads spend their time primarily waiting for data, not calculating, because HBM is far slower than the compute throughput of a GPU. We need to increase the efficiency of our memory operations whenever possible. The memory controller of a GPU is much wider than a CPU, fetching cache lines (32B/64B/128B) at a time, depending on architecture. As threads issue loads, the memory fabric on the chip coalesces loads bound for the same cache lines into single, wide loads in order to optimize memory access.

    For this aggregation kernel, I am careful to ensure that each thread in the wavefront accesses contiguous memory. This ensures that the impact of global reads on latency are amortized across all of the threads - the GPU can satisfy the memory loads for all the threads in the wavefront using a single HBM transaction.

- This is my first time learning about multi-pass kernels. In the case of such a kernel, we must launch multiple kernels in order to achieve the desired result. This is required because the problem is one that requires the output of some blocks to be used as input for other stages of computation. Since there is no synchronization primitive offered for blocks in the same kernel launch, we must run the dependent blocks in separate kernel launches.

- There are at least two ways to go about an aggregation: halving reduction and block-level reduction.

    For halving reduction, for input of size N, you create grid of threads of some size < N, perform the aggregation such that each thread in the grid strides along N, and now you have an output of size grid. From here, you can launch kernels of grid/j, where j doubles each time until grid/j = 1. 

    For block-level reduction, each block of threads is responsible for aggregating the values computed by the threads in the block. At the end of this reduction, you are left with gridDim partials (one for each block). At that point, the input size has been compressed from something arbitrarily massive, to something on the order of thousands (hopefully). We can then launch one more kernel with gridSize = 1 and blockSize ~= 256 to process the remaining input and reduce to a single value.

    Block-level reduction thus requires a constant (and small) number of kernel launches to perform the aggregation, whereas halving reduction requires Log(n) kernel launches. Additionally, the work performed amongst the halving reduction kernel launches varies wildly. The first launch is responsible for N/2 work, while the final kernels are leaving SIMD lanes vacant as we are processing less than a hardware wavefront.


