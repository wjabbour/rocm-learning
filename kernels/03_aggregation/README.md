# Overview

An aggregation kernel is responsible for converting a set of values to a smaller set of values, typically in the form of an operation: sum, average, min.

Aggregation kernels are very important to inference workloads - LayerNorm and Softmax to name two critical ones. These algorithms are composed of multiple aggregation kernels.

# Learnings

- Memory coalescence is crucial. Most inference workloads spend their time primarily waiting for data, not calculating. We need to increase the efficiency of our memory operations whenever possible.
