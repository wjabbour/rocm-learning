# Overview

A reduction kernel is responsible for converting a set of values to a smaller set of values, typically in the form of an operation: sum, average, min.

Reduction kernels are a bottleneck for LLM inference workloads - notably, Softmax and Layernorm.

In the case of Softmax, we use this algorithm to create a probability distribution of the next token in the sequence. This requires two forms of reduction:

1) Numerical stability - if the function which produces our distribution relies on exponentiation, then even relatively small inputs may exceed the maximum size of the data type of our kernel variable. To address this, we must perform a max reduction to find the maximum element from the input and subtract that value from all input elements.
2) Normalization - we must divide each input element by the sum of all input elements. In order to find that sum, we must perform a reduction.

## Kernel Progression

| Version | Name                      | Key Strategy | Bottleneck Solved | Remaining Issue                   |
| :------ | :------------------------ | :----------- | :---------------- | :-------------------------------- |
| **v0**  | `sum_1d/v0_halving`       | -            | -                 | **reason:** xyz                   |
| **v1**  | `sum_1d/v1_block_level`   | -            | -                 | **reason:** xyz                   |
| **v0**  | `softmax/v0_naive_softmax` | -            | -                 | **reason:** xyz                   |