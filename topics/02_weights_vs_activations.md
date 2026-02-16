# Weights versus Activations

Recently, I began learning about vLLM, an open-source technology used to serve inference models at scale. Upon digging in, I found an [open GitHub issue](https://github.com/vllm-project/vllm/issues/28649) discussing at least one problem which causes RDNA vLLM throughput to be lower than necessary. Specifically, this is occuring in a "W8A8" kernel which means that both the weights and activations are 8-bit.

I know a bit about HIP kernels, GPU and CPU microarchitecture, and the (extremely) high-level pipeline of training and inference... but this seems like a good time to drill into what are these weights and activations and why do kernels need to be "tuned" for matrix shapes?