# Weights versus Activations

Recently, I began learning about vLLM, an open-source technology used to serve inference models at scale. Upon digging in, I found an [open GitHub issue](https://github.com/vllm-project/vllm/issues/28649) discussing at least one problem which causes RDNA vLLM throughput to be lower than necessary. Specifically, this is occurring in a "W8A8" kernel which means that both the weights and activations are 8-bit.

I know a bit about HIP kernels, GPU and CPU microarchitecture, and the (extremely) high-level pipeline of training and inference... but this seems like a good time to drill into what are these weights and activations and why do kernels need to be "tuned" for matrix shapes?

## What Is A Weight?

Weights are fundamental to the idea of machine learning. Before machine learning, a human needed to write imperative code: "Retrieve data chunk A from memory, add 2, if sum is greater than 10, then...". This works great. We can build trillion dollar global economies on this foundational power that computers have enabled for us.

But can we write an imperative function which, if given a 1MB array of RGB values corresponding to a photograph, we decide which number (0-9) is represented in that picture? What if the picture is blurry? What if the number is varied slightly, but still recognizable?

There are infinite permutations of an image of the number 3 that a human can look at and quickly (and easily) determine that this a picture of the number 3. It is not possible to account for infinite possibilities in a purely imperative program.

Introducing: **the weight**. Instead of writing an imperative set of rules to identify a number, we need to introduce some structure or framework which "learns" how to identify the number. The process of training involves providing this structure with labeled data, a picture along with the answer, and allowing the structure to adjust internal parameters until its guess always matches the label. These internal parameters are called weights.