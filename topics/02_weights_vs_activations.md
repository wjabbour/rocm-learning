# Weights Versus Activations

Recently, I began learning about vLLM, an open-source technology used to serve inference models at scale. Upon digging in, I found an [open GitHub issue](https://github.com/vllm-project/vllm/issues/28649) discussing at least one problem which causes RDNA vLLM throughput to be lower than necessary. Specifically, this is occurring in a "W8A8" kernel which means that both the weights and activations are 8-bit.

I know a bit about HIP kernels, GPU and CPU microarchitecture, and the (extremely) high-level pipeline of training and inference... but this seems like a good time to drill into what are these weights and activations and why do kernels need to be "tuned" for matrix shapes?

## What is a Weight?

Weights are fundamental to the idea of machine learning. Before machine learning, a human needed to write imperative code: "Retrieve data chunk A from memory, add 2, if sum is greater than 10, then...". This works great. We can build trillion dollar global economies on this foundational power that computers have enabled for us.

But can we write an imperative function which, if given a 1MB array of RGB values corresponding to a photograph, we decide which number (0-9) is represented in that picture? What if the picture is blurry? What if the number is varied slightly, but still recognizable?

There are infinite permutations of an image of the number 3 that a human can look at and quickly (and easily) determine that this a picture of the number 3. It is not possible to account for infinite possibilities in a purely imperative program.

Introducing: **the weight**. Instead of writing an imperative set of rules to identify a number, we need to introduce some structure or framework which "learns" how to identify the number. The process of training involves providing this structure with labeled data, a picture along with the answer, and allowing the structure to adjust internal parameters until its guess always matches the label. These internal parameters are called weights.

Once this number recognition model has been trained, its weights will be set to static values that most accurately predict the number given a 1MB input array.

## What is an Activation?

Great, now we have this giant group of tunable parameters which have been tuned via training and are ready for use. How do we use them? At a very high level, we are going to perform a series of passes on the input data. In each pass, we will perform a mathmetical operation on a specific set of weights against a specific set of the 1MB input data to produce an intermediate result. This intermediate result is called an **activation**. This activation will be processed by each pass, modified by different sets of the weights.

Once the final pass is complete, the activation will correspond with the model's choice: which number does the input represent?

## The Difference

Weights are static values that are generated from a very computationally expensive training process. Weights never change once training is complete. Activations are ephemeral values used during the inference pipeline. 

