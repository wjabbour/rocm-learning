# Thoughts

## 2/13/2026

LLM decoders are auto-regressive: to generate output token N + 1, the model must consider the previous N tokens. This is the exact problem that transormers revolutionized on the training side - transformers enabled parallelizable training pipelines via large matrix multiplications. Can we find some way of parallelizing output generation?

Once again, we should look to biology to guide the design of our technological system. How do humans think and how do we form sentences? We usually have some abstract ideas in our mind and generate sentences based on those ideas. Can we create idea vectors for answer sentiments, use those answer sentiments and answer-text-segment awareness to parallelize?

Hopefully there is some improvement we can make on the current state-of-the-art speculative decoding.

## 2/12/2026

In the past, computers were as large as rooms and people were relegated slices of time on that shared resource; but now we all have phones in our pocket. Right now, I'm paying $20 a month for a Gemini subscription.

## 2/5/2026

It is greatly underestimated how close we currently are to replacing most non-physical human jobs.

## 12/13/2025

To create artificial intelligence, it's helpful to create computational systems in the spirit of biological systems.

For example, we associate some importance with our memories which dictates how quickly we can recall, and for how long we can retain those memories.

If we can apply a similar idea to LLMs, we could increase the usefulness of the context window, decreasing its average size.

Edit: this idea already exists. I need to take a look at [this article.](https://arxiv.org/abs/2310.08560#:~:text=Large%20language%20models%20(LLMs)%20have,Comments:)

## 12/6/2025

GPU architecture enables highly parallel computation, but at the expense of scheduling freedom. For example, waves of the same kernel are typically bound to the same set of CUs by the scheduler because the cost of switching between kernels with different memory and register footprints is expensive. CPUs operate in an inherently different manner, relying on fast context-switching and pre-emption to make slices of progress across potentially thousands of processes.

There must be a compromise.

## 12/3/2025

An if statement in a kernel can negatively impact performance via branch divergence.

## 11/28/2025

PCIe is a multi-layer protocol for asynchronously (no shared clock) transmitting data between the host system and a device. Since a PCIe bus has its own internal clock, it is abstracted from the timing and other hardware aspects of the host system. These two facts enable a high degree of interopability between PCIe devices and host systems.

## 11/25/2025

Bits don't exist. Only electricity exists.

## 11/24/2025

The MI250 has 362 TFLOPs of half-precision performance with a peak HBM bandwidth of 3.2 TB/s. A TFLOP is one trillion floating point operations. The GPU can pull in, at most, (3.2 * 10^12 bytes) / 2 = 1.6 trillion FP16 values per second, but can perform 362 trillion operations per second.
