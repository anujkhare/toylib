# toylib-llm

A reimplementation of a GPT-2 sytle model in Jax. The idea is to implement it _mostly_ from scratch, but we've used existing libraries for many components as a starting point to keep things simple. Notably, the optimzers (optax) and tokenizer (huggingface) are reused directly from existing libraries.

I started this a little bit before [nanochat](https://github.com/karpathy/nanochat/tree/master) came out, with the initial focus around using the [TinyStories](https://arxiv.org/abs/2305.07759) dataset to train a model. However, given the amazing resource, I pivoted this project to simply rebuilding parts of nanochat that interested me and have closely followed it's implementation. The nanochat respository presents the final state of the project - rebuilding it is a great opportunity to question and understand the various decisions and implementation details that went into it.

## TODOs

* [*] Baseline: Train a single-device model
  * [*] Checkpointing
  * [*] Train a single-device model
* [ ] Inference setup
  * [ ] Efficient batched inference
  * [ ] KV caching
* [ ] Evaluations
  * [*] Val split and loss
  * [*] Sampling / inference eval
  * [*] val bpb
  * [ ] CORE metric
* [ ] Improvements
  * [*] Param counts
  * [*] Scale the batch size for stable gradients
  * [ ] LR/optimizers for different parts of the model
  * [ ] LR schedule
  * [ ] Modify optimizer
  * [ ] Gradient clipping
  * [ ] Mixed-precision training
  * [ ] bf16 inference
  * [ ] Chinchilla optimal total FLOPs
  * [ ] CPU/TPU memory usage, profiling
  * [ ] fix dep management
* [ ] Scaling
  * [*] Memory and parameter analysis
  * [*] Training budget - # tokens
  * [*] Multi-core training?
  * [ ] Gradient accumulation / micro-batching -- memory usage is still pretty high
    * [ ] work out memory analysis
    * [ ] how to make sure that we use the minimal amount of memory in the loop
    * [ ] Do we need to remat?
  * [*] TF Grain
  * [ ] Handle interrupts: restore checkpoints and dataset iterators
  * [ ] Profile / improve resource usage
  * [ ] Multi-pod training?
* [ ] Mid-training
* [ ] Post-training
