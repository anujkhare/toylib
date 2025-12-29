A reimplementation of a GPT-2 sytle model in Jax. The idea is to implement it _mostly_ from scratch, but we've used existing libraries for many components as a starting point to keep things simple. Notably, the optimzers (optax) and tokenizer (huggingface) are reused directly from existing libraries.

I started this a little bit before [nanochat](https://github.com/karpathy/nanochat/tree/master) came out, with the initial focus around using the [TinyStories](https://arxiv.org/abs/2305.07759) dataset to train a model. However, given the amazing resource, I pivoted this project to simply rebuilding parts of nanochat that interested me and have closely followed it's implementation. The nanochat respository presents the final state of the project - rebuilding it is a great opportunity to question and understand the various decisions and implementation details that went into it.


### TODOs
* [ ] Baseline: Train a single-device model
    * [ ] Metrics
        * [*] Sampling / inference eval
        * [ ] val bpb
        * [ ] CORE metric
    * [*] Checkpointing
    * [ ] Train a single-device model
* [ ] Inference setup
    * [ ]
* [ ] Evaluations
* [ ] Improvements
    * [*] Param counts
    * [ ] Scale the batch size for stable gradients
    * [ ] Chinchilla optimal total FLOPs
    * [ ] Mixed-precision training
    * [ ] bf16 inference
    * [ ] CPU/TPU memory usage, profiling
    * [ ] LR/optimizers for different parts of the model
    * [ ] LR schedule
    * [ ] Modify optimizer
    * [ ] Gradient clipping
    * [ ] fix dep management
* [ ] Scaling
    * [*] Memory and parameter analysis
    * [ ] Training budget - # tokens
    * [ ] Multi-core training?
    * [ ] Multi-pod training?
    * [ ] Gradient accumulation / micro-batching
* [ ] Mid-training
* [ ] Post-training
* [ ] Nice-to-haves
    * [ ] Scaling laws
    * [ ] Handle interrupts: restore checkpoints and dataset iterators


### Notes
Initial code up and running with basic training loop, validation, sampling, and checkpointing set up!

Trained a d12 model (~117M parameters) with a single TPU on colab, using batch size of 16 for about 50k steps, which is roughly `16 * 2048 * 50k = ~1.64M` tokens. For a Chinchilla optimal compute training, we need to train this model for roughly `20 * 117M = ~2.34B` tokens.

Further, the batch size we used is way too small, at just `32k` tokens per batch. GPT-2 used `~1M` tokens per optimization step. We target `~512k` tokens. To achieve this, we have two options:
1. Micro-batching: run through multiple batches for each optimization step
2. Use more devices or devices with more memory

Here, we hit another issue with our implementation - our training is all single-device right now and does not scale to multiple devices.
