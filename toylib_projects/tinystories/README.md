A reimplementation of a GPT-2 sytle model in Jax. The idea is to implement it _mostly_ from scratch, but we've used existing libraries for many components as a starting point to keep things simple. Notably, the optimzers (optax) and tokenizer (huggingface) are reused directly from existing libraries.

I started this a little bit before [nanochat](https://github.com/karpathy/nanochat/tree/master) came out, with the initial focus around using the [TinyStories](https://arxiv.org/abs/2305.07759) dataset to train a model. However, given the amazing resource, I pivoted this project to simply rebuilding parts of nanochat that interested me and have closely followed it's implementation. The nanochat respository presents the final state of the project - rebuilding it is a great opportunity to question and understand the various decisions and implementation details that went into it.


### TODOs
* [ ] Baseline: Train a single-device model
    * [ ] Metrics
    * [ ] Checkpoint
    * [ ] Handle interrupts: restore checkpoints and dataset iterators
    * [ ] Train a single-device model
* [ ] Inference setup
    * [ ]
* [ ] Evaluations
* [ ] Mid-training
* [ ] Post-training