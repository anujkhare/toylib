# toylib-llm

A reimplementation of a GPT-2 sytle model in Jax. The idea is to implement it _mostly_ from scratch, but we've used existing libraries for many components as a starting point to keep things simple. Notably, the optimzers (optax) and tokenizer (huggingface) are reused directly from existing libraries.

I started this a little bit before [nanochat](https://github.com/karpathy/nanochat/tree/master) came out, with the initial focus around using the [TinyStories](https://arxiv.org/abs/2305.07759) dataset to train a model. However, given the amazing resource, I pivoted this project to simply rebuilding parts of nanochat that interested me and have closely followed it's implementation. The nanochat respository presents the final state of the project - rebuilding it is a great opportunity to question and understand the various decisions and implementation details that went into it.

## Logs

### 2026-02-15

Trained an initial d20 model with bs=48, seq_len=2048, 100k steps.

The minimum val-bpb is `~1.45` which seems very high, though it's unclear to me what a comparable number from nanochat is. The quoted numbers on nanochat are for a `d24` model at val bpb of `~0.75`.

Things to investigate:

0. Just look through <https://github.com/karpathy/nanochat/discussions/481> and pull in all the optimizations made there.
1. Tokenizer is different: we're using the default GPT-2 tokenizer from HF which has `vocab_size=50257`, with `bpb=~6.5` (TODO: check), whereas nanochat trains a custom tokenizer with `vocab_size=32768` and `bpb=xx`. The larger vocab size likely makes it harder for our model to learn initially.
2. Can we actually compare bpb...?
3. Add the CORE metric evaluation

## TODOs

* [*] Baseline: Train a single-device model
  * [*] Checkpointing
  * [*] Train a single-device model
* [ ] Evaluations
  * [*] Val split and loss
  * [*] Sampling / inference eval
  * [*] val bpb
  * [ ] CORE metric
* [ ] Improvements
  * [*] Param counts
  * [*] Scale the batch size for stable gradients
  * [*] LR/optimizers for different parts of the model
  * [*] Gradient clipping
  * [ ] Train batch size at ~512k tokens, per device batch size ~32, 8 devices (`32 * 8 * 2048 = 512k`)
    * [ ] Smaller vocab size of the tokenizer (~25% HBM lowering)
    * [ ] bf16 training (~50% HBM lowering)
    * [ ] fp8 training (~50% HBM lowering)
    * [ ] 4 microbatches (4x effective batch size)
    * [ ] remat all the attention layers (??)
    * [ ] we init the model on CPU first - this probably needs to change to directly on device (RAM OOMs with large models)
      * [ ] the module init() is done as post_init right now - do it explicitly instead
  * [ ] GQA
  * [ ] Weight init
  * [ ] Per-layer scalers
  * [ ] SSSL attention
  * [ ] Value embeddings
  * [ ] LR schedule
  * [*] Chinchilla optimal total FLOPs
* [ ] Scaling
  * [*] Memory and parameter analysis
  * [*] Training budget - # tokens
  * [*] Multi-core training?
  * [ ] Set up some sort of cross compile equivalent for local iterations
  * [ ] CPU/TPU memory usage, profiling
  * [ ] Gradient accumulation / micro-batching -- memory usage is still pretty high
    * [ ] work out memory analysis
    * [ ] how to make sure that we use the minimal amount of memory in the loop
    * [ ] Do we need to remat?
  * [*] TF Grain
  * [ ] Handle interrupts: restore checkpoints and dataset iterators
* [ ] Inference setup
  * [ ] Efficient batched inference
  * [ ] bf16 inference
  * [ ] KV caching
* [ ] Post-training
* [ ] Code health improvements
  * [x] fix dep management
  * [ ] How to make everything more configurable?
  * [ ] Refactor out the experiment and train loop

## Compilation Analysis

The `compile.py` script analyzes JAX compilation, memory usage, and generates profiling traces.

### CLI Usage

```bash
python -m toylib_projects.tinystories.scripts.compile \
    --num_devices 8 \
    --output_dir /tmp/compile_analysis \
    --trace_steps 3 \
    --batch_size_per_device 4 \
    --depth 4
```

### Colab/Notebook Usage

```python
# IMPORTANT: Setup fake devices BEFORE any JAX imports
from toylib_projects.tinystories.scripts.compile import setup_fake_devices
setup_fake_devices(num_devices=1)

# Now import and run
from toylib_projects.tinystories.scripts.compile import CompileConfig, main

config = CompileConfig(
    num_devices=1,
    batch_size_per_device=4,
    depth=4,
    skip_trace=True,  # Often useful in Colab
)
main(config)
```

### View Traces

* **Perfetto**: Open <https://ui.perfetto.dev> and load the `.json.gz` trace file
* **TensorBoard**: `tensorboard --logdir /tmp/compile_analysis/traces`
