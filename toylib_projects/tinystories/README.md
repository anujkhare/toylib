# toylib-llm

A reimplementation of a GPT-2 sytle model in Jax. The idea is to implement it _mostly_ from scratch, but we've used existing libraries for many components as a starting point to keep things simple. Notably, the optimzers (optax) and tokenizer (huggingface) are reused directly from existing libraries.

I started this a little bit before [nanochat](https://github.com/karpathy/nanochat/tree/master) came out, with the initial focus around using the [TinyStories](https://arxiv.org/abs/2305.07759) dataset to train a model. However, given the amazing resource, I pivoted this project to rebuilding gpt-2 following nanochat's implementation.

There are a few key differences: framework (jax vs pytorch), nn libary (nanochat uses layers and sharding primitives from pytorch, this repo rebuilds base layers in jax).

## TODOs

* [x] Baseline: Train a single-device model
  * [x] Checkpointing
  * [x] Train a single-device model
* [ ] Evaluations
  * [x] Val split and loss
  * [x] Sampling / inference eval
  * [x] val bpb
  * [x] CORE metric
* [ ] Improvements
  * [x] Param counts
  * [x] Scale the batch size for stable gradients
  * [x] LR/optimizers for different parts of the model
  * [x] Gradient clipping
  * [ ] Train batch size at ~512k tokens, per device batch size ~32, 8 devices (`32 * 8 * 2048 = 512k`)
    * [ ] Smaller vocab size of the tokenizer (~25% HBM lowering)
    * [x] bf16 training (~50% HBM lowering)
    * [ ] fp8 training (~50% HBM lowering)
    * [x] 4 microbatches (4x effective batch size)
    * [x] remat all the attention layers (??)
    * [ ] we init the model on CPU first - this probably needs to change to directly on device (RAM OOMs with large models)
      * [ ] the module init() is done as post_init right now - do it explicitly instead
  * [ ] GQA
  * [ ] Weight init
  * [ ] Per-layer scalers
  * [ ] SSSL attention
  * [ ] Value embeddings
  * [ ] LR schedule
  * [x] Chinchilla optimal total FLOPs
* [ ] Scaling
  * [x] Memory and parameter analysis
  * [x] Training budget - # tokens
  * [x] Multi-core training?
  * [x] ~~Set up some sort of cross compile equivalent for local iterations~~
  * [ ] CPU/TPU memory usage, profiling
  * [ ] Gradient accumulation / micro-batching -- should memory usage be constant irrespective of the number of microbatches?
  * [*] TF Grain
  * [ ] Handle interrupts: restore checkpoints and dataset iterators
* [ ] Inference setup
  * [ ] Efficient batched inference
  * [ ] bf16 inference
  * [ ] KV caching
* [ ] Post-training
* [ ] Code health improvements
  * [x] fix dep management

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
