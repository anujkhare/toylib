# Experiments

Training runs for post 2, in the order they ran, and what each one showed. `BENCHMARK.md`
holds the hardware, speed and cost measurements these runs were planned from. All runs are on
ClimbMix with our 32,768-token tokenizer, report `val/bits_per_byte` on nanochat's validation
shard, and log to the [toylib-p02-climbmix](https://wandb.ai/anujkhare18/toylib-p02-climbmix)
W&B project.

| Run | What it tested | Result |
| --- | --- | --- |
| `health_a100_8x32` | `RUNS["d20"]`'s setup, for 3% of its budget | Every health check passed |
| `d12` | Our baseline recipe at nanochat's reference size | 0.949, 6 to 10% behind nanochat |
| `local_nanochat` | nanochat's current recipe, on a laptop | 1.692 against the baseline's 1.753 |
| `d12_nanochat` | nanochat's current recipe at d12 | **0.877**, 7.6% ahead of the baseline and inside nanochat's range |
| `d20_nanochat` | The post's model | **0.783**, inside the 0.75 to 0.90 expected |

## Health run

200 steps on `RUNS["d20"]`'s real schedule: 105M tokens, 3% of the budget, in 59 minutes on 8
shards. [W&B run](https://wandb.ai/anujkhare18/toylib-p02-climbmix/runs/health_a100_8x32).

| Check | Expected | Seen | Passed |
| --- | --- | --- | --- |
| Starts at chance | Loss ln(32,768) = 10.397 | 10.398 | Yes |
| Bits per byte at chance | 15 bits over about 4.7 bytes | 3.22 | Yes |
| Bits per byte falls at every eval | No rises | 3.22, 2.44, 2.03, 1.86, 1.72, 1.60, 1.51, 1.43, 1.38 | Yes |
| Drops by at least 30% | Under 2.25 | 1.38, down 57% | Yes |
| Train matches validation | Within 0.3 nats | 4.47 against 4.45 | Yes |
| Update size | 0.001% to 10% of the weights | 0.08% to 0.13% | Yes |
| Gradients stable | Under 10 times the median | Median 0.72, peak 2.2 | Yes |
| Everything finite | No NaNs | None | Yes |

Samples went from random words to grammatical but empty English, such as "This is the most popular
and most popular among others", which is right for 3% of a run. Clipping at 1.0 rarely applies.

**What to expect from the full run.** nanochat's best d24 reaches 0.718 on this validation shard,
with value embeddings and a larger model. Its earlier scaling sweep shows about 0.80 at 3.8B
tokens, on older data. So finishing between 0.75 and 0.90 would be healthy for this smaller model.
Those are other models' numbers, so this is a rough guide. The trial shows the run is set up
correctly, not where it will end.

**Not yet tested on Colab:** the notebook itself. Run it with `SMOKE=1` first, as
`notebooks/COLAB.md` says. Writing the 120-shard token files and resuming from the bucket
both worked in the d12 runs below.

## d12 baseline

A full run at a smaller size, to compare with nanochat before spending 209 units on d20.
`train.RUNS["d12"]`: 12 blocks, 768 dims, 6 heads, 110M parameters by nanochat's count, 12
tokens per parameter, so 2,520 steps of 524,288 tokens, or 1.32B tokens, as 16 microbatches
of 16. [W&B run](https://wandb.ai/anujkhare18/toylib-p02-climbmix/runs/d12).

**It finished at 0.949 bits per byte on nanochat's 41.9M evaluation tokens, 6 to 10% worse than
nanochat.** nanochat's scaling sweep in `dev/LOG.md` reaches 0.897 at about 1.24B tokens, on
FineWeb with an older architecture; my estimate for ClimbMix is about 0.865. Before the run I
called above about 0.95 a reason to investigate, and this lands right at that line. The run
is healthy, so the gap is in what it learns, not how it trains.

| Step | 0 | 250 | 500 | 1,000 | 1,500 | 2,000 | 2,500 | Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bits per byte | 3.22 | 1.34 | 1.12 | 1.02 | 0.982 | 0.962 | 0.946 | 0.949 |

The final number reads 12 times as much text as the evals before it, which explains the 0.003
between them. Training loss tracked validation throughout, gradients settled from 0.21 to
0.08, and the update ratio fell from 0.11% to 0.01% as the learning rate decayed.

Where the gap might come from:

- **Architecture.** nanochat's current model adds value embeddings, sliding-window attention and
  several smaller pieces, listed below.
- **Optimizer and schedules.** Ours follow nanochat's learning rates at `B_REF`, but its
  current Muon, weight decay and schedules differ.
- **The 0.865 itself.** It's an estimate across datasets, so the true gap could be smaller.

**Speed: 85,000 tokens per second, 24.4% MFU, below d20's 32.7%.** I had assumed d12 would match
d20's MFU, which predicted 3.2 hours; it took 4.4. A smaller model spends more of each step
outside the big matmuls, but this wasn't profiled.

**Cost: 33.5 compute units** in all, including writing the token files (about 25 minutes),
two restarts, and the final evaluation.

**Runtimes.** Colab ended the first runtime 3.5 hours in, right after the step 2,000
checkpoint; resuming lost no steps. A second runtime finished training, and a third ran only
the final evaluation, because the second was stopped before reporting it (see
`toylib_projects/scripts/colab_idle_stop.py`). Host memory stayed under 7 GiB of 85, so
memory didn't cause the first loss.

## Porting nanochat's recipe

Rather than add one difference at a time and measure each, this ports nanochat's current
recipe whole and checks that it beats the baseline. Each piece is off by default, so `RUNS["d12"]`
and `RUNS["d20"]` train exactly as before. `NANOCHAT_OPTIONS` in `train.py` turns them all on.

### Model, `decoder_only_model.py`

`decoder_only_model.NANOCHAT` holds these `ModelConfig` settings:

| Setting | nanochat's value | What it does |
| --- | --- | --- |
| `window_pattern` | `"SSSL"` | Sliding-window attention. Each layer takes the next letter of the pattern, cycling: S attends to the previous quarter of the context (512 tokens at 2,048), L to all of it. The last layer is always L. `ModelConfig.window_sizes()` lists them. |
| `value_embeddings` | `True` | On alternating layers, always including the last, a second embedding table per layer is added to the attention values. A per-head gate, `3 * sigmoid(x[..., :12] @ W)`, controls how much. d12 has 6 tables and d20 has 10. |
| `residual_scalars` | `True` | Before each block, `x = resid_lambda[i] * x + x0_lambda[i] * embedding`, both learned. They start at 1.15 to 1.05 and 0.20 to 0.05 from the first layer to the last. |
| `smear` | `True` | Mixes the previous token's embedding into each position, through a gate on its first 24 channels. The gate's scale starts at 0, so it starts off. |
| `backout` | `True` | Subtracts a learned share, starting at 0.2, of the residual after the middle block from the final residual. |
| `qk_scale` | `1.2` | Multiplies the RMS-normalized queries and keys. |
| `mlp_init_scale` | `0.4` | Scales the MLP's first layer at initialization. |
| `embedding_init_std` | `0.8` | The token embedding's initial standard deviation. |
| `rope_base` | `100_000` | RoPE's base, up from 10,000. |

Other model changes:

- **The blocks run in a Python loop instead of a scan**, since they no longer share a shape. With
  remat each block is wrapped in `jax.remat`. Parameter paths didn't change, so old
  checkpoints still load.
- **`attention_impl="cudnn"` uses `jax.nn.dot_product_attention`'s fused cuDNN kernel**, with
  `is_causal` and `local_window_size` for the windows. The default stays the masked matmul, which
  also returns attention weights for the figures. On d12 the fused path was 47% faster
  (`BENCHMARK.md`). `--attention cudnn` sets it from the command line.
- **`metrics.flops_per_token`** adds the gate matmuls and counts attention over each layer's
  window, as nanochat's `estimate_flops` does, so MFU stays comparable.

### Optimizer and schedules, `muon.py` and `train.py`

`muon.muon` is an optax transform that matches nanochat's Muon step. A NumPy transcription
of its PyTorch code checks it in `muon_test.py`. For each weight matrix it does the following:

1. Nesterov momentum.
2. Scales each row to equal norm, MuonEq.
3. Orthogonalizes with 5 Polar Express iterations in bfloat16, in place of Newton-Schulz.
4. Rescales the result to √min(rows, cols), Muon+.
5. NorMuon: divides by a running estimate of each row's update size, with β2 0.9.
6. Applies cautious weight decay: only where the update and the weight agree in sign.
7. Multiplies the learning rate by √max(1, rows/cols).

toylib stores weights as [in, out] where PyTorch stores [out, in], so the step transposes on
the way in and out.

`train.create_nanochat_optimizer_config` routes each parameter to its group:

| Group | Optimizer | Learning rate | Betas | Weight decay |
| --- | --- | --- | --- | --- |
| Block matrices | Muon | 0.02 | Momentum 0.85 to 0.97 to 0.90 | 0.28 → 0, cosine, scaled below |
| Output layer | AdamW | 0.008 · s | (0.8, 0.96) | 0.01 |
| Token embedding | AdamW | 0.3 · s | (0.8, 0.995) | 0.001 |
| Value embeddings | AdamW | 0.15 · s | (0.8, 0.995) | 0.01 |
| `resid_lambdas` | AdamW | 0.005 | (0.8, 0.95) | 0.05 |
| `x0_lambdas` | AdamW | 0.5 | (0.96, 0.95) | 0 |
| Smear and backout | AdamW | 0.2 | (0.8, 0.95) | 0 |

- **s = (dim/768)^−0.5**, so wider models take smaller steps on the AdamW groups.
- **Batch scaling.** Every learning rate except smear and backout scales with
  √(batch / 524,288).
- **Weight-decay scaling.** Muon's weight decay scales with √(batch / 524,288) and with
  `D_REF / D`, where `D_REF` is d12's 12 tokens per parameter and `D` is this run's token
  budget. That gives 0.28 for d12 and about 0.106 for d20.
- **Learning rate:** 40 warmup steps, then the peak until 35% of the run, then a linear
  warmdown to 5%, in `warmup_warmdown_schedule`.
- **Muon momentum:** 0.85 to 0.97 over the first 400 steps, back to 0.90 through the
  warmdown, in `muon_momentum_schedule`.
- **Muon weight decay:** a cosine from its start to 0 over the run, in
  `cosine_weight_decay_schedule`.
- **No gradient clipping**, as nanochat does, so `max_grad_norm=0`.

`train.scaling_params` counts parameters nanochat's way, the blocks with their value gates
plus the output layer: 110,100,912 for d12 and 435,160,240 for d20, matching its numbers.

### Runs

`NANOCHAT_OPTIONS` turns on the model and optimizer above; the runs that use it are:

- `RUNS["local_nanochat"]`, the 4-block laptop run.
- `RUNS["d12_nanochat"]`.
- `RUNS["d20_nanochat"]`, which also sets cuDNN attention and saves every 250 steps instead of 500.

## Local check

4 blocks, 256 dims, 600 steps of 8,192 tokens on 8 shards, `RUNS["local_nanochat"]` against
`RUNS["local"]`:

| Eval | 0 | 1 | 2 | 3 | 4 | 5 | Final |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 3.189 | 2.114 | 1.965 | 1.887 | 1.822 | 1.777 | 1.753 |
| nanochat recipe | 3.188 | 2.194 | 1.931 | 1.827 | 1.760 | 1.713 | **1.692** |

It ends 0.06 ahead. That's despite the weight-decay rule being far outside its tuned range:
`D_REF / D` is about 770,000 at this budget. The peak gradient norm was 16 with no clipping, and
training stayed finite.

## d12 with nanochat's recipe

`RUNS["d12_nanochat"]`, the same size, steps and batch as the d12 baseline, with cuDNN attention.
The run would stop early if it was 0.02 worse than the baseline at any evaluation from step
250, or not better by step 750. [W&B run](https://wandb.ai/anujkhare18/toylib-p02-climbmix/runs/d12_nanochat).

**It finished at 0.877 bits per byte on nanochat's 41.9M evaluation tokens, 0.072 or 7.6% below the
baseline's 0.949, and inside nanochat's 0.87 to 0.90 for this size.** It was ahead at every
evaluation.

| Step | 125 | 250 | 500 | 1,000 | 1,500 | 1,875 | 2,125 | 2,500 | Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1.642 | 1.343 | 1.120 | 1.024 | 0.982 | 0.962 | 0.954 | 0.946 | 0.949 |
| nanochat recipe | 1.323 | 1.116 | 1.032 | 0.974 | 0.931 | 0.905 | 0.890 | 0.875 | **0.877** |
| Difference | −0.319 | −0.227 | −0.087 | −0.050 | −0.051 | −0.057 | −0.063 | −0.071 | −0.072 |

Part of the large early lead is the schedule. nanochat holds the peak learning rate until 35%
of the run, while the baseline starts decaying sooner. The gap narrows to about 0.05 by step
1,000, then widens to 0.07 through the warmdown, so the lead isn't only the early schedule.
Which pieces contribute most wasn't measured. Without clipping,
the gradient norm fell from 0.21 at step 210 to 0.12 at step 2,220.

**Speed: about 125,000 tokens per second against the baseline's 85,000**, almost all from
cuDNN attention.

**Runtimes.** Colab ended the first runtime after 2.4 hours, just after the step 2,000
checkpoint. The watcher only reported the loss and didn't restart the run, so it sat for six
hours until it resumed on a new runtime, again without losing steps. For d20 a supervisor
recreates the session and resumes on its own.

**Cost: about 21 compute units** over the two runtimes, against the baseline's 33.5.

## d20 with nanochat's recipe

`RUNS["d20_nanochat"]` trains the post's model with this recipe: 6,640 steps of 524,288 tokens,
3.5B tokens, 8 tokens per parameter, as 32 microbatches of 8.
[W&B run](https://wandb.ai/anujkhare18/toylib-p02-climbmix/runs/d20_nanochat).

**It finished at 0.783 bits per byte on nanochat's 41.9M evaluation tokens**, inside the 0.75
to 0.90 the health run set as healthy, and 0.094 below d12 with the same recipe. It improved at
every evaluation, and was still falling about 0.003 per 250 steps at the end.

| Step | 250 | 500 | 1,000 | 2,000 | 3,000 | 4,000 | 5,000 | 6,000 | Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d20 | 1.078 | 0.988 | 0.929 | 0.893 | 0.863 | 0.835 | 0.811 | 0.791 | **0.783** |

Without clipping, the gradient norm ended at 0.12, as d12's did.

**Speed: about 46,000 tokens per second at 42.6% MFU**, against the baseline's 31,500 and
32.7%. That's 1.46 times faster, close to the gain cuDNN attention gave d12. The run took 21.3
hours on one A100 runtime, against the 31 planned, and never needed to resume.

**Cost: about 145 compute units**, against the 209 planned.

**Not yet checked:** samples. The run printed none, so whether they read as real text, the
other half of milestone 4, needs the checkpoint.
