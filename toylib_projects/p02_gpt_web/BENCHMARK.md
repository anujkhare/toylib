# d20 hardware, configuration and cost

Measured on 2026-11-01 and 02 on a Colab `NVIDIA A100-SXM4-40GB`, with JAX 0.11.1, the real
`d20` model (477M parameters) and real ClimbMix tokens. `scripts/benchmark.py` and
`scripts/health_check.py` reproduce the numbers. Training runs and their results, including
the health run and the d12 runs, are in `EXPERIMENTS.md`.

**Recommendation: train on an A100 with 32 microbatches of 8, remat on and a context of 2,048.
At a batch of 524,288 tokens it runs at 31,500 tokens per second, and 3.5B tokens take about
31 hours and 209 compute units.** It peaks at 19.7 GiB, so it fits the 40 GB card. A 3% trial
run on this setup passed every health check (`EXPERIMENTS.md`).

An H100 couldn't be tested, because Colab had none to assign. It's likely the faster choice,
but whether it's cheaper depends on a rate we couldn't see. Use one if Colab offers it at under
about 14.8 compute units an hour.

## H100 against A100

| | A100, 40 GB | H100 |
| --- | --- | --- |
| Status | Measured | **Not measured** |
| Tokens per second | 31,500 | About 69,000, estimated at 2.2 times the A100 |
| Time for 3.5B tokens | 31 hours | About 14 hours, estimated |
| Compute units an hour | 6.73 | Unknown |
| Compute units for 3.5B tokens | 209 | 14 × its rate |
| Break-even rate | | 14.8 an hour, assuming the 2.2 times |

Every H100 request failed with a 503, Colab's capacity error, for about 1.7 hours. A 503 is
not a quota refusal, so it may work at another time, but it means an H100 can't be planned on.

The 2.2 times has two independent sources: MosaicML measured it on 8 × H100 against A100 at
bfloat16, and nanochat's own reports give 2.3 (135,000 against 59,000 tokens per second per
GPU). It's well below the 3.2 times that peak FLOPs suggest. Neither source used this JAX code
or its attention, so treat it as an estimate.

To decide on the day, request the H100 and read `colab --auth=adc usage`.

## Configurations at 524,288 tokens

All on the A100, in full training steps. The first number is microbatches, the second sequences
in each.

| Configuration | Tokens/s | Step | Peak | Verdict |
| --- | --- | --- | --- | --- |
| **32 × 8, remat** | **31,520** | **16.6 s** | **19.7 GiB** | **Chosen** |
| 16 × 16, remat | 31,761 | 16.5 s | 30.7 GiB | 0.8% faster, but needs 97% of the card |
| 32 × 8, recompute everything | 30,123 | 17.4 s | 15.1 GiB | 4.4% slower for 4.6 GiB less |
| 32 × 16, context 1,024, remat | 37,261 | 14.1 s | 19.7 GiB | 18% faster, but a shorter context |
| 64 × 8, context 1,024, no remat | 29,243 | 17.9 s | 31.8 GiB | Worse on both counts |

And at 64 sequences a step, which is 131,072 tokens:

| Configuration | Tokens/s | Step | Peak |
| --- | --- | --- | --- |
| 8 × 8, remat | 30,587 | 4.28 s | 19.7 GiB |
| 4 × 16, remat | 31,149 | 4.21 s | 30.7 GiB |
| 32 × 2, remat | 27,989 | 4.68 s | 12.0 GiB |
| 32 × 2, no remat | 20,839 | 6.29 s | 25.4 GiB |
| 8 × 8, remat on attention only | 24,478 | 5.35 s | 36.9 GiB |
| 4 × 16, softmax in bf16 | 30,877 | 4.24 s | 30.7 GiB |
| 4 × 16, latency-hiding scheduler | 31,189 | 4.20 s | 30.7 GiB |
| 16 × 4 or 4 × 16, no remat | Out of memory | - | Asked for 35 and 134 GiB |

**The big batch costs nothing in speed.** It's 3% faster per token than the smaller one,
because the optimizer runs a quarter as often.

**The context is the one real lever.** A context of 1,024 tokens saves 15% of the time, or 31
compute units on the full run, at the same memory. Only 8.8% of documents are longer than
1,024 tokens, but those hold 20.7% of the text beyond that point. This keeps 2,048 anyway: it
matches nanochat, so `val/bits_per_byte` stays comparable, and 31 units is about 1.5% of the
balance. Set `seq_len=1_024` and `batch_size=512` if time matters more.

## Why 524,288 tokens and 3.5B tokens

nanochat tunes its learning rates at this batch size, which it calls `B_REF`. At the earlier
131,072 it would halve them, and we weren't. Running at `B_REF` means they apply as they are.
For this size of model, its batch-size rule predicts 2^20, or 1M tokens, so 524,288 is the lower
of two sensible choices. It's also the batch this report sets out to fit.

**Why not 12 tokens per parameter?** There was no good reason. It's `base_train.py`'s default,
which I took without checking it. nanochat's speedrun passes 8 instead, and calls 10.5 compute-
optimal. Going below that leaves the model slightly undertrained, which it accepts to save time.

**We were also counting differently.** nanochat counts only the blocks and the output layer,
which is 435M here. Our 5.7B came from multiplying 12 by the total of 477M.

| Tokens per parameter, nanochat's count | Tokens | Steps | A100 hours | Compute units |
| --- | --- | --- | --- | --- |
| **8, its speedrun** | **3.5B** | **6,640** | **31** | **209** |
| 10.5, its compute-optimal comment | 4.6B | 8,700 | 41 | 275 |
| 12, its default | 5.2B | 9,960 | 47 | 314 |
| The previous plan, 12 × 477M | 5.8B | 11,000 | 52 | 347 |

`train.RUNS["d20"]` now uses 8. It's the cheapest of these, it's what nanochat ships, and 3.5B
tokens sits inside the 2.5B to 5B where Olsson et al. saw induction heads, which is what the post
needs. The previous 5.8B was above that range. nanochat's scaling sweep suggests more tokens would
lower the final loss somewhat, so raise `max_steps` if the final number matters more than cost.

These hours are 16.87 s per step, of which evals and samples add 0.165 s and checkpoints 0.077 s.
That's about $21 at Colab's reported $0.10 a unit. The balance is 2,018 units.

A runtime reportedly lasts about 24 hours at most, and isn't guaranteed, so plan on at least 2
sessions. A lost runtime costs at most 500 steps, about 2.3 hours.

## Where the time goes

The GPU stays at about 99% and 380 W of its 400 W limit. Step times held steady, and data takes
3 ms a step, so **we're GPU-bound and data loading doesn't matter.**

| Check | Result |
| --- | --- |
| Data loading | 3 ms of a 16.6 s step |
| Attention | About 30% of a step, against 19% of its FLOPs |
| Optimizer | About 0.2 s a step, 1% at this batch size |
| bf16 matmuls | 241 TFLOP/s against 111 for float32, so tensor cores are working |
| MFU | 32.7% |

The 30% comes from halving the context, which removes half of attention's cost and saved 15%. A
fused attention kernel would attack this. I wrote here that JAX has none, but it does, through
cuDNN; see "Fused attention" below.

**How we compare.** 31,500 tokens per second is 82% of 40% MFU (38,500), 68% of nanochat's 48%
MFU applied to this model (46,200), and 53% of the 59,000 derived from a community 8 × A100 run.
That last figure implies 66% MFU on an A100, which is high, and it's a different model using
fused attention. Your April d20 run managed 26,600, assuming the April notebook's batch.

## Corrections to the first version of this report

- **Remat is not a bottleneck.** I said it cost about 20%, from assuming a gradient costs 3 times
  a forward pass. Measured directly, 32 × 2 is **26% faster with remat** (4.68 s against 6.29 s).
  The policy already keeps MLP and linear outputs, and only attention is recomputed.
- **An 80 GB card wouldn't gain 38,000 tokens per second.** That followed from the same mistake.
- **The H100 row was wrong.** It scaled tokens per second by peak FLOPs and ignored FLOPs per
  token. 48% MFU on this model is 46,200, not 42,600, and an H100's MFU doesn't transfer to an A100.
- **A checkpoint is 3.5 GB, not 6 GB.**

## Metrics

| Metric | Why it matters | Status |
| --- | --- | --- |
| `val/bits_per_byte` | Doesn't depend on the tokenizer | Already correct, now on more text |
| `val/loss`, `train/loss` | A gap between them means overfitting | Already there |
| `train/grad_norm` | Spikes and NaNs show first here, and clipping hides them | **New** |
| `train/update_to_weight` | About 0.1% is healthy; far off means a bad learning rate | **New** |
| `train/lr_multiplier` | Confirms the schedule | **New** |
| `train/tokens_seen` | The axis for comparing with other runs | **New** |
| `train/tokens_per_second`, `train/mfu` | Throughput, including evals and saves | **New** |
| Samples | A plain sanity check | Already there |

`val/bits_per_byte` matches nanochat's `evaluate_bpb`: total bits over total bytes of the target
tokens, with special tokens left out. Both use the last shard, `shard_06542`, so ours compares
directly with its published numbers. One difference remains: we average per batch rather than
summing, which is negligible at these batch sizes.

Each eval now reads 3.3M tokens, up from 0.3M, and takes 27 s. nanochat reads 41.9M, so the
notebook does one final pass of that size to give a comparable number.

## Fused attention

`jax.nn.dot_product_attention(implementation="cudnn")` runs cuDNN's fused kernel, with causal
masking and sliding windows built in. On `RUNS["d12_nanochat"]`, in a 4-step benchmark at 16
microbatches of 16 on an A100:

| Attention | Tokens per second | MFU | Peak memory |
| --- | --- | --- | --- |
| Masked matmuls, the default | 90,900 | 22.1% | 19.7 GiB |
| cuDNN, `--attention cudnn` | **133,300** | **32.5%** | 19.6 GiB |

That's 47% faster at the same memory. Full training ran at about 125,000 tokens per second,
evaluations included. `RUNS["d20_nanochat"]` uses it; d20's speed with it isn't measured yet.

## Reproducing

Copy the token files to `/content/data` on the runtime, then:

```bash
python -m toylib_projects.p02_gpt_web.scripts.benchmark --data-dir /content/data --steps 6 --eval --checkpoint
```

```bash
python -m toylib_projects.p02_gpt_web.scripts.health_check --data-dir /content/data --steps 200
```

`--no-remat`, `--microbatch-size`, `--microbatches`, `--seq-len` and `--dtype` set the options
above. 16 × 16 needs `XLA_PYTHON_CLIENT_MEM_FRACTION=0.97` on a 40 GB card.

## Sources

- nanochat: [README](https://github.com/karpathy/nanochat), `base_train.py`, `estimate_flops`, `get_peak_flops`, `runs/speedrun.sh`, `dev/LOG.md`
- [8 × A100 d20 report](https://cloudfare-proxy.pages.dev/spaces/nanochat-students/README/discussions/9)
- [nanochat model card, H100 throughput and MFU](https://huggingface.co/HarleyCooper/nanochat561)
- [MosaicML H100 and A100 benchmark](https://www.databricks.com/blog/coreweave-nvidia-h100-part-1)
- [nanochat JAX port on TPUs](https://github.com/tucan9389/nanochat-jax/discussions/1)
- [nanoGPT model sizing and FLOPs](https://deepwiki.com/karpathy/nanoGPT/7.2-model-sizing-and-flop-estimation)
- [Colab compute unit prices, secondary source](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)
- [Colab limits, secondary source](https://uw-madison-datascience.github.io/ML-X-Nexus/Toolbox/Compute/GoogleColab.html)
