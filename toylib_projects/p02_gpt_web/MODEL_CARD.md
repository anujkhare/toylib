# Model card: toylib d20

Two checkpoints of one model, trained from scratch in JAX for post 2, "Does a GPT read, or just
remember?":

- **`d20_nanochat`, the base model:** a 20-layer decoder-only transformer pre-trained on 3.5B
  tokens of ClimbMix web text.
- **`d20_sft`, the chat model:** the base model fine-tuned on nanochat's conversation mixture.

The recipe follows Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) as of
mid-2026, ported to toylib's own layers. `EXPERIMENTS.md` has the full record of every run.

| | `d20_nanochat` | `d20_sft` |
| --- | --- | --- |
| Use | Text continuation | Chat |
| Trainable parameters | 896,533,746 | 896,656,626 |
| Vocabulary | 32,768 | 32,776 (8 chat tokens added) |
| Context | 2,048 tokens | 2,048 tokens |
| Training tokens | 3.48B | 489M more, 294M of them trained |
| Validation bits per byte | 0.783 (ClimbMix) | 0.295 (chat mixture, assistant tokens) |
| CORE | 0.2218 | Not measured |
| ChatCORE | −0.012 (plain transcript) | **0.182** |
| A100 hours, training | 21.3 | 2.3 |
| Weights | `gs://tinystories-checkpoints/toylib-p02-climbmix/d20_nanochat` | `gs://tinystories-checkpoints/toylib-p02-climbmix/d20_sft` |

## Architecture

A pre-norm decoder-only transformer (`decoder_only_model.py`) with nanochat's current
additions (`decoder_only_model.NANOCHAT`):

| Setting | Value |
| --- | --- |
| Layers | 20 |
| Model width | 1,280 (64 × depth) |
| Attention heads | 10 query and 10 key-value heads, 128 dimensions each (no grouped-query attention) |
| MLP | 1,280 → 5,120 → 1,280, ReLU² activation, no biases |
| Normalization | RMSNorm with no learned scale: before attention, the MLP and the output layer, and on queries and keys |
| Positions | RoPE, base 100,000 |
| Attention | Causal, with sliding windows in the pattern SSSL: three layers see the previous 512 tokens, the fourth all 2,048. The last layer always sees everything. |
| Q and K scale | 1.2, after their RMSNorm |
| Value embeddings | A second token embedding on alternating layers (10 of 20, always including the last), added to the attention values through a per-head gate, `3·sigmoid(x[:, :12] @ W)` |
| Residual scalars | Before each block, `x = resid_λᵢ·x + x0_λᵢ·embedding`, both learned |
| Smear | Mixes the previous token's embedding into each position, through a gate on its first 24 channels |
| Backout | Subtracts a learned share of the residual from after the middle block, just before the output layer |
| Embeddings | The input embedding and the output layer are separate (untied) |
| Logits | Soft-capped at 15 with `15·tanh(x/15)` |
| Precision | float32 weights, bfloat16 compute |

### Parameters

| Component | Parameters |
| --- | --- |
| Attention projections (Q, K, V, out), 20 layers | 131,072,000 |
| MLPs, 20 layers | 262,144,000 |
| Value-embedding gates | 1,200 |
| Token embedding | 41,943,040 |
| Value embeddings, 10 tables | 419,430,400 |
| Output layer | 41,943,040 |
| Residual scalars, smear and backout | 66 |
| **Total trainable** | **896,533,746** |

The RoPE tables (5,242,880 values) are stored with the weights but frozen.

Nearly half the parameters are value-embedding tables, which are lookups and cost almost no
compute. nanochat's count for scaling laws takes the blocks' matrices, their gates and the
output layer: **435,160,240**, which `train.scaling_params` reproduces. Training budgets
below are in tokens per parameter by that count.

The chat model's 8 extra rows add 10,240 parameters to each of the token embedding, the
output layer and the 10 value-embedding tables, 122,880 in all.

## Tokenizer

- **Type:** byte-level BPE, trained by our own `tokenizer/bpe.py`. It uses nanochat's
  pre-tokenization split pattern and runs as plain Python adapted from
  [minbpe](https://github.com/karpathy/minbpe).
- **Size:** 32,768 ids: 256 bytes, `<|endoftext|>` at id 0, and 32,511 merges. Every byte has
  an id, so there's no unknown token.
- **Training data:** 8 ClimbMix shards, about 2B characters, as nanochat uses.
- **Chat tokens:** the chat model appends nanochat's 8 at ids 32,768 to 32,775:
  `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`,
  `<|python_start|>`, `<|python_end|>`, `<|output_start|>` and `<|output_end|>`.
- **Location:** `gs://tinystories-checkpoints/toylib-p02-climbmix/data_shards120/tokenizer_32768.json`.

## Training

### Stage 1: pre-training (`train.py`, `RUNS["d20_nanochat"]`)

**Data.** [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle), the
web-text mixture nanochat trains on:

- **Train:** 120 shards of about 54M tokens each. Every document starts with
  `<|endoftext|>`, and documents are concatenated into rows of 2,048.
- **Seen:** 3.48B tokens, about half of the 120 shards, so less than one epoch and no
  repeats.
- **Validation:** the last shard, as in nanochat. The final number is on 41.9M tokens.

**Budget.**

- **Steps:** 6,640 steps of 524,288 tokens (256 rows of 2,048), as 32 microbatches of 8 with
  rematerialization.
- **Tokens per parameter:** 8, by nanochat's count.

**Optimizer.** Muon for the block matrices and AdamW for everything else, with nanochat's
groups and rates:

| Group | Optimizer | Peak learning rate | Betas or momentum | Weight decay |
| --- | --- | --- | --- | --- |
| Block matrices | Muon (Polar Express, NorMuon, cautious decay) | 0.02 | 0.85 → 0.97 → 0.90 | ~0.106 → 0, cosine |
| Output layer | AdamW | 0.008 · s | (0.8, 0.96) | 0.01 |
| Token embedding | AdamW | 0.3 · s | (0.8, 0.995) | 0.001 |
| Value embeddings | AdamW | 0.15 · s | (0.8, 0.995) | 0.01 |
| Residual scalars | AdamW | 0.005 and 0.5 | (0.8, 0.95) and (0.96, 0.95) | 0.05 and 0 |
| Smear and backout | AdamW | 0.2 | (0.8, 0.95) | 0 |

- **Width scaling:** s = (1280/768)^−0.5.
- **Schedule:** 40 warmup steps, then the peak until 35% of the run, then a linear decay to
  5% of it.
- **Gradient clipping:** none.

**Result.** It reached 0.783 validation bits per byte and was still improving about 0.003
every 250 steps.

**Compute.** One Colab A100 40 GB:

- **Throughput:** about 46,000 tokens a second, 42.6% MFU, with cuDNN's fused attention.
- **Time and cost:** 21.3 hours, about 145 Colab compute units.

### Stage 2: supervised fine-tuning (`sft.py`, `RUNS["d20_sft"]`)

The current nanochat runs a single conversation fine-tuning stage, `chat_sft.py`; midtraining
was merged into it in January 2026. This stage ports that one.

**Data** (`sft_data.py`):

| Source | Conversations | Epochs | Purpose |
| --- | --- | --- | --- |
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) train | 460,341 | 1 | General conversation |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) auxiliary_train | 99,842 | 3 | Multiple-choice answer format |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) train | 7,473 | 4 | Math, with calculator calls |

- **Size:** 789,759 conversations, shuffled together, 485.5M tokens in all.
- **Loss mask:** the loss covers only the assistant's tokens, 294M of them.
- **Calculator:** GSM8K's `<<12/60=0.2>>` spans become calculator calls. The call is trained,
  but the calculator's result isn't.
- **Packing:** each row of 2,049 tokens starts a conversation, then takes the longest of the
  next 100 that still fits. 239,000 rows are 99.1% full, and no conversation is cut.
  Conversations are capped at 2,048 tokens.
- **Validation:** 29,849 conversations, from SmolTalk's test split, the first 5,200 MMLU test
  questions and the first 420 GSM8K test problems.

**Budget.** One pass: 933 steps of 256 rows.

**Optimizer.** It continues from pre-training's optimizer state at step 6,640:

- **Learning rates:** every group starts at 0.8 of its pre-training peak, holds there for half
  the run, then falls linearly to 0.
- **Muon:** momentum ramps from 0.85 to 0.95 over 300 steps, and its weight decay is 0.
- **New rows:** the 8 chat-token rows start from the initializer, with empty optimizer moments.

**Compute.** One Colab A100 40 GB:

- **Training:** 2 h 20 min, about 9 s a step.
- **Evaluation:** another 1 h 30 min, for validation and ChatCORE every 200 steps and the full
  ChatCORE at the end.
- **Cost:** about 28 compute units.

**Logs.** This run logged to a file, not W&B. The log, `chatcore.json` and every graded
completion are in `gs://tinystories-checkpoints/toylib-p02-climbmix/d20_sft_outputs`.

### Total budget

| | A100 hours | Colab compute units |
| --- | --- | --- |
| Pre-training | 21.3 | ~145 |
| Fine-tuning, with its evaluations | 3.8 | ~28 |
| CORE and base ChatCORE | ~1 | ~9.5 |
| **Total** | **~26** | **~183** |

These are the final runs only. The health run, the d12 runs and the benchmarks behind them are
in `EXPERIMENTS.md` and `BENCHMARK.md`.

## Evaluation

### CORE (base model)

CORE is the DCLM paper's 22 in-context tasks. Each is centered on chance, so 0 is chance and 1
is perfect, and then they're averaged. `scripts/core_eval.py` ports nanochat's evaluation. It
reproduces GPT-2 124M's reference score to within 0.01 (0.1217 against 0.1139), so treat
scores as ±0.01.

| Model | Parameters | Training tokens | CORE |
| --- | --- | --- | --- |
| GPT-2 124M | 124M | | 0.1139 |
| **d20_nanochat** | 897M stored, 435M by nanochat's count | 3.5B | **0.2218** |
| nanochat d20, October 2025 | 561M | 11.2B | 0.2219 |
| GPT-2 1.5B | 1.5B | | 0.2565 |

Per task, it's ahead of GPT-2 1.5B on ARC, OpenBookQA and SQuAD, and behind it on recall
(Jeopardy, Wikidata QA), LAMBADA, the Winograd tasks and copying. `EXPERIMENTS.md` has all 22.

### ChatCORE (chat model)

ChatCORE is nanochat's chat benchmark:

- **Multiple choice, chance 25%:** ARC-Easy, ARC-Challenge and MMLU.
- **Generated greedily, chance 0%:** GSM8K and HumanEval, with up to 512 new tokens.
- **Score:** each task is centered on chance, then the five are averaged.

`scripts/chat_eval.py` ports nanochat's tasks, prompts, answer extraction and code tests. GSM8K
runs with the calculator.

| Task | Chance | Base, chat format | **d20_sft** | nanochat d20 SFT, October 2025 |
| --- | --- | --- | --- | --- |
| ARC-Easy | 0.25 | 0.245 | **0.542** | 0.388 |
| ARC-Challenge | 0.25 | 0.258 | **0.436** | 0.281 |
| MMLU | 0.25 | 0.255 | **0.345** | 0.315 |
| GSM8K | 0 | 0.000 | **0.030** | 0.046 |
| HumanEval | 0 | 0.000 | **0.116** | 0.085 |
| **ChatCORE** | 0 | 0.002 | **0.182** | 0.088 |

- **Comparison with nanochat:** its base model had the same CORE as ours. It was fine-tuned on
  an older, smaller mixture, so most of our lead is probably the newer recipe, which repeats
  MMLU and GSM8K. That hasn't been checked against nanochat's current code.
- **Noise:** HumanEval has only 164 problems. During training, on a 24-problem subset, it
  swung between 0.04 and 0.17.

### What fine-tuning changed

- **Multiple choice:** the gain is the answer format, and most of it came in the first 200
  steps. CORE's in-context ARC-Easy was already 65%; in the chat format the base model picked
  letters at chance.
- **GSM8K:** 565 of 1,319 replies end with a `#### <number>` answer and 458 use the calculator,
  against none before. Only 39 are right: the arithmetic is done by the calculator, but the
  steps are often set up wrongly.
- **HumanEval:** 160 of 164 replies write a function body, against none before, and 19 pass.

## Intended use

- **For:** studying and teaching how small language models train, as the subject of a blog post
  and its figures.
- **Base model:** continues text.
- **Chat model:** answers short questions in nanochat's chat format, on the local
  `scripts/chat_web.py` page or through `generate.py`. A Mac M4 runs it at about 29 tokens a
  second.

## Limitations

- **Accuracy:** it states wrong facts fluently. Its knowledge comes from 3.5B tokens of web
  text, and it is weak at recall-heavy tasks.
- **Reasoning:** math and code are poor. GSM8K is 3% even with a calculator, and HumanEval 12%.
- **Context:** 2,048 tokens. Three of every four layers see only the last 512.
- **Language:** ClimbMix is mostly English, and other languages were not evaluated.
- **Safety:** there was no preference tuning, RLHF or safety training, and no output
  filtering. SmolTalk's conversations are the only alignment it has had. Don't put it in front
  of users.
- **Biases:** web text carries social biases and errors, and none of them were measured.
- **Tools:** the calculator works only where the decoding loop supplies results
  (`chat_eval.py`). The chat page doesn't run it, so there the model guesses the results.
- **Greedy decoding:** the base model tends to repeat itself.
- **Variance:** each result is a single run with a single seed.

## Reproducing

The commands are in `README.md` under "Commands": training, then "Evaluating and talking to d20"
and "Fine-tuning d20 to chat".

## Citation and license

- **Built on:** nanochat (Karpathy, 2025–2026), for its recipe, data choices and evaluations.
  The data is ClimbMix, SmolTalk, MMLU and GSM8K, each under its own license.
- **Repository:** github.com/anujkhare/toylib.
- **License:** the repository doesn't have one yet, so none is granted for these weights.
