# Post 2: How does a small GPT keep a story straight?

A character named Lily appears in sentence one and has to still be Lily in sentence
ten. Nothing in a transformer is built to remember her, and yet TinyStories models do.
This post follows a small GPT's prediction of one story, token by token, and looks
inside for how it reads back. As in post 1, the aim is intuition through looking at the
network, not a measured comparison of architectures.

This replaces the planned character-level Shakespeare post. Its central experiment
survives: TinyStories asks the same question about copying, and characters' names make
it far more natural than random strings. `decoder_only_model.py` also already exists
and handles BPE tokens, so a character vocabulary would have added work and no insight.

## The story

Each section ends on a problem that the next one solves. *First version* marks what
`figures.py` already draws from the small local model. Those figures are placeholders
until the TPU run, and `POST.md` holds predictions made before seeing them.

1. **The hook.** Hand the model a name it has never seen. Does it use the name again?
   *First version (`names`).* Today it almost never does, so this is the before picture:
   does the TPU run change that?
2. **What a story costs.** *First version (`surprise`).* Colour each token by how
   many bits it took to predict. A first mention is expensive; a repeat is cheap. So
   where does the model get what it needs to predict a repeat?
3. **What counting achieves.** *First version (`baselines`).* Word frequency, then bigram
   and trigram tables, on the same axis as the model. A table can't remember Lily, so
   the gap between it and the model is what reading earned.
4. **Learning to write.** *First version (`samples`, `training`).* One prompt continued
   at steps 0, 10, 100, 1k and the end: noise, then common words, then grammar, then
   something like a plot.
5. **How far back does it read?** *First version (`context`).* Limit the context and
   measure the loss. It's a direct answer that needs no interpretation of weights.
6. **Who reads whom.** *First version (`attention`, `attention_distance`).* Attention over
   one passage, block by block, then how much weight each head puts at each distance.
   Attention shows where a block reads from, but not what it does with it.
7. **What each block adds.** *First version (`lens`).* Send each block's state through
   the output layer (the logit lens) and watch a prediction form with depth.
8. **Does it copy?** *First version (`copying`).* The experiment. Measured at every
   checkpoint, with each name as a control for the others:
   - **Finish an invented name:** "...named Spoundo... Sp-ound" -> "o"? This is the
     [A][B]...[A] -> [B] pattern of induction heads, on token pairs absent from
     training.
   - **Bring an invented name back:** does its first token get likelier at a slot where
     a character could appear?
   - **Repeat a random string:** the classic test, and the hardest, since a story never
     looks like this. The control is the same tokens shuffled.
9. **Words that fill the same slots.** *First version (`embeddings`).* Names, animals and
   colours in the output layer, which has grouped them, and in the input embeddings,
   which haven't yet, perhaps because they start random while the output starts at zero.
   Keep as a supporting view: it doesn't explain copying.
10. **Where it fails.** Contradictions the model writes, and what attention was doing at
    the time. Needs the better model, because this one fails everywhere.

**Later, with a better model:**

- Find a copying head directly: rank heads by attention from the second mention to the
  token after the first, then zero that head and see the copy margin fall.
- An interactive page in the spirit of `p01_cnn_mnist/explorer`: edit a story and watch
  surprise, attention and the top guess update. Sampling and the forward pass for a
  small model are feasible in the browser; check the timing first, as in post 1's plan.
- If random repetition never appears, train a second run with repeated strings mixed in
  and show both, as the original plan proposed.

## Results so far

The `local` run: 4 blocks, 256 dims, 7.3M parameters, 5,000 steps at batch 32, about
41M tokens (7.8% of one pass, 28% of the Chinchilla estimate), 88 minutes on a laptop
CPU, on our own 8,192-token tokenizer. Validation loss was 1.74 nats at step 4,750, the
last eval; train loss was about 1.73 at the end, so it isn't overfitting. The curve was
still falling.

**The result so far: the model keeps names it knows, and barely keeps names it doesn't.**
Common names came back in 68 of 72 sampled stories, and invented ones in 3 of 288.
Reading an invented name does change what comes next (a margin of 3.0 nats for finishing
it, 3.8 for bringing it back), but from almost nothing to very little: the slot where a
name could go gives it about 1 in 800, and 17 of the 24 names stay under 1 in 100. A
large margin alone would have told the wrong story, which is why `names` draws the
probabilities.

| Figure | Takeaway | Measured |
| --- | --- | --- |
| `training` | Most of the loss goes in the first 500 steps | 9.0 to 2.5 nats by step 500, 1.74 by 4,750 |
| `baselines` | Tables stop well short of the model | 8.50, 5.35, 4.47 bits; model 2.45 |
| `surprise` | A rare word gets cheaper once it has appeared; the commonest tokens don't | Rarest band 6.8 to 3.0 bits; commonest 0.8 to 1.2 |
| `samples` | Readable by step 100, characters hold by step 1,000 | Plot still wanders at the end |
| `attention` | Averaged over heads, nothing stands out on one passage | No single strong link |
| `attention_distance` | Block 1 reads widely; blocks 2 to 4 stay close | Dips at 16+ tokens again; they survived four times the windows on the last model, and RoPE is a guess |
| `context` | Reading back helps up to about 64 tokens, then stops | 5.59 bits at 1 token, 2.55 at 64, 2.50 at 255 |
| `lens` | Each block helps, and only the last says a name | Loss 11.6, 7.3, 5.9, 4.5, 2.5 bits; finish margin 0, 0, -0.1, 0.5, 3.0 nats |
| `copying` | Names are read; the order of a random string isn't | Final margins 10.7, 3.8, 3.0, 0.04 nats |
| `names` | Known names come back; most invented ones stay unlikely | P(name) about 0.37 for common, 0.001 for invented |
| `embeddings` | The output layer has sorted words into groups; the embeddings haven't | Within-group cosine 0.27 against 0.03 |

**Our tokenizer against GPT-2's top 8,192 tokens,** same model and steps: 0.604 bits per
byte at the last eval against 0.598. Text costs about the same, and now every text can be
encoded with no `<unk>`. Nats and bits per token are not comparable between the two,
since the tokens differ, and that includes the table baselines. The copying results
changed for a different reason: the old ones came from five names.

**Read these carefully:**

- **Twenty-four names changed the picture.** Five names gave a finish margin of 5.4 nats
  and a bring-back margin of 3.0. These 24 give 3.0 and 3.8, so the old order of the two
  was luck. The names were fixed before any model was measured on them. No two tokens of
  a name occur side by side in training, and they are kept as the search found them.
- **The averages still hide the spread.** Finish margins run from -0.96 to 8.8 nats,
  with a median of 2.2, and four are negative. Three names are really finished: Snuda,
  Frogy and Spoundo, above P = 0.5.
- **Bringing a name back is scored on its first token.** The best are ` Bree`, ` Star`
  and ` Bloom`, at 0.13, 0.09 and 0.07, which the model can use for other words and names.
- **Any invented name raises the others.** Against a story with no name, the name's first
  token gains 8.3 nats when it's the name in the story, and 4.5 when a different invented
  name is. The margin of 3.8 is what's left. The common names show no such thing: 10.5
  and -0.2 nats. The old model's displacement of 3.9 nats, where a different character
  pushed a common name down, is gone on this tokenizer, so that finding wasn't solid.
- **Random strings needed a better control.** Against an unrelated string, a repeat
  saves 0.16 nats (± 0.09 over six seeds). A shuffled second copy saves the same 0.16, so
  that's a token getting cheaper after it has appeared, not reading in order. `copying`
  now uses the shuffle, over 256 strings.
- **The lens understates early blocks.** The output layer only ever trains on the last
  block's state, so a name appearing "only in block 4" may partly be a limit of the method.
- **Context can't exceed the story.** Stories average about 200 tokens, so the flat end
  of `context` is partly that nothing useful lies further back.

## What changed from `tinystories`

The review found these, each now covered by a test. Fixes 1 to 5 change results, so
treat any number from before this move as out of date.

| # | Problem | Effect | Fix |
| --- | --- | --- | --- |
| 1 | RoPE's `sin`, `cos` and `gamma` tables were trainable | Positions drifted, and every block carried and optimized three extra arrays | Computed from position in `__call__`, so they hold no state. Old checkpoints no longer restore |
| 2 | Bits per byte looked up `inputs`, and averaged per-token ratios | Wrong token, wrong formula: 1.44 where the true value was 0.96 in a small case | Total bits over total bytes of the targets, as in nanochat |
| 3 | Bytes per token decoded tokens one at a time | Tokens holding part of a character counted as 3 bytes, so a curly quote cost 6 instead of 3 | Counts GPT-2's byte-level characters |
| 4 | Text datasets ended after one pass, and each eval read different batches | Training crashed at the end of an epoch; eval drifted through the validation set, then silently stopped | They rewind at the end; `repeatable=True` for validation |
| 5 | `sample` wrote past the end of its buffer | Past `seq_len`, tokens silently overwrote each other | The buffer slides, as nanoGPT crops |
| 6 | Depths such as 5 and 9 gave head counts that don't divide the model dim | Crash inside `einops` | `find_num_heads`, and `ModelConfig` checks up front |
| 7 | Sampling prompts had no BOS, though training text always does | Prompts started in a state the model never saw | Prepended |
| 8 | Restoring a dataset dropped its token buffer, and repeated a row group | Tokens lost or read twice on resume; the old `xfail` test | Both fixed; the test passes |
| 9 | One token fell between consecutive batches | Small; a gap at each batch boundary | The last target stays in the buffer |
| 10 | Comment said microbatches multiply the batch size | They split it | Comment corrected |

Also new: `return_intermediates` on the model (`blocks.k.attn_weights`, `blocks.k.hidden`),
and `save_steps` and `save_final` in `CheckpointConfig`, for dense early checkpoints and a
final one, which the interval used to miss.

## Compared with nanoGPT

[nanoGPT](https://github.com/karpathy/nanoGPT) is the classic GPT-2; our model follows
its successor [nanochat](https://github.com/karpathy/nanochat), so the two differ in
architecture by design.

**Ported:**

| From nanoGPT | Here | Result |
| --- | --- | --- |
| Random windows from a memory-mapped file (`get_batch`) | `TokenBinDataset` | Fast, needs no text at training time, and batches aren't one contiguous run |
| Warmup then cosine decay to a tenth of the peak (`get_lr`) | `warmup_cosine_schedule` | Used by every TinyStories run; optional in `create_experiment` |
| Gradient clipping at 1.0 | `TinyStoriesRun.max_grad_norm` | On by default for these runs |
| Fixed batches for eval (`estimate_loss` averages many) | `repeatable=True`, `num_eval_steps=20` | Eval curves compare like with like |
| Output layer on the last position only when sampling | `model.hidden` + `model.logits` | Skips most of the output layer's cost per sampled token |
| Cropping context when sampling; `top_k` capped at the vocabulary | `sample` | Fixes bug 5 |
| AdamW with nanoGPT's betas and decay | `create_adamw_optimizer_config` | Kept as a baseline; see below |

**Measured, not ported:** Muon on the blocks stays the default. On a 4-block, 128-dim
model at step 600, validation loss was 2.45 for nanochat's Muon and AdamW split, against
2.85 for AdamW at 3e-3 and 3.35 at 1e-3, for about 7% more time per step. That's one
short run per setting, not a tuning sweep.

**Not ported, on purpose:**

- **Weight tying.** nanochat dropped it, and the small vocabulary already removes most of
  the cost it would save.
- **Learned positions, LayerNorm, GELU, biases.** RoPE, RMS norm and ReLU² are the newer
  choices, and the reason this model follows nanochat.
- **Dropout.** nanoGPT's own default is 0 for pretraining.
- **Padding the vocabulary to a multiple of 64.** It suits GPU tensor cores, and does
  nothing useful with 8,192, which already is a multiple.
- **MFU reporting.** Useful on TPUs, so worth a look then. Steps per second is enough here.

**Not from nanoGPT, but bigger than all of the above:** our own tokenizer. nanoGPT has no
trainer; it loads GPT-2's encoding from `tiktoken`. nanochat trains one, with a compiled
`rustbpe` for training and `tiktoken` for encoding, which would mean two more
dependencies. `tokenizer/bpe.py` keeps nanochat's split pattern and adapts
[minbpe](https://github.com/karpathy/minbpe)'s plain Python, with `re` from the standard
library and pair counts updated as it merges, so training on all of TinyStories takes
minutes. Every byte has an id, so there is no unknown token. At 8,192 tokens it averages
4.18 bytes per token, and every common name and all 32 words in `WORD_GROUPS` are single
tokens. Going from GPT-2's 50,257 tokens to 8,192 cuts the 4-block model from 28.9M to
7.3M parameters, and trains 2.7 times faster on this CPU.

**Worth trying from nanochat for the TPU run:** the 1.2 scale on Q and K, and 0.4 on the
MLP's first layer. Both are small and cheap to test. KV caching would matter for a
browser explorer more than for training.

## Milestones

1. ~~**Move the project and review it.**~~ Done: this folder, with fixes and tests.
2. ~~**Our own tokenizer and token files.**~~ Done: `tokenizer/bpe.py`, `tinystories.py`.
3. ~~**A small local run**, with checkpoints dense early.~~ Done: `train.py --run local`.
4. ~~**First version of the figures.**~~ Done: `probes.py`, `figures.py`.
5. ~~**Read every figure critically.**~~ Done: see Results so far.
6. **The d20 run.** See below. *Done when:* it finishes on ClimbMix, validation bits per
   byte beat the local run's by a clear margin, and samples read as real text.
7. **The copying section.** *Done when:* the curve is re-measured on the new checkpoints
   with a larger, preselected set of names, sampled stories reuse them or demonstrably
   don't, and the head that does the copying is identified, if there is one.
8. **`POST.md`**, and the explorer if the timing allows.

## The d20 run on ClimbMix

The next run moves to ClimbMix, the web text nanochat trains on, with a 20-block model:
1,280 dims, 10 heads, a 32,768-token tokenizer of our own and 477M parameters. TinyStories'
526M tokens would give 1.1 tokens per parameter, which invites memorising, while ClimbMix
has thousands of shards. 120 shards give about 6.4B tokens.

`notebooks/01_d20_climbmix_colab.ipynb` runs it on Colab, and `notebooks/COLAB.md` has the
commands. A smoke run of the notebook passes locally, including a resume, and the full
model trains for a step on CPU. Neither has run on Colab yet.

**This changes what the post can measure.** The story section above, `probes.py`'s name
templates and the invented names in `figures.py` all assume stories: those names were
vetted against TinyStories, and web text has no steady cast of characters. The figures
should work on the new checkpoints once `Inspector` can read them, but the copying
question needs new probes, or a new question, before the post can use this run.

Olsson et al. saw induction heads at 2.5 to 5 billion tokens, and this run sees 5.7B.

## Files

- `POST.md`: the draft, predictions first.
- `tokenizer/bpe.py`: our tokenizer, trained on TinyStories.
- `tinystories.py`: download, train the tokenizer, write token files.
- `webtext.py`: the same for ClimbMix, in parallel across shards.
- `notebooks/`: the d20 Colab notebook and its guide.
- `data.py`: `TokenBinDataset` for token files; older parquet and Grain loaders for text
  shards.
- `decoder_only_model.py`: the model, loss and sampling.
- `experiment.py`: the shared harness, plus sampling during eval.
- `train.py`: `TINYSTORIES_RUNS`, the model and optimizer settings the notebook uses, and
  the older nanochat-style `create_experiment`.
- `metrics.py`: bits per byte. `analyze.py`: token budget and Chinchilla estimate.
- `probes.py`: measurements on a trained model. `figures.py`: every figure.
- `tokenizer/` (other files), `scripts/`: tokenizer tools, CORE eval and compilation
  analysis from the nanochat work, which still use `transformers`. `NANOCHAT_NOTES.md` holds that work's notes and TODOs.
- `Train.ipynb`, `generated/`: earlier Colab notebook and bundles. Stale: they predate
  the harness move. Only their imports were renamed (the notebook) or nothing at all
  (the bundles).

## Commands

```bash
uv sync --extra posts --extra tinystories --extra dev
```

```bash
uv run python -m toylib_projects.p02_tinystories.tinystories --vocab-size 8192
```

```bash
uv run python -m toylib_projects.p02_tinystories.webtext --num-train-shards 120
```

```bash
uv run python -m toylib_projects.p02_tinystories.train --run local
```

```bash
uv run python -m toylib_projects.p02_tinystories.figures --run local
```

```bash
uv run pytest toylib_projects/p02_tinystories
```

`--run smoke` checks the whole pipeline in about a minute, and `--resume` continues from
the newest checkpoint.
