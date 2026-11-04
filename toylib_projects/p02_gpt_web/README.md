# Post 2: Does a GPT read, or just remember?

A language model predicts the next token from two sources: what it learned about language
in training, and the text in front of it. Reading "Marie Curie moved to Paris", it can
guess "Paris" again because the word is common, or because it just read it. This post
trains a 20-block GPT on web text and tries to tell those apart, by following its
predictions token by token and looking inside for how it reads back. As in post 1, the aim
is intuition through looking at the network, not a measured comparison of architectures.

Web text suits the question. It's messy, documents repeat their own rare names and terms,
and this run sees 3.5B tokens, inside the range where
[Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
saw induction heads, the attention pattern behind [A][B]...[A] -> [B] copying, form.

## The story

Each section ends on a problem that the next one solves. No figure exists yet: all of them
wait for the d20 run, and `POST.md` holds predictions made before seeing them.

1. **The hook.** Hand the model a name it has never seen. Does it use the name again?
   (`names`)
2. **What a page costs.** Colour each token by how many bits it took to predict. A first
   mention is expensive; a repeat is cheap. So where does the model get what it needs to
   predict a repeat? (`surprise`)
3. **What counting achieves.** Word frequency, then bigram and trigram tables, on the same
   axis as the model. A table can't read back, so the gap between it and the model is what
   reading earned. (`baselines`)
4. **Learning to write.** One prompt continued at points in training, from the samples
   logged at each eval: noise, then common words, then grammar, then something like facts.
   (`samples`, `training`)
5. **How far back does it read?** Limit the context and measure the loss. It's a direct
   answer that needs no interpretation of weights. (`context`)
6. **Who reads whom.** Attention over one passage, then how much weight each head puts at
   each distance. Attention shows where a block reads from, but not what it does with it.
   (`attention`, `attention_distance`)
7. **What each block adds.** Send each block's state through the output layer (the logit
   lens) and watch a prediction form with depth. (`lens`)
8. **Does it copy?** The experiment, on the final model, with each name as a control for
   the others. (`names`, which also prints the random-string margin)
   - **Finish an invented name:** "...led by Dr. Spoundo... Dr. Sp-ound" -> "o"? This is
     the induction pattern, on token pairs absent from training.
   - **Bring an invented name back:** does its first token get likelier at a slot where a
     name belongs?
   - **Repeat a random string:** the classic test, and the hardest, since a web page rarely
     looks like this. The control is the same tokens shuffled.
9. **Words that fill the same slots.** Surnames, animals, colours and countries in the output
   layer and in the input embeddings. A supporting view of remembering, since it doesn't
   explain copying. (`embeddings`)
10. **Where it fails.** Facts the model gets wrong, and what attention was doing at the time.

**Later:**

- Find a copying head directly: rank heads by attention from the second mention to the
  token after the first, then zero that head and see the copy margin fall.
- An interactive page in the spirit of `p01_cnn_mnist/post/explorer`: edit a passage and
  watch surprise, attention and the top guess update. A 477M model won't run in a browser,
  so this needs a smaller one.
- If random repetition never appears, train a second run with repeated strings mixed in
  and show both.

## What to watch for in the results

Earlier work on a small model showed where these measurements mislead:

- **A margin isn't use.** A large margin can go from almost nothing to very little. Judge
  copying by probabilities and by whether samples write the name back, which is why `names`
  draws probabilities.
- **Averages hide names.** Margins vary by nats between names, so use all 24, fixed before
  any model is measured on them. They were vetted against 8 shards, so re-run
  `--find-names` on the full token file before trusting that a pair is unseen.
- **An unrelated string is a weak control.** A token gets cheaper after it has appeared,
  in any order, so a repeat has to beat the same tokens shuffled.
- **Any odd name can lift another.** Compare against a different invented name in the text,
  as `names` does, and not only against no name.
- **Bringing a name back is scored on its first token,** which can start other words.
- **The lens understates early blocks.** The output layer only ever trains on the last
  block's state, so a name appearing only late may partly be a limit of the method.
- **Context can't exceed the document.** Documents average about 600 tokens, so the flat end
  of `context` is partly that nothing useful lies further back.

## What changed from the older project

The review found these, each now covered by a test. Fixes 1 to 5 change results, so
treat any number from before these fixes as out of date.

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
and the harness always saves the last step, which the interval used to miss.

## Compared with nanoGPT

[nanoGPT](https://github.com/karpathy/nanoGPT) is the classic GPT-2; our model follows
its successor [nanochat](https://github.com/karpathy/nanochat), so the two differ in
architecture by design.

**Ported:**

| From nanoGPT | Here | Result |
| --- | --- | --- |
| Random windows from a memory-mapped file (`get_batch`) | `TokenBinDataset` | Fast, needs no text at training time, and batches aren't one contiguous run |
| Warmup then cosine decay to a tenth of the peak (`get_lr`) | `warmup_cosine_schedule` | Used by every run in `RUNS`; optional in `create_experiment` |
| Gradient clipping at 1.0 | `Run.max_grad_norm` | On by default for these runs |
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
  nothing useful with 32,768, which already is a multiple.
- **MFU reporting.** Useful on TPUs, so worth a look then. Steps per second is enough here.

**Not from nanoGPT, but bigger than all of the above:** our own tokenizer. nanoGPT has no
trainer; it loads GPT-2's encoding from `tiktoken`. nanochat trains one, with a compiled
`rustbpe` for training and `tiktoken` for encoding, which would mean two more
dependencies. `tokenizer/bpe.py` keeps nanochat's split pattern and adapts
[minbpe](https://github.com/karpathy/minbpe)'s plain Python, with `re` from the standard
library, pair counts updated as it merges and a heap to find the next merge. Trained on 8
shards, about 2B characters, it takes minutes. Every byte has an id, so there is no
unknown token.

**Worth trying from nanochat in a later run:** the 1.2 scale on Q and K, and 0.4 on the
MLP's first layer. Both are small and cheap to test. KV caching would matter for a
browser explorer more than for training.

## Milestones

1. ~~**Move the project and review it.**~~ Done: this folder, with fixes and tests.
2. ~~**Our own tokenizer and token files.**~~ Done: `tokenizer/bpe.py`, `tokenize_webtext.py`.
3. ~~**Figures and probes, tried on a small model.**~~ Done: `probes.py`, `figures.py`.
4. **The d20 run.** See below. *Done when:* it finishes on ClimbMix, and samples read as
   real text.
5. **Read every figure critically.** *Done when:* each has a takeaway and a measured number
   here, with a title that states it, and `POST.md`'s predictions are scored.
6. **The copying section.** *Done when:* sampled text reuses invented names or demonstrably
   doesn't, and the head that does the copying is identified, if there is one.
7. **`POST.md`** written around the results.

## The d20 run

`train.RUNS["d20"]`: 20 blocks, 1,280 dims, 10 heads, 32,768 tokens, a context of 2,048 and
477M parameters. It trains on batches of 524,288 tokens, as 32 microbatches of 8, for 3.5B
tokens from 120 shards of ClimbMix, the web text nanochat trains on. That's 8 tokens per
parameter by nanochat's count, which leaves out the input embedding. `notebooks/01_d20_climbmix_colab.ipynb` runs it on Colab, and
`notebooks/COLAB.md` has the commands.

The tokenizer is trained on 8 shards, as nanochat does, and stored at
`gs://tinystories-checkpoints/toylib-p02-climbmix/data_shards120/tokenizer_32768.json`.
On held-out text it averages 4.69 bytes per token, and every word in `WORD_GROUPS` is a
single token. The token files will join it on the first run.

On a Colab A100 it runs at 31,500 tokens per second, so the full run needs about 31 hours and
209 compute units (`BENCHMARK.md`), and a 3% trial passed every health check. A full d12
run, `RUNS["d12"]`, reached 0.949 bits per byte against nanochat's 0.87 to 0.90 at the same
size, about 6 to 10% behind. Porting nanochat's current architecture and optimizer,
`RUNS["d12_nanochat"]`, reached 0.877, 7.6% below that baseline and inside nanochat's range, and
d20 trained with that recipe, `RUNS["d20_nanochat"]`, and reached 0.783 in 21.3 hours for
about 145 units. `EXPERIMENTS.md` has every run and the full list of what was ported. The notebook
itself has only run locally so far. Figures read checkpoints from the bucket
with `--checkpoint-dir`, and a model this size needs an accelerator to draw them in
reasonable time.

## Files

- `POST.md`: the draft, predictions first.
- `tokenizer/bpe.py`: our tokenizer.
- `tokenize_webtext.py`: download ClimbMix, train the tokenizer, write token files.
- `notebooks/`: the d20 Colab notebook and its guide.
- `BENCHMARK.md`: hardware, configuration, speed and cost.
- `EXPERIMENTS.md`: the training runs and their results, and the nanochat recipe port.
- `muon.py`: nanochat's current Muon, as an optax transform.
- `data.py`: `TokenBinDataset` for token files; older parquet and Grain loaders for text
  shards.
- `decoder_only_model.py`: the model, loss and sampling.
- `experiment.py`: the shared harness, plus sampling during eval.
- `train.py`: `RUNS`, `create_run_experiment`, and the older nanochat-style
  `create_experiment`.
- `metrics.py`: bits per byte. `analyze.py`: token budget and Chinchilla estimate.
- `probes.py`: measurements on a trained model. `figures.py`: every figure.
- `tokenizer/` (other files), `scripts/`: tokenizer tools, CORE eval and compilation
  analysis from the nanochat work, which still use `transformers`. `NANOCHAT_NOTES.md` holds that work's notes and TODOs.
- `Train.ipynb`, `generated/`: earlier Colab notebook and bundles. Stale: they predate
  the harness move. Only their imports were renamed (the notebook) or nothing at all
  (the bundles).

## Commands

```bash
uv sync --extra posts --extra webtext --extra dev
```

```bash
uv run python -m toylib_projects.p02_gpt_web.tokenize_webtext --num-train-shards 8
```

```bash
uv run python -m toylib_projects.p02_gpt_web.train --run local
```

```bash
uv run python -m toylib_projects.p02_gpt_web.figures --run local
```

```bash
uv run pytest toylib_projects/p02_gpt_web
```

These train `RUNS["local"]`, a 4-block model, on 8 shards. For `--run smoke`, prepare with
`--vocab-size 1000 --num-train-shards 2`. `--resume` continues from the newest checkpoint.
