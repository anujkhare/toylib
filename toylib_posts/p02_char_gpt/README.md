# Post 2: When does a small GPT learn to copy?

An induction head completes the pattern [A][B] ... [A] -> [B], and needs at
least two layers ([Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)).
In that paper's small models induction heads formed abruptly, at about 2.5 to
5 billion training tokens. A TinyShakespeare run sees far fewer, so whether
copying appears at this scale is the open part of the experiment.

**Experiment.** Feed a random 100-character string twice in a row and compare
the loss on the second copy with the first, at every checkpoint. If copying
never appears, mix repeated random sequences into training and show both runs.

**Model.** Reuse `toylib_projects/tinystories/decoder_only_model.py` with a
character vocabulary. Fix the RoPE tables first (review fix 1), and consider
promoting the model into `toylib.nn` once a second post needs it.

**Data.** [TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt):
1,115,394 characters, 65 unique. Chunk it into fixed-length windows and serve
them with `ArrayDataset({"inputs": ..., "targets": ..., "mask": ...})`, the
batch format `decoder_only_model.train_step` already expects.

**Supporting views:**

- Per-character loss over a passage: `viz.token_heatmap(list(text), loss)`.
- One prompt sampled at steps 0, 100, 1k and 5k (`inference.load_model` per checkpoint).
- Attention per head on one line of text: `viz.heatmap(weights, row_labels=..., col_labels=...)`.
- Character embeddings projected to 2D.
- Trained vs. frozen RoPE tables: rerun with the bug restored and compare loss.

## Milestones

1. **Character tokenizer and a 90/10 split** in `toylib/data/shakespeare.py`.
   *Done when:* encode then decode returns the original text, and the vocabulary has 65 entries.
2. **A small config** (two to four layers) training through the harness.
   *Done when:* samples move from noise to word-like text, and validation loss tracks training loss.
3. **Checkpoints and fixed-prompt sampling** every N steps.
   *Done when:* the samples-over-training figure renders.
4. **Attention weights through `return_intermediates`, and the repeated-sequence test** across checkpoints.
   *Done when:* a curve of second-copy minus first-copy loss over training steps.
5. **One or two supporting views, then `POST.md`.**
