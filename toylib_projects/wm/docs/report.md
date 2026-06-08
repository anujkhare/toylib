# World Model — Issues & Investigations Report

A running log of concrete problems hit while building the world model, the
reasoning used to diagnose them, and the decisions that came out of each
investigation. Each issue is self-contained; new issues are appended over time
and linked from the index below.

> This is a **decision log**, not a design doc. For architecture see
> `docs/designs/`. The intent here is that someone (including future us) can
> read *why* a thing is the way it is, and what was ruled out along the way.

---

## Index

| # | Issue | Status | Summary |
|---|-------|--------|---------|
| 1 | [Ball not reconstructed by the VAE](#issue-1--ball-not-reconstructed-by-the-vae) | In progress | Decoded frames drop/misplace the ball. Built a latent probe to localize the failure to the encoder vs. the decoder. |

<!-- Add new rows above this line. Keep the anchor links in sync with the
     section headers further down. -->

---

## Issue 1 — Ball not reconstructed by the VAE

**Status:** In progress
**Components:** `vision_encoder/` (VAE), `probe/` (diagnostic)

### 1.1 The symptom

The VAE reconstructions look good globally — background, bricks, and paddle come
back cleanly — but the **ball is frequently missing, blurred, or in the wrong
place** in decoded frames. This matters: the ball is the single most important
object for every downstream physics objective, and a codec that silently drops
it caps the quality of everything trained in latent space.

### 1.2 The core question

A bad reconstruction of the ball has two very different root causes, and they
demand opposite fixes:

1. **The encoder never stored the ball.** The 8× downsampled latent
   (128→16×16×4 by default) is small, and the reconstruction loss is dominated
   by large, low-frequency regions (background/bricks). A tiny, fast-moving
   object is cheap to ignore. If the information isn't in the latent, no decoder
   change can recover it — we'd need to change the encoder/latent/loss.
2. **The decoder fails to render a ball the latent *does* encode.** Here the
   information is present but the decoder hallucinates a plausible frame without
   it. The fix lives in the decoder (capacity, loss weighting), not the encoder.

We cannot tell these apart by looking at reconstructions alone. We need to read
the latent directly.

### 1.3 Approach: a latent probe

Train a small **MLP probe on top of the frozen encoder latent** to regress the
ground-truth ball and paddle positions. The logic:

- If the probe recovers ball position **well**, the latent *does* encode it →
  the failure is in the **decoder**.
- If the probe **fails**, the encoder discarded the ball → the failure is at the
  **encoder/latent** level.

This turns a vague "the ball looks wrong" into a measurable yes/no on where the
information lives. The probe is implemented in `probe/model.py` (model) and
`probe/train.py` (training), reusing the shared `Experiment` harness.

### 1.4 Decisions

Each decision below records the choice, the alternatives, and the reasoning.

**D1 — Probe the RAM state values directly (`ball_x`, `ball_y`, `paddle_x`).**
The compiled dataset already stores per-frame RAM state under `source/`. We
regress those values (normalized by `TARGET_SCALE = 255`) rather than
re-extracting pixel coordinates from frames or deriving paddle endpoints.
Rationale: the RAM values are exact ground truth, free, and unambiguous; pixel
re-extraction would add a noisy, bug-prone preprocessing step that could itself
be the thing under test.

**D2 — Freeze the encoder at the *optimizer* level, not with `stop_gradient`.**
The optimizer is a multi-transform that routes the whole `encoder` sub-tree to
`optax.set_to_zero()` and the MLP head to Adam
(`optimizer_for_param = lambda path: "frozen" if path[0].name == "encoder" else
"trainable"`). The encoder still receives gradients; the optimizer drops its
updates. Rationale: "what is trainable" becomes a single, inspectable decision
in the training script instead of a `stop_gradient` buried in the model — and it
keeps `MLPProbe` a plain feed-forward module. A regression test asserts the
encoder is byte-for-byte unchanged after steps while the head moves.

**D3 — `flatten` pooling by default (made an enum).**
The probe reduces the `(B, h, w, C)` latent to a feature vector before the MLP.
`flatten` keeps the full spatial grid (preserves *where* the ball is); `mean`
pools over the grid and only helps if position is encoded in channel statistics
— usually it isn't. `flatten` is the default. The option is a `Pooling` enum
(not a raw string) so invalid modes fail loudly and the bundler/CLI carry a
typed value.

**D4 — Reuse the shared `Experiment` harness; extract common modules to `wm/`.**
Model-agnostic infrastructure (`experiment.py`, `logger.py`, generic
`metrics.py`, `dataloader.py`) was lifted from `vision_encoder/` up to `wm/` so
the probe trains through the exact same loop as the VAE. VAE-specific metrics
stay in `vision_encoder/`. Rationale: the probe is a diagnostic, not a new
training stack — sharing the harness means the comparison is apples-to-apples.

**D5 — Make the VAE checkpoint optional, and support `gs://` checkpoints.**
`--vae-checkpoint-dir` can be omitted (encoder is randomly initialized) for
smoke-testing the pipeline and as a weak baseline. The path may be a local path
*or* a `gs://bucket/...` URI; orbax routes the latter through
tensorstore/epath, with `gcsfs` added to the `toylib-wm` deps. Rationale: we
need to iterate on the probe without always having a trained VAE on disk, and
real checkpoints live in GCS.

**D6 — A pass-through encoder baseline (`EncoderType.PASSTHROUGH`).**
`IdentityEncoder` feeds the raw image straight to the MLP head (no VAE). The
ball is trivially recoverable from pixels, so this is the **upper bound** on how
well *any* latent could be probed. Rationale: a probe number is meaningless in
isolation; the pass-through run tells us the ceiling.

**D7 — Report **R²** as the interpretable metric.**
Raw MSE/MAE depend on how much each target varies, so they have no fixed "good"
threshold. The forward pass now emits, per target, `r2 = 1 − MSE/Var(target)`
(variance precomputed once from the train labels, closed over as a jit
constant), alongside `mse_*` and RAM-unit `mae_raw_*`. Rationale: R² is
normalized by the target's own variance, so it reads directly as "fraction of
the ball's motion the latent explains."

### 1.5 How to read the results

R² is the headline number (watch the **eval/val** task, not train — the MLP head
can memorize a small set):

| R² (val) | Interpretation |
|---|---|
| ≈ 0 or negative | Latent carries no recoverable info — no better than predicting the mean. |
| ~0.5–0.85 | Partially present / coarse. |
| ≳ 0.9 | Strongly encoded; position is essentially recoverable from the latent. |

Use the **comparison ladder** rather than an absolute cutoff:

1. **Pass-through** (`--encoder-type passthrough`) → upper bound the latent could hit.
2. **Random encoder** (`--encoder-type vae`, no checkpoint) → the "no training" floor.
3. **Trained VAE** (`--encoder-type vae --vae-checkpoint-dir ...`) → the real number.

The trained VAE must clearly beat the random floor; how close it gets to the
pass-through bound tells us how much the ball survived encoding.

Two Breakout-specific caveats:
- **Quantization floor.** A 16×16 latent over a 128px frame is ~8px/cell, so a
  residual MAE of a few RAM units is the floor even with perfect encoding — don't
  read that as "missing."
- **Compare ball vs. paddle in the same run.** The paddle is large/slow and will
  probe well almost regardless. High `paddle_x` R² with near-zero `ball_*` R² is
  the classic "decoder hallucinates a ball the latent never stored" signature —
  i.e. root cause #1 from §1.2.

### 1.6 Status / next steps

- [x] Probe model, training script, freeze, optional checkpoint, GCS loading.
- [x] R² metric + pass-through baseline.
- [ ] Run the ladder on a trained VAE checkpoint and record the `ball_*` /
      `paddle_*` R² numbers here.
- [ ] Conclude encoder-vs-decoder and link to the follow-up fix (new issue).
