# World Model — Issues & Investigations Report

A running log of concrete problems hit while building the world model, the
reasoning used to diagnose them, and the decisions that came out of each
investigation. Each issue is self-contained; new issues are appended over time
and linked from the index below.

> This is a **decision log**, not a design doc. For architecture see
> `docs/designs/`. The intent here is that someone (including future us) can
> read *why* a thing is the way it is, and what was ruled out along the way.

**Per-issue template.** Each issue follows the same shape so they read
consistently:

1. **Symptom** — what was observed.
2. **Hypotheses** — the candidate root causes, each stated so it can be
   confirmed or ruled out.
3. **Experiments** — one per investigation, each with **setup**, **results**,
   and **observations**. Add as much free-text detail as needed.
4. **Conclusion** — which hypothesis survived, and the follow-up it spawned.

Headings can be added or adapted per issue; the four beats above are the
backbone.

---

## Index

| # | Issue | Status | Summary |
|---|-------|--------|---------|
| 1 | [Ball not reconstructed by the VAE](#issue-1--ball-not-reconstructed-by-the-vae) | In progress | Decoded frames drop/misplace the ball. Latent probe shows `ball_x` is in the latent (→ decoder's fault, H2); `ball_y` is weak across all encoders incl. raw pixels, but labels are verified correct, so a probe-ceiling check is pending before deciding H1 vs H2 for y. |

<!-- Add new rows above this line. Keep the anchor links in sync with the
     section headers further down. -->

---

## Issue 1 — Ball not reconstructed by the VAE

**Status:** In progress
**Components:** `vision_encoder/` (VAE), `probe/` (diagnostic)

### 1.1 Symptom

The VAE reconstructions look good globally — background, bricks, and paddle come
back cleanly — but the **ball is frequently missing, blurred, or in the wrong
place** in decoded frames. This matters: the ball is the single most important
object for every downstream physics objective, and a codec that silently drops
it caps the quality of everything trained in latent space.

### 1.2 Hypotheses

A bad reconstruction of the ball has two very different root causes, and they
demand opposite fixes. They are mutually distinguishable: if we can read the
ball position *out of the latent*, the information is present; if we can't, it
isn't.

**H1 — The encoder never stored the ball.** The 8× downsampled latent
(128→16×16×4 by default) is small, and the reconstruction loss is dominated by
large, low-frequency regions (background/bricks). A tiny, fast-moving object is
cheap to ignore. If the information isn't in the latent, no decoder change can
recover it — the fix would have to change the encoder, latent size, or loss
weighting.

**H2 — The decoder fails to render a ball the latent *does* encode.** Here the
information is present in the latent but the decoder hallucinates a plausible
frame without it. The fix lives in the decoder (capacity, loss weighting), not
the encoder.

We cannot tell H1 and H2 apart from reconstructions alone. We need to read the
latent directly — which motivates Experiment 1.

### 1.3 Experiments

#### Experiment 1 — Latent probe: is the ball recoverable from the latent?

**Goal.** Decide between H1 and H2 by measuring whether the ball/paddle
positions can be read back out of the frozen encoder's latent.

**Setup.** Train a small **MLP probe on top of the frozen encoder latent** to
regress the ground-truth ball and paddle positions. The logic:

- If the probe recovers ball position **well**, the latent *does* encode it →
  H2 (decoder's fault).
- If the probe **fails**, the encoder discarded the ball → H1 (encoder/latent's
  fault).

Implemented in `probe/model.py` (model) and `probe/train.py` (training),
reusing the shared `Experiment` harness. The key methodology decisions, with
the alternatives considered:

- **Targets = RAM state values (`ball_x`, `ball_y`, `paddle_x`), normalized by
  255.** The compiled dataset stores per-frame RAM state under `source/`; we use
  it directly rather than re-extracting pixel coordinates, which would add a
  noisy preprocessing step that could itself be the thing under test.
- **Encoder frozen at the *optimizer* level, not via `stop_gradient`.** A
  multi-transform routes the `encoder` sub-tree to `optax.set_to_zero()` and the
  MLP head to Adam (`"frozen" if path[0].name == "encoder" else "trainable"`).
  This keeps "what is trainable" a single inspectable decision and leaves
  `MLPProbe` a plain feed-forward module. A test asserts the encoder is
  byte-for-byte unchanged while the head moves.
- **`flatten` pooling by default (a `Pooling` enum).** `flatten` preserves the
  full spatial grid (where the ball is); `mean` pools it away and only helps if
  position lives in channel statistics — it usually doesn't.
- **Reuse the shared harness; common modules extracted to `wm/`.** The probe
  trains through the same loop as the VAE, so the comparison is apples-to-apples.
- **Optional checkpoint + `gs://` support.** `--vae-checkpoint-dir` can be
  omitted (random encoder) or point at a GCS URI (orbax via tensorstore/epath,
  `gcsfs` added to deps).
- **Comparison ladder, not an absolute threshold:**
  1. **Pass-through** (`EncoderType.PASSTHROUGH`, `IdentityEncoder` feeds raw
     pixels to the head) → upper bound any latent could hit.
  2. **Random encoder** (`EncoderType.VAE`, no checkpoint) → the "no training"
     floor.
  3. **Trained VAE** (checkpoint) → the real number.
- **R² as the headline metric.** Raw MSE/MAE have no fixed "good" threshold
  because they depend on how much each target varies. The forward pass emits, per
  target, `r2 = 1 − MSE/Var(target)` (variance precomputed from the train labels,
  closed over as a jit constant), alongside `mse_*` and RAM-unit `mae_raw_*`.

**How to read it.** Watch the **eval/val** task, not train (the MLP head can
memorize a small set):

| R² (val) | Interpretation |
|---|---|
| ≈ 0 or negative | Latent carries no recoverable info — no better than predicting the mean. |
| ~0.5–0.85 | Partially present / coarse. |
| ≳ 0.9 | Strongly encoded; position is essentially recoverable from the latent. |

The trained VAE must clearly beat the random floor; how close it gets to the
pass-through bound tells us how much the ball survived encoding.

**Results.** Probe R² on **eval/val**, comparison ladder run on Colab (the
trained VAE) plus the random / pass-through baselines:

| Encoder | `ball_x` R² | `ball_y` R² |
|---|---|---|
| Pass-through (pixels) | ~0.89 | ~0.09 |
| Random VAE | ~0.84 | ~0.11 |
| Trained VAE | ~0.91 | ~0.24 |

**Observations.**

- **`ball_x` is strongly recoverable from *every* encoder** (0.84–0.91), and the
  trained VAE beats the random encoder by only ~0.07. Horizontal ball position
  survives almost any encoding — so the x information *is* in the latent. For the
  x-coordinate this is evidence **against H1**.
- **`ball_y` is weak everywhere** (best 0.24), and — the surprising part — even
  **pass-through (raw pixels) only reaches 0.09**. Pass-through is meant to be the
  upper bound on recoverable info, so a low number there means the bottleneck is
  *not* the VAE latent: something limits *every* rung, including raw pixels.
- **Training the VAE more than doubles `ball_y`** (0.09–0.11 → 0.24). So the
  trained encoder makes vertical position *more* accessible than raw pixels do —
  the encoder is the best rung for y, not the worst. This also argues against a
  simple "encoder dropped the ball vertically" story.
- The pass-through *under*-performing the trained VAE on y breaks the clean
  "pass-through = ceiling" reading: when the small MLP head is the limiter, the
  ladder stops being a strict bound. The real puzzle is the **x-easy / y-hard
  asymmetry that holds across all encoders**.
- **Quantization floor.** A 16×16 latent over a 128px frame is ~8px/cell, so a
  residual MAE of a few RAM units is the floor even with perfect encoding — don't
  read that as "missing."

#### Experiment 2 — Label sanity check: do the RAM labels track the visible ball?

**Goal.** The x-easy/y-hard asymmetry survives even on raw pixels, which points
at the *target* rather than the encoder. First suspect: the `ball_y` label
itself (a noisy or miscalibrated target would depress R² for every encoder
equally). Rule it in or out before chasing the probe/representation.

**Setup.** The labels are raw Atari 2600 RAM bytes (`ball_x`=addr 99,
`ball_y`=addr 101, `paddle_x`=addr 72) — *not* pixel coordinates. Two checks
(`probe/`-adjacent scripts under `wm/scripts/`):

1. `calib_ball_labels.py` — detect the ball's actual pixel centroid (the only
   red blob in the central play band, walls/bricks/paddle masked out) on 2k
   random frames and least-squares fit `pixel = a·ram + b` for each axis. The
   fit R² measures how faithfully the RAM label tracks the visible ball.
2. `viz_ball_labels.py` — overlay a box at the calibrated ball position (and a
   paddle tick) on a montage of random frames for manual inspection.

**Results.**

| Axis | calibrated map (128px) | fit R² | residual std |
|---|---|---|---|
| `ball_x` | `0.800·ram − 38.9` | 1.0000 | 0.14 px |
| `ball_y` | `0.795·ram − 16.7` | 0.9997 | 0.31 px |

The overlay montage confirms it visually: the box sits on the ball in every
sampled frame, across the full vertical range — including small `ball_y` (ball
up in the brick band) and large `ball_y` (ball near the paddle).

**Observations.** The `ball_y` label is **correct** — it encodes the visible
ball's vertical position to sub-pixel accuracy. So the probe's weak `ball_y` is
**not** a labeling artifact. The label hypothesis is ruled out; the asymmetry
must come from the *probe/representation* (e.g. a tiny ~2px ball is hard for a
small MLP to vertically localize from a flattened grid; or frames where the
ball is occluded among same-colored bricks / off-screen between lives add
irreducible y-error). That is the next thing to chase.

### 1.4 Conclusion

*Partial — `ball_x` resolved, `ball_y` still open.*

**Per-experiment conclusions.**

| Exp | Question | Conclusion |
|---|---|---|
| 1 — Latent probe | Is ball position recoverable from the latent? | **`ball_x`: yes** (R²≈0.91, only ~0.07 above the random/pixel floor) → the info is in the latent. **`ball_y`: weak everywhere** (best 0.24; pass-through pixels only 0.09), so the limit isn't the VAE latent. |
| 2 — Label sanity check | Is the `ball_y` label itself wrong/noisy? | **No.** The RAM label tracks the visible ball to sub-pixel accuracy (R²=0.9997, <0.4 px). The label hypothesis is ruled out. |

**What this means for H1 vs H2.**

- **`ball_x` → H2 (decoder).** The x-position is present in the latent, so a bad
  x-reconstruction is the decoder hallucinating, not the encoder dropping info.
- **`ball_y` → undecided, but two causes eliminated.** Ruled out: (a) wrong label
  (Exp 2), and (b) the trained encoder dropping vertical info relative to pixels
  (it actually *doubles* `ball_y` over the pixel/random baselines). What remains
  is that **no rung — not even raw pixels — recovers `ball_y` well**, which looks
  like a *probe/measurement ceiling*, not a statement about latent content. We
  can't attribute the low number to the encoder until that ceiling is lifted.

**Planned next steps.**

1. **Lift the probe ceiling for y.** Train the probe to convergence (the full
   ~10k-step schedule, watching eval/val R² plateau — *not* the 150-step CPU
   smoke run) and increase MLP head capacity. If `ball_y` stays low even with a
   well-fit, higher-capacity head, the ceiling is real.
2. **Segment y-error by regime.** Split metrics into ball-in-play vs. ball-in-
   brick-band (camouflaged among same-colored bricks) vs. off-screen/between-
   lives, to isolate where the irreducible y-error comes from.
3. **Decide H1 vs H2 for y** once 1–2 are in, then spin off the fix as its own
   issue: decoder change (H2) vs. encoder/latent/loss change (H1).
4. **`ball_x` follow-up (H2).** Track the decoder-side fix for the x-coordinate
   reconstruction as a separate issue.
