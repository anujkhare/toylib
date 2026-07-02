from __future__ import annotations

# ============================================================
# External Imports
# ============================================================

from PIL import Image
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Iterator, NamedTuple
from typing import Literal
import abc
import argparse
import dataclasses
import einops
import functools
import h5py
import hdf5plugin
import jax
import jax.numpy as jnp
import jaxtyping as jt
import json
import math
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import time
import typing

# ============================================================
# toylib_projects.wm.datagen.preprocess_frames - /Users/anuj/Desktop/code/toylib/toylib_projects/wm/datagen/preprocess_frames.py
# ============================================================

"""Frame preprocessing for the vision encoder pipeline.

This module is the **single source of truth** for transforming a raw Atari
Breakout frame `(210, 160, 3) uint8` into the shape the vision encoder (VAE)
sees. It is used in two places:

  1. **Offline materialization** — `generate_vision_enc_data.py` runs every
     sampled frame through `preprocess_frames` once and writes the result to
     the materialized VAE dataset.
  2. **Online inference** — the training and evaluation pipelines call the
     same function whenever they need to encode raw frames into VAE latents
     (e.g. evaluating reconstruction on held-out raw episodes).

Keeping the preprocessing logic in one place guarantees that the same pixels
the VAE was trained on are the same pixels seen at eval / rollout time.


Decisions and rationale
-----------------------

**Why a separate preprocessing module at all?**
Without one, the resize/crop logic gets duplicated across the materialization
script and every downstream consumer (training loop, eval, viz of decoded
samples). When the parameters drift, latents stop being comparable across
runs. Putting it in one module with a single `PreprocessConfig` dataclass
forces all callers through the same code path.

**Why crop out the top of the frame (the scoreboard)?**
The Atari Breakout scoreboard occupies the top ~32 rows of the 210-row frame
and contains the current score and number of lives rendered as digit sprites.
Those digits are:
  - **Highly informative shortcut signals** for any model that learns them —
    a VAE will happily devote latent capacity to memorizing digit shapes.
  - **Not part of the game physics** we want the world model to capture.
The same rationale applies to the bottom ~15 rows (border/walls under the
paddle). Defaults remove both.

Cropping is configurable — set `crop_top=0` and `crop_bottom=210` to keep the
full frame, e.g. if you want the digits as a downstream conditioning signal.

**Why a 160×160 square crop by default?**
Atari Breakout's active play area is naturally square once the scoreboard is
removed: rows 32–192 × cols 0–160 = 160×160. Keeping a square crop means the
VAE's downsampled latent grid is square too, which simplifies every later
DiT (no asymmetric positional encodings, no per-axis padding). This matches
the original `docs/designs/dataset.md` "storage crop" recommendation.

**Why 128×128 as the default target resolution?**
A balance:
  - At 128×128, the paddle is ~13 px wide and the ball is ~2 px — both still
    clearly resolved, so the VAE can reconstruct them sharply.
  - 128 is a clean power-of-two-compatible size: a stride-8 VAE produces a
    16×16 latent grid (256 tokens for DiT), and a stride-4 VAE produces 32×32
    (1024 tokens). Both fit comfortably into a small DiT.
  - At 64×64 the ball is only 1 px and reconstruction quality suffers; at
    256×256 the dataset gets 4× larger with marginal extra detail given the
    Atari pixel art aesthetic.

**Why LANCZOS as the default resize filter?**
For downscaling, LANCZOS preserves sharp edges (paddle, brick boundaries)
better than BILINEAR while avoiding the aliasing of NEAREST. The frames are
pixel-art-style so we want the resized image to still look crisp, not blurry.
Alternative filters are exposed for ablation.

**Why uint8 in / uint8 out?**
Storage size. The materialized VAE dataset is ~5 GB at 128×128 uint8; using
float32 would inflate it 4×. The VAE's input pipeline converts to float32 and
normalizes (typically to `[-1, 1]`) just before the encoder.

**What about color manipulation (hue rotation, palette swap, etc.)?**
Deliberately not done here. The preprocessing is *deterministic* and
*invertible-up-to-resize* by design — what you encode is what you reconstruct
against. Augmentations belong in the training data loader, not in the
materialization step.


Conventions
-----------

- All crops are expressed as Python half-open intervals: `frame[top:bottom,
  left:right]`. So `crop_top=32, crop_bottom=192` keeps rows 32..191
  (inclusive), i.e. 160 rows total.
- Frame layout is `(H, W, 3)` for a single frame and `(N, H, W, 3)` for a
  batch — same convention as the rest of the codebase.
"""
NATIVE_H = 210
NATIVE_W = 160
_RESIZE_FILTERS: dict[str, int] = {
    "lanczos": Image.Resampling.LANCZOS,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
}
ResizeFilter = Literal["lanczos", "bilinear", "bicubic", "nearest", "box"]


@dataclass(frozen=True)
class PreprocessConfig:
    """All knobs for raw-frame → VAE-input preprocessing.

    The defaults are tuned for the canonical Stage 1 dataset (60Hz Atari
    Breakout at native 210×160) and the Track 2 KL-VAE (128×128 input).

    Attributes
    ----------
    crop_top, crop_bottom :
        Half-open row range kept from the raw frame. Default `[32, 192)`
        removes the scoreboard strip on top and the dead area below the
        paddle row, yielding a 160-row block.
    crop_left, crop_right :
        Half-open column range. Default `[0, 160)` keeps the full width,
        which is exactly the play area for Breakout.
    target_h, target_w :
        Output resolution after resize. Default 128×128.
    resize_filter :
        Pillow resampling filter. Default "lanczos" (sharp, good downscaling).

    The crop is applied first, then the resize.
    """

    crop_top: int = 32
    crop_bottom: int = 192
    crop_left: int = 0
    crop_right: int = 160
    target_h: int = 128
    target_w: int = 128
    resize_filter: ResizeFilter = "lanczos"

    @property
    def cropped_shape(self) -> tuple[int, int]:
        """Shape after crop, before resize. (h, w)."""
        return (self.crop_bottom - self.crop_top, self.crop_right - self.crop_left)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """Final output shape `(H, W, 3)`."""
        return (self.target_h, self.target_w, 3)

    def validate(self) -> None:
        """Cheap sanity checks. Called on construction by `preprocess_frame[s]`."""
        if not 0 <= self.crop_top < self.crop_bottom <= NATIVE_H:
            raise ValueError(
                f"crop_top/crop_bottom must satisfy 0 <= top < bottom <= {NATIVE_H}, got [{self.crop_top}, {self.crop_bottom})"
            )
        if not 0 <= self.crop_left < self.crop_right <= NATIVE_W:
            raise ValueError(
                f"crop_left/crop_right must satisfy 0 <= left < right <= {NATIVE_W}, got [{self.crop_left}, {self.crop_right})"
            )
        if self.target_h <= 0 or self.target_w <= 0:
            raise ValueError(
                f"target_h/target_w must be positive, got {self.target_h}×{self.target_w}"
            )
        if self.resize_filter not in _RESIZE_FILTERS:
            raise ValueError(
                f"resize_filter must be one of {list(_RESIZE_FILTERS.keys())}, got {self.resize_filter!r}"
            )


def preprocess_frame(
    frame: np.ndarray, config: PreprocessConfig | None = None
) -> np.ndarray:
    """Crop + resize one frame.

    Parameters
    ----------
    frame :
        `(H, W, 3)` uint8 numpy array. Must match `NATIVE_H × NATIVE_W` by
        default; pass a custom `PreprocessConfig` if working with a different
        source resolution (e.g. for tests).
    config :
        Preprocess parameters. Defaults to `PreprocessConfig()` if omitted.

    Returns
    -------
    np.ndarray :
        `(target_h, target_w, 3)` uint8.
    """
    if config is None:
        config = PreprocessConfig()
    config.validate()
    if frame.dtype != np.uint8:
        raise ValueError(f"frame dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"frame shape must be (H, W, 3), got {frame.shape}")
    cropped = frame[
        config.crop_top : config.crop_bottom, config.crop_left : config.crop_right, :
    ]
    img = Image.fromarray(cropped)
    if (config.target_h, config.target_w) != cropped.shape[:2]:
        img = img.resize(
            (config.target_w, config.target_h), _RESIZE_FILTERS[config.resize_filter]
        )
    return np.asarray(img, dtype=np.uint8)


def ram_to_pixel(
    ram_x: float, ram_y: float, config: PreprocessConfig | None = None
) -> tuple[float, float] | None:
    """Map a RAM/native-frame coordinate to the preprocessed-frame pixel grid.

    The stored ``ball_x`` / ``ball_y`` / ``paddle_x`` state values live in the
    native ``210×160`` (row/col) coordinate system that Stage 1 records. The
    preprocessed frame the VAE sees is the result of ``crop`` then ``resize``
    (see :func:`preprocess_frame`). This function applies the *same* crop +
    scale so a state coordinate can be located in the preprocessed frame — the
    single source of truth used by the region-based reconstruction metrics
    (``ball_region_psnr`` etc.) so they never re-derive the transform.

    Convention: ``ram_x`` is a **column** in ``[0, 160)`` and ``ram_y`` is a
    **row** in ``[0, 210)`` (matching ``dataset.md``: ``ball_y > 105`` is the
    lower half of the 210px-tall frame). The returned ``(px, py)`` is
    ``(column, row)`` in the ``target_w × target_h`` preprocessed frame.

    Parameters
    ----------
    ram_x, ram_y :
        Native-frame column / row (e.g. ``ball_x`` / ``ball_y``).
    config :
        Preprocess parameters. Defaults to ``PreprocessConfig()`` if omitted.
        **Pass the config the dataset was compiled with** — read it from the
        ``.h5`` file's ``config_json`` attr — so the mapping matches the pixels
        the VAE actually saw.

    Returns
    -------
    tuple[float, float] | None :
        ``(px, py)`` column/row in the preprocessed frame, or ``None`` if the
        point falls outside the crop window (e.g. a between-lives frame with no
        ball, or a coordinate in the cropped-away scoreboard). Callers skip
        ``None`` frames when aggregating region metrics.
    """
    if config is None:
        config = PreprocessConfig()
    crop_h = config.crop_bottom - config.crop_top
    crop_w = config.crop_right - config.crop_left
    cx = ram_x - config.crop_left
    cy = ram_y - config.crop_top
    if not (0.0 <= cx < crop_w and 0.0 <= cy < crop_h):
        return None
    px = cx * config.target_w / crop_w
    py = cy * config.target_h / crop_h
    return (float(px), float(py))


def preprocess_frames(
    frames: np.ndarray, config: PreprocessConfig | None = None
) -> np.ndarray:
    """Crop + resize a batch of frames.

    Parameters
    ----------
    frames :
        `(N, H, W, 3)` uint8 numpy array.
    config :
        Preprocess parameters. Defaults to `PreprocessConfig()` if omitted.

    Returns
    -------
    np.ndarray :
        `(N, target_h, target_w, 3)` uint8.

    Notes
    -----
    This is a Python-level loop over Pillow operations. For our scale
    (~100k frames per materialization run) it takes a few minutes and is not
    worth vectorising further. If a downstream pipeline needs faster batch
    preprocessing during training, copy the logic into a jit-compiled JAX
    function — the resize semantics here (LANCZOS) match `jax.image.resize`
    with `method="lanczos3"` when target sizes match.
    """
    if config is None:
        config = PreprocessConfig()
    config.validate()
    if frames.dtype != np.uint8:
        raise ValueError(f"frames dtype must be uint8, got {frames.dtype}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames shape must be (N, H, W, 3), got {frames.shape}")
    n = frames.shape[0]
    out = np.empty((n, config.target_h, config.target_w, 3), dtype=np.uint8)
    for i in range(n):
        out[i] = preprocess_frame(frames[i], config)
    return out


# ============================================================
# toylib_projects.wm.datagen.generate_vision_enc_data - /Users/anuj/Desktop/code/toylib/toylib_projects/wm/datagen/generate_vision_enc_data.py
# ============================================================

"""Materialize a vision-encoder (VAE) training dataset from Stage 1 raw.

Reads `data/raw/mode_MM_diff_D/episodes_shard_*.h5` produced by Stage 1 and
writes three flat HDF5 files containing pre-resized, pre-shuffled frames
ready for VAE training:

    data/compiled/
    ├── vae_train.h5    — N_train frames
    ├── vae_val.h5      — N_val   frames
    └── vae_test.h5     — N_test  frames

Each output file has the same flat schema (see "Output schema" below). The
training loop just opens these and reads sequentially — no Stage 1 access
needed at train time.


Decisions and rationale
-----------------------

**Why materialize at all (vs. sampling raw at train time)?**
Stage 1 stores complete episodes. A VAE wants individual shuffled frames at a
fixed resolution. The two access patterns are incompatible enough that doing
the conversion online costs a lot:

  - Random-access reads into LZ4-compressed HDF5 frames pay decompression
    cost per frame.
  - The same frame is read ~10–50× across all VAE training epochs; on-the-fly
    preprocessing redoes the resize each time.
  - Stage 1 layout may evolve; we don't want VAE training depending on it.

Materializing once trades a few minutes + a few GB of disk for fast,
reproducible training.

**Why episode-level splits (not frame-level)?**
Random frames within an episode are extremely correlated — adjacent frames
differ by a few pixels. If train/val/test split by frame, the val set
contains frames nearly identical to ones in train, and reconstruction error
is meaninglessly low. Splitting at the episode level guarantees the val/test
sets contain genuinely held-out trajectories.

Splits are assigned deterministically from `(shard_idx, episode_name)` via a
seeded hash so a re-run with the same `--base-seed` produces the same
partition.

**Why stratified sampling over (mode, difficulty, score-bucket)?**
Three competing concerns:

  1. **Diversity** — adjacent frames are near-duplicates. Use a temporal
     stride per source episode (controlled by `--input-fps`) to avoid them.
  2. **Coverage** — every game mode and difficulty must be represented. A
     pure-random sample over-represents whatever modes happen to have
     longer episodes.
  3. **Wall-state variety** — the canonical "full wall at start" frame shows
     up at the start of every episode. Pure-random sampling biases the
     dataset toward this homogeneous look. Score is a cheap proxy for
     "how much of the wall has been broken" — bucketing on it forces the
     sampler to spend a chunk of its budget on partially / mostly cleared
     walls.

We stratify candidate frames by `(mode, difficulty, score_bucket)` and
sample a target count from each non-empty stratum. The default bucket
boundaries `[0, 100, 500, 2000, +inf]` partition gameplay into
roughly: pristine wall / first few bricks broken / mid-game / late-game /
post-clear. With 4 modes × 2 difficulties × 5 buckets, the sampler has 40
strata to fill, which gives strong balance guarantees.

**Why an "input fps" knob instead of a raw stride?**
A user-friendlier framing: the Stage 1 data is at 60Hz native. Asking "what
effective frame rate do I want to sample at?" is more intuitive than asking
"what stride should I use?". Internally the implementation just computes
`stride = NATIVE_FPS // input_fps`. The default of 60 means "every frame is
a candidate", which is what the user gets if they don't think about it.

Lowering `--input-fps` is the right move for the VAE: at 60Hz consecutive
frames are nearly identical. For Track 2 a setting of `--input-fps 6` (stride
10) gives candidate frames spaced ~165 ms apart, which is a comfortable
visual diversity threshold.

**Why keep per-frame source metadata?**
Each output row carries `(shard_idx, episode_idx, frame_idx, mode,
difficulty, paddle_x, ball_x, ball_y, score, lives)`. These are tiny
(~40 bytes per frame, ~4 MB total at 100k frames) and unlock:

  - **Caption generation** at the same compile pass or later — the state
    values are right there, no need to re-load Stage 1.
  - **Stratified evaluation** — FID per mode, reconstruction error per
    score bucket, etc.
  - **Debugging** — "which raw episode did this weird sample come from?"

**Why three separate output files (train/val/test) instead of one with split
indicators?**
At training time we open exactly one of them and stream sequentially. Three
files = three flat memmaps with zero filter logic in the data loader. The
extra metadata overhead is negligible.

**Why a fixed shuffle inside each output file?**
The training loop is expected to do `seq[idx]` for `idx` in a per-epoch
shuffle, *not* a sequential read. But writing the file shuffled means a
*partial* read (e.g. first 10k for a quick sanity-train) is also random and
representative, instead of being all from one mode.


Output schema
-------------

Each output `.h5` file:

```
vae_split.h5
├── attrs:
│     n_frames       (int)
│     height         (int)
│     width          (int)
│     source_root    (str)
│     config_json    (str)   ← full PreprocessConfig + sampler config, for provenance
│     created_utc    (str)
├── frames           — (N, H, W, 3) uint8, LZ4-compressed
└── source/                  ← per-frame provenance + state, tiny vs frames
    ├── shard_idx        — (N,) int32
    ├── episode_idx      — (N,) int32   # index inside shard
    ├── frame_idx        — (N,) int32   # index inside episode
    ├── mode             — (N,) int32
    ├── difficulty       — (N,) int32
    ├── paddle_x         — (N,) float32
    ├── ball_x           — (N,) float32
    ├── ball_y           — (N,) float32
    ├── score            — (N,) int32
    └── lives            — (N,) int32
```

Usage (run from `toylib_projects/wm/`)
--------------------------------------

```
# Default: 100k train / 10k val / 10k test at 128×128 from data/raw/
uv run python -m datagen.generate_vision_enc_data

# Different shape / scale
uv run python -m datagen.generate_vision_enc_data     --target-size 128     --n-train 200000 --n-val 20000 --n-test 20000     --input-fps 6     --base-seed 0

# Side-by-side run on a smoke dataset
uv run python -m datagen.generate_vision_enc_data     --input-root /tmp/wm_smoke     --output-root /tmp/wm_vae_smoke
```
"""
NATIVE_FPS = 60
DEFAULT_SCORE_BUCKETS = (0, 100, 500, 2000)
DEFAULT_TRAIN_FRAC = 0.8
DEFAULT_VAL_FRAC = 0.1


class FrameSample(NamedTuple):
    """One selected source-frame, fully addressable into the raw shard tree."""

    shard_idx: int
    episode_name: str
    frame_idx: int
    mode: int
    difficulty: int
    paddle_x: float
    ball_x: float
    ball_y: float
    score: int
    lives: int
    score_bucket: int


def _find_shards(root: Path) -> list[Path]:
    """All `episodes_shard_*.h5` under root (recursive)."""
    return sorted(root.rglob("episodes_shard_*.h5"))


def _assign_split(
    episode_key: tuple[int, str], seed: int, train_frac: float, val_frac: float
) -> str:
    """Deterministic train/val/test from a stable per-episode key."""
    h = (seed, episode_key[0], episode_key[1])
    rng = np.random.default_rng(abs(hash(h)) % 2**32)
    r = rng.random()
    if r < train_frac:
        return "train"
    elif r < train_frac + val_frac:
        return "val"
    else:
        return "test"


def _score_bucket(score: int, boundaries: tuple[int, ...]) -> int:
    """Index of the bucket `score` falls into. 0 = first bucket, len(boundaries) = last."""
    for i, b in enumerate(boundaries):
        if score < b:
            return i
    return len(boundaries)


def _candidate_frames_for_episode(
    shard_idx: int,
    episode_name: str,
    grp: h5py.Group,
    stride: int,
    score_buckets: tuple[int, ...],
) -> Iterator[FrameSample]:
    """Yield strided candidate frames from one episode, tagged with state."""
    L = int(grp.attrs["length"])
    mode = int(grp.attrs.get("mode", 0))
    difficulty = int(grp.attrs.get("difficulty", 0))
    paddle_x = grp["states/paddle_x"][:]
    ball_x = grp["states/ball_x"][:]
    ball_y = grp["states/ball_y"][:]
    score = grp["states/score"][:]
    lives = grp["states/lives"][:]
    for t in range(0, L, stride):
        yield FrameSample(
            shard_idx=shard_idx,
            episode_name=episode_name,
            frame_idx=t,
            mode=mode,
            difficulty=difficulty,
            paddle_x=float(paddle_x[t]),
            ball_x=float(ball_x[t]),
            ball_y=float(ball_y[t]),
            score=int(score[t]),
            lives=int(lives[t]),
            score_bucket=_score_bucket(int(score[t]), score_buckets),
        )


def _enumerate_candidates(
    shards: list[Path],
    stride: int,
    score_buckets: tuple[int, ...],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, list[FrameSample]]:
    """Walk every shard / episode / strided frame, partitioned by split."""
    out: dict[str, list[FrameSample]] = {"train": [], "val": [], "test": []}
    for shard_idx, shard in enumerate(tqdm(shards, desc="indexing shards")):
        with h5py.File(shard, "r") as f:
            for ep_name in sorted(f.keys()):
                split = _assign_split((shard_idx, ep_name), seed, train_frac, val_frac)
                for cand in _candidate_frames_for_episode(
                    shard_idx, ep_name, f[ep_name], stride, score_buckets
                ):
                    out[split].append(cand)
    return out


def _stratified_sample(
    candidates: list[FrameSample], n_target: int, rng: np.random.Generator
) -> list[FrameSample]:
    """Pick ~n_target candidates with equal weight across (mode, diff, score_bucket).

    For each non-empty stratum we sample `n_target // n_strata` frames
    (uniform without replacement, up to the stratum size). If that doesn't
    reach `n_target` because some strata are small, we top up with extra
    samples from the largest strata.
    """
    if not candidates:
        return []
    strata: dict[tuple[int, int, int], list[FrameSample]] = defaultdict(list)
    for c in candidates:
        strata[c.mode, c.difficulty, c.score_bucket].append(c)
    n_strata = len(strata)
    per_stratum = n_target // n_strata
    selected: list[FrameSample] = []
    remainders: list[list[FrameSample]] = []
    for stratum_cands in strata.values():
        if len(stratum_cands) <= per_stratum:
            selected.extend(stratum_cands)
            continue
        idx = rng.choice(len(stratum_cands), size=per_stratum, replace=False)
        picked = [stratum_cands[i] for i in idx]
        selected.extend(picked)
        picked_set = set((int(i) for i in idx))
        remainders.append(
            [c for i, c in enumerate(stratum_cands) if i not in picked_set]
        )
    if len(selected) < n_target and remainders:
        pool = [c for r in remainders for c in r]
        if pool:
            need = n_target - len(selected)
            top_up_n = min(need, len(pool))
            idx = rng.choice(len(pool), size=top_up_n, replace=False)
            selected.extend((pool[i] for i in idx))
    rng.shuffle(selected)
    return selected[:n_target]


def _materialize_split(
    samples: list[FrameSample],
    shards: list[Path],
    output_path: Path,
    preproc: PreprocessConfig,
    config_json: str,
    source_root: Path,
) -> None:
    """Read selected raw frames, preprocess them, and write to a single .h5."""
    n = len(samples)
    h, w = (preproc.target_h, preproc.target_w)
    if n == 0:
        print(f"  (skip {output_path.name}: 0 samples)")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as out:
        out.attrs["n_frames"] = np.int32(n)
        out.attrs["height"] = np.int32(h)
        out.attrs["width"] = np.int32(w)
        out.attrs["source_root"] = str(source_root)
        out.attrs["config_json"] = config_json
        out.attrs["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        frames_ds = out.create_dataset(
            "frames",
            shape=(n, h, w, 3),
            dtype=np.uint8,
            chunks=(1, h, w, 3),
            **hdf5plugin.LZ4(),
        )
        src = out.create_group("source")
        src_arrays = {
            "shard_idx": np.empty(n, dtype=np.int32),
            "episode_idx": np.empty(n, dtype=np.int32),
            "frame_idx": np.empty(n, dtype=np.int32),
            "mode": np.empty(n, dtype=np.int32),
            "difficulty": np.empty(n, dtype=np.int32),
            "paddle_x": np.empty(n, dtype=np.float32),
            "ball_x": np.empty(n, dtype=np.float32),
            "ball_y": np.empty(n, dtype=np.float32),
            "score": np.empty(n, dtype=np.int32),
            "lives": np.empty(n, dtype=np.int32),
        }
        by_episode: dict[tuple[int, str], list[tuple[int, FrameSample]]] = defaultdict(
            list
        )
        for out_i, s in enumerate(samples):
            by_episode[s.shard_idx, s.episode_name].append((out_i, s))
        shard_ep_index: dict[tuple[int, str], int] = {}
        with_ep_idx_cache: dict[int, list[str]] = {}
        for (shard_idx, ep_name), pairs in tqdm(
            by_episode.items(), desc=f"materializing {output_path.name}"
        ):
            shard_path = shards[shard_idx]
            with h5py.File(shard_path, "r") as f:
                if shard_idx not in with_ep_idx_cache:
                    with_ep_idx_cache[shard_idx] = sorted(f.keys())
                ep_names = with_ep_idx_cache[shard_idx]
                grp = f[ep_name]
                wanted_idx = np.array([s.frame_idx for _, s in pairs], dtype=np.int64)
                order = np.argsort(wanted_idx)
                sorted_idx = wanted_idx[order]
                raw_frames = grp["frames"][sorted_idx.tolist()]
                inverse = np.argsort(order)
                raw_frames = raw_frames[inverse]
                proc = preprocess_frames(raw_frames, preproc)
                ep_idx_in_shard = ep_names.index(ep_name)
                for (out_i, s), proc_frame in zip(pairs, proc):
                    frames_ds[out_i] = proc_frame
                    src_arrays["shard_idx"][out_i] = shard_idx
                    src_arrays["episode_idx"][out_i] = ep_idx_in_shard
                    src_arrays["frame_idx"][out_i] = s.frame_idx
                    src_arrays["mode"][out_i] = s.mode
                    src_arrays["difficulty"][out_i] = s.difficulty
                    src_arrays["paddle_x"][out_i] = s.paddle_x
                    src_arrays["ball_x"][out_i] = s.ball_x
                    src_arrays["ball_y"][out_i] = s.ball_y
                    src_arrays["score"][out_i] = s.score
                    src_arrays["lives"][out_i] = s.lives
        for k, arr in src_arrays.items():
            src.create_dataset(k, data=arr)


def compile_vision_enc_dataset(
    *,
    input_root: Path,
    output_root: Path,
    n_train: int,
    n_val: int,
    n_test: int,
    input_fps: int,
    score_buckets: tuple[int, ...],
    preproc: PreprocessConfig,
    base_seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, Path]:
    """Public API: build the full train/val/test triple. Returns paths written."""
    shards = _find_shards(input_root)
    if not shards:
        raise SystemExit(
            f"No episodes_shard_*.h5 found under {input_root}. Run Stage 1 generation first (datagen.run_stage1)."
        )
    if NATIVE_FPS % input_fps != 0:
        raise ValueError(
            f"--input-fps must divide {NATIVE_FPS} evenly, got {input_fps}"
        )
    stride = NATIVE_FPS // input_fps
    print(f"Found {len(shards)} shard(s) under {input_root}")
    print(
        f"Sampling stride: {stride} (input_fps={input_fps}Hz from native {NATIVE_FPS}Hz)"
    )
    candidates = _enumerate_candidates(
        shards, stride, score_buckets, base_seed, train_frac, val_frac
    )
    for split, cands in candidates.items():
        print(f"  candidates[{split}] = {len(cands):,}")
    rng = np.random.default_rng(base_seed)
    targets = {"train": n_train, "val": n_val, "test": n_test}
    config_json = json.dumps(
        {
            "preprocess": asdict(preproc),
            "sampler": {
                "input_fps": input_fps,
                "stride": stride,
                "score_buckets": list(score_buckets),
                "train_frac": train_frac,
                "val_frac": val_frac,
                "base_seed": base_seed,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
            },
            "source_root": str(input_root),
        },
        sort_keys=True,
    )
    written: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        sampled = _stratified_sample(candidates[split], targets[split], rng)
        out_path = output_root / f"vae_{split}.h5"
        print(
            f"\n{split}: target={targets[split]:,}  sampled={len(sampled):,} → {out_path}"
        )
        _materialize_split(
            samples=sampled,
            shards=shards,
            output_path=out_path,
            preproc=preproc,
            config_json=config_json,
            source_root=input_root,
        )
        written[split] = out_path
    return written


# ============================================================
# toylib.nn.module - /Users/anuj/Desktop/code/toylib/toylib/nn/module.py
# ============================================================


def _is_array(x: typing.Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic)) or hasattr(
        x, "__jax_array__"
    )


def _is_random_key(x: str) -> bool:
    return x == "key"


def _is_supported_container(x: typing.Any) -> bool:
    return isinstance(x, (list, tuple))


def _wrap_init(orig: typing.Callable) -> typing.Callable:

    def wrapped(self) -> None:
        orig(self)
        for v in self.__dict__.values():
            if isinstance(v, Module) and (not hasattr(v, "_trainable_param_keys")):
                v.init()
            elif _is_supported_container(v):
                for elem in v:
                    if isinstance(elem, Module) and (
                        not hasattr(elem, "_trainable_param_keys")
                    ):
                        elem.init()
        self._trainable_param_keys = self._get_trainable_param_keys()
        if hasattr(self, "key"):
            self.key = None

    return wrapped


@dataclasses.dataclass
class Module(abc.ABC):
    """
    Defines a base class to use for the neural network modules in toylib.

    Assumes that all jax arrays are leaf nodes that are trainable and
    everything else is a static param. Defines the flatten and unflatten methods
    to make the modules compatible with jax `jit` and `grad` functions.

    Refer https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees

    Inspired by equinox and the Custom PyTres and Initialization section in jax docs.

    Every subclass automatically receives two dtype fields inherited from this base:

        param_dtype: storage dtype for trainable parameters (default float32).
        dtype: compute dtype for forward-pass operations (default float32).
    """

    param_dtype: np.dtype | type = jnp.float32
    dtype: np.dtype | type = jnp.float32

    def __init_subclass__(cls, **kwargs: typing.Any) -> None:
        """Initialize subclass as a dataclass and register as a pytree node.

        Sub-classes of dataclasses are not automatically dataclasses, so we need to explicitly convert them.
        We also register the class as a pytree with jax so that it can be used with jax transformations like jit and grad.

        Also wraps the subclass's init() to recursively initialize any sub-Module instances
        created during init(), then compute _trainable_param_keys. This means calling init()
        on the top-level module is sufficient to initialize the entire module tree.
        """
        super().__init_subclass__(**kwargs)
        cls = dataclasses.dataclass(cls, kw_only=True)
        cls = jax.tree_util.register_pytree_with_keys_class(cls)
        if "init" in cls.__dict__:
            original_init = cls.__dict__["init"]
            cls.init = _wrap_init(original_init)

    @abc.abstractmethod
    def init(self) -> None:
        """Initialize all the trainable parameters in the module."""
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> typing.Any:
        """Run a forward pass of the module."""
        pass

    def _get_trainable_param_keys(self) -> list[str]:
        """Get the list of attribute names that are trainable parameters."""
        param_keys = []
        for k, v in self.__dict__.items():
            if (
                _is_array(v)
                and (not _is_random_key(k))
                or isinstance(v, Module)
                or (
                    _is_supported_container(v)
                    and all((isinstance(elem, Module) for elem in v))
                )
            ):
                param_keys.append(k)
        return param_keys

    def tree_flatten_with_keys(self) -> tuple:
        params_with_keys = []
        aux_data = dict()
        for k, v in self.__dict__.items():
            if k not in self._trainable_param_keys:
                aux_data[k] = v
        for k in self._trainable_param_keys:
            v = self.__dict__[k]
            params_with_keys.append((jax.tree_util.GetAttrKey(k), v))
        return (params_with_keys, aux_data)

    @classmethod
    def tree_unflatten(cls, static, dynamic) -> "Module":
        obj = object.__new__(cls)
        param_keys = static["_trainable_param_keys"]
        for k, v in zip(param_keys, dynamic):
            obj.__setattr__(k, v)
        for k, v in static.items():
            obj.__setattr__(k, v)
        return obj


# ============================================================
# toylib.nn.layers - /Users/anuj/Desktop/code/toylib/toylib/nn/layers.py
# ============================================================


class Linear(Module):
    """Defines a simple feedforward layer: which is a linear transformation."""

    in_features: int
    out_features: int
    key: jt.PRNGKeyArray
    use_bias: bool = False
    init_std: typing.Optional[float] = None
    weights: typing.Optional[jt.Float[jt.Array, "in_features out_features"]] = None
    bias: typing.Optional[jt.Float[jt.Array, " out_features"]] = None

    def init(self) -> None:
        w_key = self.key
        in_features = self.in_features
        out_features = self.out_features
        if self.init_std is not None:
            std = self.init_std
            s = std * math.sqrt(3)
            self.weights = jax.random.uniform(
                key=w_key, shape=(in_features, out_features), minval=-s, maxval=s
            ).astype(self.param_dtype)
        else:
            std = min(1.0, math.sqrt(out_features / in_features)) / math.sqrt(
                in_features
            )
            self.weights = (
                jax.random.normal(key=w_key, shape=(in_features, out_features)) * std
            ).astype(self.param_dtype)
        self.bias = (
            jax.numpy.zeros((out_features,), dtype=self.param_dtype)
            if self.use_bias
            else None
        )

    def __call__(
        self, x: jt.Float[jt.Array, "... in_features"]
    ) -> jt.Float[jt.Array, "... out_features"]:
        x = jax.numpy.dot(x.astype(self.dtype), self.weights.astype(self.dtype))
        if self.use_bias:
            x = x + self.bias.astype(self.dtype)
        return x


class Embedding(Module):
    """Defines an embedding layer that stores an embedding matrix for discrete tokens."""

    vocab_size: int
    embedding_dim: int
    key: jt.PRNGKeyArray
    weights: typing.Optional[jt.Float[jt.Array, "vocab_size embedding_dim"]] = None

    def init(self) -> None:
        self.weights = jax.random.normal(
            self.key, (self.vocab_size, self.embedding_dim)
        ).astype(self.param_dtype)

    def __call__(
        self, tokens: jt.Integer[jt.Array, "... seq_len"]
    ) -> jt.Float[jt.Array, "... seq_len embedding_dim"]:
        return jax.numpy.take(self.weights, tokens, axis=0).astype(self.dtype)


class Conv2D(Module):
    """2D convolution with NHWC layout, optional bias, and 'SAME' or integer padding.

    Uses ``jax.lax.conv_general_dilated`` under the hood. Kernels are stored as
    ``(kernel_size, kernel_size, in_channels, out_channels)`` (HWIO layout) to
    match the NHWC input convention.

    Weight init follows the same pattern as Linear: if ``init_std`` is set,
    weights are drawn uniformly in ``[-init_std*sqrt(3), +init_std*sqrt(3)]``;
    otherwise a fan-in-aware default ``min(1, sqrt(out/in)) / sqrt(in)`` is
    used (matching arXiv:2310.17813), where ``in/out`` here are the effective
    fan-in / fan-out of the conv (``kernel_size^2 * channels``).
    """

    in_channels: int
    out_channels: int
    key: jt.PRNGKeyArray
    kernel_size: int = 3
    stride: int = 1
    padding: typing.Union[int, str] = "SAME"
    use_bias: bool = True
    init_std: typing.Optional[float] = None
    weights: typing.Optional[jt.Float[jt.Array, "kh kw in_channels out_channels"]] = (
        None
    )
    bias: typing.Optional[jt.Float[jt.Array, " out_channels"]] = None

    def init(self) -> None:
        k = self.kernel_size
        fan_in = k * k * self.in_channels
        fan_out = k * k * self.out_channels
        if self.init_std is not None:
            std = self.init_std
            s = std * math.sqrt(3)
            self.weights = jax.random.uniform(
                key=self.key,
                shape=(k, k, self.in_channels, self.out_channels),
                minval=-s,
                maxval=s,
            ).astype(self.param_dtype)
        else:
            std = min(1.0, math.sqrt(fan_out / fan_in)) / math.sqrt(fan_in)
            self.weights = (
                jax.random.normal(
                    key=self.key, shape=(k, k, self.in_channels, self.out_channels)
                )
                * std
            ).astype(self.param_dtype)
        self.bias = (
            jnp.zeros((self.out_channels,), dtype=self.param_dtype)
            if self.use_bias
            else None
        )

    def _resolved_padding(self) -> typing.Union[str, list[tuple[int, int]]]:
        if isinstance(self.padding, str):
            return self.padding
        p = int(self.padding)
        return [(p, p), (p, p)]

    def __call__(
        self, x: jt.Float[jt.Array, "B H W in_channels"]
    ) -> jt.Float[jt.Array, "B H_out W_out out_channels"]:
        x = jax.lax.conv_general_dilated(
            x.astype(self.dtype),
            self.weights.astype(self.dtype),
            window_strides=(self.stride, self.stride),
            padding=self._resolved_padding(),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.use_bias:
            x = x + self.bias.astype(self.dtype)
        return x


class GroupNorm(Module):
    """Group Normalization over the channel dimension of an NHWC tensor.

    Splits the ``num_features`` channels into ``num_groups`` equal-sized groups
    and normalizes each group's activations (over the (H, W, C/G) volume per
    sample). Learnable per-channel scale and bias are applied after norm.

    Statistics are computed in float32 for numerical stability and cast back
    to ``self.dtype`` on the way out. Matches the convention used by
    ``rms_norm`` below.
    """

    num_features: int
    num_groups: int = 32
    eps: float = 1e-05
    scale: typing.Optional[jt.Float[jt.Array, " num_features"]] = None
    bias: typing.Optional[jt.Float[jt.Array, " num_features"]] = None

    def init(self) -> None:
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"num_features ({self.num_features}) must be divisible by num_groups ({self.num_groups})"
            )
        self.scale = jnp.ones((self.num_features,), dtype=self.param_dtype)
        self.bias = jnp.zeros((self.num_features,), dtype=self.param_dtype)

    def __call__(
        self, x: jt.Float[jt.Array, "B H W num_features"]
    ) -> jt.Float[jt.Array, "B H W num_features"]:
        orig_dtype = x.dtype
        B, H, W, C = x.shape
        G = self.num_groups
        x32 = x.astype(jnp.float32).reshape(B, H, W, G, C // G)
        mean = jnp.mean(x32, axis=(1, 2, 4), keepdims=True)
        var = jnp.var(x32, axis=(1, 2, 4), keepdims=True)
        x32 = (x32 - mean) * jax.lax.rsqrt(var + self.eps)
        x32 = x32.reshape(B, H, W, C)
        scale = self.scale.astype(jnp.float32).reshape(1, 1, 1, C)
        bias = self.bias.astype(jnp.float32).reshape(1, 1, 1, C)
        return (x32 * scale + bias).astype(orig_dtype)


def upsample_nearest(
    x: jt.Float[jt.Array, "B H W C"], factor: int = 2
) -> jt.Float[jt.Array, "B H_out W_out C"]:
    """Nearest-neighbor upsample of an NHWC image by ``factor`` along H and W.

    Pure function — no trainable parameters. Used in the VAE decoder to expand
    the spatial grid before a regular convolution. Avoids the checkerboard
    artifacts that transposed convolutions exhibit.
    """
    B, H, W, C = x.shape
    return jax.image.resize(x, (B, H * factor, W * factor, C), method="nearest")


def rms_norm(x: jt.Float[jt.Array, "... dim"]) -> jt.Float[jt.Array, "... dim"]:
    """Applies RMS Normalization over the last dimension of the input tensor.

    The mean-square computation is done in float32 for numerical stability,
    regardless of the input dtype. The output is cast back to the input dtype.

    Args:
        x: Input tensor

    Returns:
        The RMS normalized tensor of the same shape as input x.
    """
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-09)
    return (x / rms).astype(orig_dtype)


# ============================================================
# toylib.nn.attention - /Users/anuj/Desktop/code/toylib/toylib/nn/attention.py
# ============================================================


class RotaryPositionalEmbedding(Module):
    """Implements Rotary Positional Embeddings (RoPE) as described in https://arxiv.org/abs/2104.09864."""

    seq_len: int = 1024
    qkv_dim: int = 128
    base: int = 100000

    def init(self) -> None:
        positions = jnp.arange(0, self.seq_len)
        freqs = self.base ** (jnp.arange(0, self.qkv_dim, 2) / self.qkv_dim)
        self.gamma = einops.einsum(positions, 1.0 / freqs, "t, d -> t d")
        self.cos = jnp.cos(self.gamma).astype(self.param_dtype)
        self.sin = jnp.sin(self.gamma).astype(self.param_dtype)

    def __call__(
        self, x: jt.Float[jt.Array, "... seq_len qkv_dim"], t0: int = 0
    ) -> jt.Float[jt.Array, "... seq_len qkv_dim"]:
        t, d = x.shape[-2:]
        if t0 + t > self.seq_len:
            raise ValueError(
                f"Position index out of range of RoPE cache:t0 ({t0}) + t ({t}) > seq_len ({self.seq_len})"
            )
        sin = self.sin[t0 : t0 + t, :].astype(self.dtype)
        cos = self.cos[t0 : t0 + t, :].astype(self.dtype)
        x1, x2 = (
            x[..., : d // 2].astype(self.dtype),
            x[..., d // 2 :].astype(self.dtype),
        )
        es_shape = "... t d, t d -> ... t d"
        y1 = einops.einsum(x1, cos, es_shape) + einops.einsum(x2, sin, es_shape)
        y2 = -einops.einsum(x1, sin, es_shape) + einops.einsum(x2, cos, es_shape)
        return jnp.concatenate([y1, y2], axis=-1)


def scaled_dot_product_attention(
    q: jt.Float[jt.Array, "... seq_len qkv_dim"],
    k: jt.Float[jt.Array, "... seq_len qkv_dim"],
    v: jt.Float[jt.Array, "... seq_len qkv_dim"],
    mask: typing.Optional[jt.Float[jt.Array, "... seq_len seq_len"]],
) -> tuple[
    jt.Float[jt.Array, "... seq_len qkv_dim"], jt.Float[jt.Array, "... seq_len seq_len"]
]:
    """Compute scaled dot product attention.

    Given query (`q`), key (`k`), and value (`v`) tensors, this function first computes the
    attention weights as the softmax of the dot product of `q` and `k`, scaled by the square
    root of the dimension of the keys. If a mask is provided, it is applied to the attention
    logits before the softmax is computed.

    Finally, the attention weights are used to compute the weighted average of the given values.

    NOTE: the batch dimension is not explicitly handled in this function.

    Args:
        q: query tensor
        k: keys tensor
        v: values tensor
        mask: optional boolean mask to apply to the attention logits

    Returns:
        tuple of final values and attention weights

    """
    d_k = q.shape[-1]
    assert q.shape[-1] == k.shape[-1], "q and k must have the same feature dimension"
    attention_logits = jnp.matmul(q, k.swapaxes(-1, -2)) / jnp.sqrt(d_k)
    if mask is not None:
        attention_logits = jnp.where(mask, attention_logits, -1000000000.0)
    attention_weights = jax.nn.softmax(
        attention_logits.astype(jnp.float32), axis=-1
    ).astype(q.dtype)
    values = jnp.matmul(attention_weights, v)
    return (values, attention_weights)


class MultiHeadAttention(Module):
    """
    The MultiHeadAttention defines `num_heads` attention heads. For the given input `Q`, `K`, `V`
    tensors, `num_head` linear projections of dim `qkv_dim / num_heads` are produced.

    An attention weight is then computed using the scaled dot product attention method. The
    weighted average of the values are then concatenated from the various heads to produce a
    single output value vector. A final linear layer is applied on top of this with non-linearity.
    """

    qkv_dim: int
    num_heads: int
    key: jt.PRNGKeyArray
    use_qk_norm: bool = True

    def init(self) -> None:
        qkv_dim = self.qkv_dim
        keys = jax.random.split(self.key, 4)
        init_std = 1 / math.sqrt(qkv_dim)
        self.q_projection = Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[0],
            init_std=init_std,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.k_projection = Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[1],
            init_std=init_std,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.v_projection = Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[2],
            init_std=init_std,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.linear = Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[3],
            init_std=0.0,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(
        self,
        Q: jt.Float[jt.Array, "... seq_len qkv_dim"],
        K: jt.Float[jt.Array, "... seq_len qkv_dim"],
        V: jt.Float[jt.Array, "... seq_len qkv_dim"],
        mask: typing.Optional[jt.Float[jt.Array, "... seq_len seq_len"]] = None,
        *,
        rope: typing.Optional[RotaryPositionalEmbedding] = None,
        return_attention_weights: bool = False,
    ) -> typing.Union[
        tuple[
            jt.Float[jt.Array, "... seq_len qkv_dim"],
            jt.Float[jt.Array, "... seq_len seq_len"],
        ],
        jt.Float[jt.Array, "... seq_len qkv_dim"],
    ]:
        Q = self.q_projection(Q)
        K = self.k_projection(K)
        V = self.v_projection(V)
        Q = einops.rearrange(
            Q,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        K = einops.rearrange(
            K,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        V = einops.rearrange(
            V,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        if mask is not None:
            mask = einops.rearrange(
                mask, "... seq_len1 seq_len2 -> ... 1 seq_len1 seq_len2"
            )
        if rope is not None:
            Q = rope(Q)
            K = rope(K)
        if self.use_qk_norm:
            Q = rms_norm(Q)
            K = rms_norm(K)
        values, attention_weights = scaled_dot_product_attention(
            q=Q, k=K, v=V, mask=mask
        )
        values = einops.rearrange(
            values, "... num_heads seq_len d -> ... seq_len (num_heads d)"
        )
        values = self.linear(values)
        if return_attention_weights:
            return (values, attention_weights)
        return values


# ============================================================
# toylib_projects.wm.vision_encoder.model - /Users/anuj/Desktop/code/toylib/toylib_projects/wm/vision_encoder/model.py
# ============================================================

"""KL-regularized VAE for the Track A1 vision codec.

Implements the architecture specified in
``docs/walkthroughs/a1_vision_codec.md`` (which is the actionable
distillation of ``docs/designs/vision_codec.md``).

3 stride-2 downsample stages give an 8× spatial reduction, so for an
input of ``(B, H, W, 3)`` the latent grid is ``(B, H/8, W/8, latent_channels)``.

Encoder pipeline (per the walkthrough — unchanged):
    Conv → ResBlock → Down → ResBlock → Down → ResBlock → Down →
    ResBlock → AttnBlock → ResBlock → GN+SiLU → Conv → split (μ, log σ²)

Decoder mirrors it, with each up-step implemented as
``upsample_nearest → Conv2D`` (channel-preserving) followed by a
ResBlock for channel reduction. Ends with Tanh so outputs land in (-1, 1).

Built on the toylib ``Module`` base class (dataclass-style,
pytree-registered). All convolutions / GroupNorm / nearest-neighbor
upsample come from ``toylib.nn.layers``; attention is reused from
``toylib.nn.attention``.

Loss pieces live in this file as pure functions:
  - ``reparameterize``                — the differentiable ε-trick
  - ``kl_divergence``                  — closed-form KL(q(z|x) || N(0, I))
  - ``recon_loss_l1``                  — mean L1 over (-1, 1) targets
  - ``beta_warmup``                    — linear KL warmup schedule
  - ``vae_loss``                       — assembles the train-time loss

Perceptual + auxiliary losses (walkthrough Milestone 5) are deliberately
left out of this file — add them in a wrapper once base reconstruction is
stable.
"""


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the Track A1 VAE.

    Defaults match the walkthrough: 64×64x3 input, 8×8×4 latent, base_ch=64
    (so 4×base_ch = 256 channels at the bottleneck).
    """

    base_ch: int = 64
    latent_channels: int = 4
    input_channels: int = 3
    num_attn_heads: int = 1
    num_norm_groups: int = 32
    log_sigma_sq_clip_min: float = -30.0
    log_sigma_sq_clip_max: float = 20.0


class ResBlock(Module):
    """Two-conv pre-activation residual block.

    GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv, plus a skip
    connection. When ``in_channels != out_channels`` the skip path goes
    through a 1×1 conv to match shapes (otherwise the skip is identity).
    """

    in_channels: int
    out_channels: int
    key: jt.PRNGKeyArray
    num_groups: int = 32

    def init(self) -> None:
        keys = jax.random.split(self.key, 3)
        self.norm1 = GroupNorm(
            num_features=self.in_channels,
            num_groups=self.num_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.conv1 = Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding="SAME",
            key=keys[0],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.norm2 = GroupNorm(
            num_features=self.out_channels,
            num_groups=self.num_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.conv2 = Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding="SAME",
            key=keys[1],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        if self.in_channels != self.out_channels:
            self.skip = Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                padding="SAME",
                key=keys[2],
                param_dtype=self.param_dtype,
                dtype=self.dtype,
            )
        else:
            self.skip = None

    def __call__(
        self, x: jt.Float[jt.Array, "B H W in_channels"]
    ) -> jt.Float[jt.Array, "B H W out_channels"]:
        h = jax.nn.silu(self.norm1(x))
        h = self.conv1(h)
        h = jax.nn.silu(self.norm2(h))
        h = self.conv2(h)
        skip = x if self.skip is None else self.skip(x)
        return h + skip


class AttentionBlock(Module):
    """Single self-attention block at the spatial bottleneck.

    Pre-norm with GroupNorm, flatten the (H, W) grid into an (H*W)-long
    sequence, run multi-head self-attention, reshape back, residual add.

    Reuses the existing toylib ``MultiHeadAttention``; the output linear
    inside it is zero-initialized, so this block is the identity at init
    (helpful for training stability).
    """

    channels: int
    key: jt.PRNGKeyArray
    num_heads: int = 1
    num_groups: int = 32

    def init(self) -> None:
        self.norm = GroupNorm(
            num_features=self.channels,
            num_groups=self.num_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.attn = MultiHeadAttention(
            qkv_dim=self.channels,
            num_heads=self.num_heads,
            key=self.key,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(
        self, x: jt.Float[jt.Array, "B H W C"]
    ) -> jt.Float[jt.Array, "B H W C"]:
        B, H, W, C = x.shape
        h = self.norm(x)
        h_seq = h.reshape(B, H * W, C)
        h_seq = self.attn(h_seq, h_seq, h_seq)
        h = h_seq.reshape(B, H, W, C)
        return x + h


class Encoder(Module):
    """Down-3× conv encoder producing per-spatial Gaussian parameters.

    Output is split along the channel axis into ``μ`` and ``log σ²``.
    ``log σ²`` is clipped before any downstream ``exp`` to avoid NaNs at init.
    """

    config: ModelConfig
    key: jt.PRNGKeyArray

    def init(self) -> None:
        cfg = self.config
        ch = cfg.base_ch
        keys = jax.random.split(self.key, 12)
        self.conv_in = Conv2D(
            in_channels=cfg.input_channels,
            out_channels=ch,
            kernel_size=3,
            padding="SAME",
            key=keys[0],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res1 = ResBlock(
            in_channels=ch,
            out_channels=ch,
            key=keys[1],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.down1 = Conv2D(
            in_channels=ch,
            out_channels=2 * ch,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=keys[2],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res2 = ResBlock(
            in_channels=2 * ch,
            out_channels=2 * ch,
            key=keys[3],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.down2 = Conv2D(
            in_channels=2 * ch,
            out_channels=4 * ch,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=keys[4],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res3 = ResBlock(
            in_channels=4 * ch,
            out_channels=4 * ch,
            key=keys[5],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.down3 = Conv2D(
            in_channels=4 * ch,
            out_channels=4 * ch,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=keys[6],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res4 = ResBlock(
            in_channels=4 * ch,
            out_channels=4 * ch,
            key=keys[7],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.attn = AttentionBlock(
            channels=4 * ch,
            num_heads=cfg.num_attn_heads,
            key=keys[8],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res5 = ResBlock(
            in_channels=4 * ch,
            out_channels=4 * ch,
            key=keys[9],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.norm_out = GroupNorm(
            num_features=4 * ch,
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.conv_out = Conv2D(
            in_channels=4 * ch,
            out_channels=2 * cfg.latent_channels,
            kernel_size=3,
            padding="SAME",
            key=keys[10],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(
        self, x: jt.Float[jt.Array, "B H W 3"]
    ) -> tuple[
        jt.Float[jt.Array, "B h w latent_channels"],
        jt.Float[jt.Array, "B h w latent_channels"],
    ]:
        h = self.conv_in(x)
        h = self.res1(h)
        h = self.down1(h)
        h = self.res2(h)
        h = self.down2(h)
        h = self.res3(h)
        h = self.down3(h)
        h = self.res4(h)
        h = self.attn(h)
        h = self.res5(h)
        h = jax.nn.silu(self.norm_out(h))
        h = self.conv_out(h)
        mu, log_sigma_sq = jnp.split(h, 2, axis=-1)
        log_sigma_sq = jnp.clip(
            log_sigma_sq,
            self.config.log_sigma_sq_clip_min,
            self.config.log_sigma_sq_clip_max,
        )
        return (mu, log_sigma_sq)


class Decoder(Module):
    """Mirror decoder: bottleneck attention then 3× nearest-neighbor upsample.

    Each upsample step is implemented as ``upsample_nearest → Conv2D``
    (channel-preserving smoothing conv), followed by a ResBlock that does
    the channel reduction. This avoids the checkerboard artifacts of
    transposed convolutions.
    """

    config: ModelConfig
    key: jt.PRNGKeyArray

    def init(self) -> None:
        cfg = self.config
        ch = cfg.base_ch
        keys = jax.random.split(self.key, 12)
        self.conv_in = Conv2D(
            in_channels=cfg.latent_channels,
            out_channels=4 * ch,
            kernel_size=3,
            padding="SAME",
            key=keys[0],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res1 = ResBlock(
            in_channels=4 * ch,
            out_channels=4 * ch,
            key=keys[1],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.attn = AttentionBlock(
            channels=4 * ch,
            num_heads=cfg.num_attn_heads,
            key=keys[2],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res2 = ResBlock(
            in_channels=4 * ch,
            out_channels=4 * ch,
            key=keys[3],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.up1_conv = Conv2D(
            in_channels=4 * ch,
            out_channels=4 * ch,
            kernel_size=3,
            padding="SAME",
            key=keys[4],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res3 = ResBlock(
            in_channels=4 * ch,
            out_channels=2 * ch,
            key=keys[5],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.up2_conv = Conv2D(
            in_channels=2 * ch,
            out_channels=2 * ch,
            kernel_size=3,
            padding="SAME",
            key=keys[6],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res4 = ResBlock(
            in_channels=2 * ch,
            out_channels=ch,
            key=keys[7],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.up3_conv = Conv2D(
            in_channels=ch,
            out_channels=ch,
            kernel_size=3,
            padding="SAME",
            key=keys[8],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res5 = ResBlock(
            in_channels=ch,
            out_channels=ch,
            key=keys[9],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.norm_out = GroupNorm(
            num_features=ch,
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.conv_out = Conv2D(
            in_channels=ch,
            out_channels=cfg.input_channels,
            kernel_size=3,
            padding="SAME",
            key=keys[10],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(
        self, z: jt.Float[jt.Array, "B h w latent_channels"]
    ) -> jt.Float[jt.Array, "B H W 3"]:
        h = self.conv_in(z)
        h = self.res1(h)
        h = self.attn(h)
        h = self.res2(h)
        h = upsample_nearest(h, factor=2)
        h = self.up1_conv(h)
        h = self.res3(h)
        h = upsample_nearest(h, factor=2)
        h = self.up2_conv(h)
        h = self.res4(h)
        h = upsample_nearest(h, factor=2)
        h = self.up3_conv(h)
        h = self.res5(h)
        h = jax.nn.silu(self.norm_out(h))
        h = self.conv_out(h)
        return jnp.tanh(h)


def reparameterize(
    mu: jt.Float[jt.Array, "B h w C"],
    log_sigma_sq: jt.Float[jt.Array, "B h w C"],
    rng_key: jt.PRNGKeyArray,
) -> jt.Float[jt.Array, "B h w C"]:
    """Reparameterization trick: ``z = μ + σ · ε`` with ``ε ~ N(0, I)``.

    Differentiable in both ``μ`` and ``log σ²``; the only non-differentiable
    bit (the normal sample) is held in ``ε`` and gradients flow around it.

    At **inference** time, prefer ``z = μ`` directly (no noise) — this
    function is only needed during VAE training.
    """
    sigma = jnp.exp(0.5 * log_sigma_sq)
    eps = jax.random.normal(rng_key, mu.shape, dtype=mu.dtype)
    return mu + sigma * eps


def kl_divergence(
    mu: jt.Float[jt.Array, "B h w C"], log_sigma_sq: jt.Float[jt.Array, "B h w C"]
) -> jt.Float[jt.Array, ""]:
    """Closed-form KL( N(μ, σ²) || N(0, I) ), summed over latent dims.

    ``L_KL = 0.5 · mean_B( sum_{h,w,C} (μ² + σ² − log σ² − 1) )``.

    Sum over (h, w, C) **then** mean over the batch — matches the Stable
    Diffusion / walkthrough convention. Swapping sum/mean here scales the
    loss magnitude by ``h*w*C`` (256 for the default config).
    """
    sigma_sq = jnp.exp(log_sigma_sq)
    per_sample = 0.5 * jnp.sum(mu**2 + sigma_sq - log_sigma_sq - 1.0, axis=(1, 2, 3))
    return jnp.mean(per_sample)


def recon_loss_l1(
    recon: jt.Float[jt.Array, "B H W C"], target: jt.Float[jt.Array, "B H W C"]
) -> jt.Float[jt.Array, ""]:
    """Mean L1 over pixels. Both args are assumed in ``[-1, 1]``."""
    return jnp.mean(jnp.abs(recon - target))


def beta_warmup(
    step: int | jt.Array, warmup_steps: int, beta_max: float
) -> jt.Float[jt.Array, ""]:
    """Linear KL warmup: ``β(step) = (step / warmup_steps) · β_max``, capped.

    Steps 0..warmup_steps ramp from 0 → β_max; thereafter β stays at β_max.
    The walkthrough recommends ``β_max = 1e-6`` and
    ``warmup_steps = 10_000`` to prevent posterior collapse early in
    training. Returns a jnp scalar so it can be threaded through jit.
    """
    if warmup_steps <= 0:
        return jnp.asarray(beta_max, jnp.float32)
    frac = jnp.minimum(jnp.asarray(step, jnp.float32) / warmup_steps, 1.0)
    return (frac * beta_max).astype(jnp.float32)


class VAE(Module):
    """Encoder + decoder bundled together.

    For inference, call ``encode`` and ``decode`` directly. For training,
    use ``__call__(x, rng_key)`` which returns ``(recon, aux)`` where
    ``aux`` contains ``mu``, ``log_sigma_sq``, and ``z`` for downstream
    loss computation.
    """

    config: ModelConfig
    key: jt.PRNGKeyArray

    def init(self) -> None:
        keys = jax.random.split(self.key, 2)
        self.encoder = Encoder(
            config=self.config,
            key=keys[0],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.decoder = Decoder(
            config=self.config,
            key=keys[1],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def encode(
        self, x: jt.Float[jt.Array, "B H W 3"]
    ) -> tuple[jt.Float[jt.Array, "B h w C"], jt.Float[jt.Array, "B h w C"]]:
        return self.encoder(x)

    def decode(self, z: jt.Float[jt.Array, "B h w C"]) -> jt.Float[jt.Array, "B H W 3"]:
        return self.decoder(z)

    def __call__(
        self,
        x: jt.Float[jt.Array, "B H W 3"],
        rng_key: typing.Optional[jt.PRNGKeyArray] = None,
    ) -> tuple[jt.Float[jt.Array, "B H W 3"], dict[str, jt.Array]]:
        mu, log_sigma_sq = self.encode(x)
        if rng_key is None:
            z = mu
        else:
            z = reparameterize(mu, log_sigma_sq, rng_key)
        recon = self.decode(z)
        return (recon, {"mu": mu, "log_sigma_sq": log_sigma_sq, "z": z})


def vae_loss(
    model: VAE,
    batch: jt.Float[jt.Array, "B H W 3"],
    rng_key: jt.PRNGKeyArray,
    beta: jt.Float[jt.Array, ""] | float = 1e-06,
) -> tuple[jt.Float[jt.Array, ""], dict[str, jt.Array]]:
    """Base VAE training loss: ``L_rec + β · L_KL``.

    Inputs are expected in ``[-1, 1]`` float32. Returns ``(loss, aux)`` where
    ``aux`` contains the individual loss components plus the model's
    intermediate tensors — suitable for plugging into the existing
    ``Experiment.forward_fn`` contract.

    Perceptual + auxiliary ball-position losses (walkthrough Milestone 5)
    are deliberately not included here; add them in a wrapper once base
    reconstruction is healthy.
    """
    recon, model_aux = model(batch, rng_key=rng_key)
    l_rec = recon_loss_l1(recon, batch)
    l_kl = kl_divergence(model_aux["mu"], model_aux["log_sigma_sq"])
    total = l_rec + beta * l_kl
    aux = {
        "l_rec": l_rec,
        "l_kl": l_kl,
        "beta": jnp.asarray(beta, jnp.float32),
        "recon": recon,
        **model_aux,
    }
    return (total, aux)


# ============================================================
# toylib_projects.wm.vision_encoder.inference - /Users/anuj/Desktop/code/toylib/toylib_projects/wm/vision_encoder/inference.py
# ============================================================

"""Shared inference helpers for the Track A1 KL-VAE vision codec.

This is the one place that knows how to **load a trained VAE and push frames
through it**. Both the evaluation script (``vision_encoder.evaluate``) and the
Stage 2 data-generation pipeline import these helpers, so the exact same
encode/decode path — normalization, deterministic ``z = mu``, batching — is
used everywhere. That guarantees the latents cached for the world model are
bit-for-bit the ones the evaluation measured.

Everything here is **Colab-friendly**: plain functions with keyword defaults,
numpy in / numpy out, no ``Experiment`` / ``Task`` scaffolding required. A
typical interactive session is::

    from toylib_projects.wm.vision_encoder import inference

    frames, source, cfg = inference.load_frames("data/compiled/vae_test.h5", n=256)
    vae = inference.load_vae("/path/to/ckpt/<run_id>")   # latest step by default
    recons = inference.reconstruct(vae, frames)          # (256, H, W, 3) uint8

Design notes
------------

**Deterministic encoding.** At inference we use ``z = mu`` (no reparameterization
noise) — see ``model.VAE.__call__``'s inference branch. ``encode_frames`` returns
``mu`` directly.

**Normalization lives here.** The compiled datasets store uint8 ``[0, 255]``
frames; the VAE consumes float32 ``[-1, 1]``. ``encode_frames`` / ``decode_latents``
apply the ``/127.5 - 1`` map (and its inverse) so callers pass and receive plain
uint8 frames.

**Batching in host code.** The jitted encode/decode operate on a fixed device
batch; the Python wrappers chunk arbitrarily large inputs into ``batch_size``
pieces so a 20k-frame test set fits without OOM. The final short batch is padded
up to ``batch_size`` and trimmed back, so the jitted function only ever sees one
input shape (one compile).
"""


def latest_step(checkpoint_dir: str | Path) -> int:
    """Return the most recent step saved under ``checkpoint_dir``.

    Raises ``ValueError`` if the directory holds no checkpoints. ``checkpoint_dir``
    may be a local path or a ``gs://bucket/...`` URI.
    """
    manager = ocp.CheckpointManager(
        str(checkpoint_dir), options=ocp.CheckpointManagerOptions()
    )
    try:
        step = manager.latest_step()
    finally:
        manager.close()
    if step is None:
        raise ValueError(f"No checkpoints found under {checkpoint_dir!r}")
    return int(step)


def load_vae(
    checkpoint_dir: str | Path,
    step: int | None = None,
    *,
    base_ch: int = 64,
    latent_channels: int = 4,
    config: ModelConfig | None = None,
    seed: int = 0,
    key: jax.random.PRNGKey | None = None,
) -> VAE:
    """Restore a VAE saved by ``vision_encoder.train``.

    Builds a template VAE with a matching config (so the pytree structure lines
    up with the saved arrays), then restores just the ``model`` item from the
    composite checkpoint (optimizer / dataset-iterator items are ignored).

    Parameters
    ----------
    checkpoint_dir :
        Directory holding the orbax checkpoint (the ``.../<run_id>`` folder that
        ``train`` writes to). Local path or ``gs://bucket/...`` URI.
    step :
        Which saved step to restore. ``None`` (default) restores the latest.
    base_ch, latent_channels :
        VAE shape — **must match the trained checkpoint**. Ignored if ``config``
        is given.
    config :
        Full ``ModelConfig``; overrides ``base_ch`` / ``latent_channels`` if set.
    seed :
        Seed for the template's (immediately overwritten) init weights. Ignored if `key` is set.
    key: Optional random key for weights init.

    Returns
    -------
    model_lib.VAE :
        The restored model, ready for ``encode`` / ``decode``.
    """
    if config is None:
        config = ModelConfig(base_ch=base_ch, latent_channels=latent_channels)
    if step is None:
        step = latest_step(checkpoint_dir)
    if key is None:
        key = jax.random.key(seed)
    template_vae = VAE(config=config, key=key)
    template_vae.init()
    template = jax.tree.map(np.asarray, template_vae)
    manager = ocp.CheckpointManager(
        str(checkpoint_dir), options=ocp.CheckpointManagerOptions()
    )
    try:
        restored = manager.restore(
            step, args=ocp.args.Composite(model=ocp.args.StandardRestore(template))
        )
    finally:
        manager.close()
    return jax.tree.map(jnp.asarray, restored["model"])


def preproc_from_h5(path: str | Path) -> PreprocessConfig:
    """Reconstruct the ``PreprocessConfig`` a compiled dataset was built with.

    Reads the ``config_json`` attr written by ``generate_vision_enc_data`` and
    rebuilds the exact crop/resize config, so ``ram_to_pixel`` maps state
    coordinates onto the same pixels the VAE saw. Falls back to defaults if the
    attr is absent.
    """
    with h5py.File(str(path), "r") as f:
        raw = f.attrs.get("config_json")
    if raw is None:
        return PreprocessConfig()
    pp = json.loads(raw)["preprocess"]
    return PreprocessConfig(
        crop_top=pp["crop_top"],
        crop_bottom=pp["crop_bottom"],
        crop_left=pp["crop_left"],
        crop_right=pp["crop_right"],
        target_h=pp["target_h"],
        target_w=pp["target_w"],
        resize_filter=pp["resize_filter"],
    )


SOURCE_KEYS: tuple[str, ...] = ("ball_x", "ball_y", "paddle_x", "mode", "score")


def load_frames(
    path: str | Path,
    n: int | None = None,
    start: int = 0,
    *,
    source_keys: typing.Sequence[str] = SOURCE_KEYS,
) -> tuple[np.ndarray, dict[str, np.ndarray], PreprocessConfig]:
    """Load a slice of frames + per-frame source state from a compiled ``.h5``.

    Convenience for notebooks: one call gives you the uint8 frames, a dict of
    the per-frame state arrays needed for the region metrics / stratification,
    and the ``PreprocessConfig`` for coordinate mapping.

    Parameters
    ----------
    path :
        A compiled ``vae_{train,val,test}.h5``.
    n :
        Number of frames to read (``None`` = all from ``start``).
    start :
        First frame index.
    source_keys :
        Which ``source/<key>`` arrays to return.

    Returns
    -------
    (frames, source, config) :
        ``frames`` ``(n, H, W, 3)`` uint8; ``source`` maps each key to an
        ``(n,)`` array; ``config`` the dataset's ``PreprocessConfig``.
    """
    with h5py.File(str(path), "r") as f:
        total = int(f.attrs["n_frames"])
        stop = total if n is None else min(start + n, total)
        frames = f["frames"][start:stop]
        source = {}
        for k in source_keys:
            ds = f.get(f"source/{k}")
            if ds is not None:
                source[k] = np.asarray(ds[start:stop])
    return (np.asarray(frames), source, preproc_from_h5(path))


@functools.partial(jax.jit, static_argnums=())
def _encode_batch(vae: VAE, frames_f32: jnp.ndarray) -> jnp.ndarray:
    """Deterministic encode of one device batch: returns ``mu`` (``z = mu``)."""
    mu, _ = vae.encode(frames_f32)
    return mu


@functools.partial(jax.jit, static_argnums=())
def _decode_batch(vae: VAE, latents: jnp.ndarray) -> jnp.ndarray:
    """Decode one device batch to float32 ``[-1, 1]`` frames."""
    return vae.decode(latents)


def _iter_padded_batches(n: int, batch_size: int) -> typing.Iterator[tuple[int, int]]:
    """Yield ``(start, count)`` spans covering ``[0, n)`` in ``batch_size`` steps."""
    for start in range(0, n, batch_size):
        yield (start, min(batch_size, n - start))


def encode_frames(vae: VAE, frames: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Encode uint8 frames to deterministic latents ``mu``.

    Parameters
    ----------
    vae :
        A loaded ``VAE`` (see :func:`load_vae`).
    frames :
        ``(N, H, W, 3)`` uint8 in ``[0, 255]``.
    batch_size :
        Device batch size. Every batch is padded up to this size so the jitted
        encoder compiles exactly once.

    Returns
    -------
    np.ndarray :
        ``(N, h, w, C)`` float32 latents (``h = H/8``, ``w = W/8``).
    """
    frames = np.asarray(frames)
    n = frames.shape[0]
    out: list[np.ndarray] = []
    for start, count in _iter_padded_batches(n, batch_size):
        chunk = frames[start : start + count]
        if count < batch_size:
            pad = np.repeat(chunk[-1:], batch_size - count, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunk_f32 = chunk.astype(np.float32) / 127.5 - 1.0
        mu = _encode_batch(vae, jnp.asarray(chunk_f32))
        out.append(np.asarray(mu)[:count])
    return np.concatenate(out, axis=0)


def decode_latents(vae: VAE, latents: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Decode latents to uint8 frames in ``[0, 255]``.

    Inverse of :func:`encode_frames`: float32 ``[-1, 1]`` decoder output is
    mapped back to uint8. Accepts ``(N, h, w, C)`` and returns ``(N, H, W, 3)``.
    """
    latents = np.asarray(latents)
    n = latents.shape[0]
    out: list[np.ndarray] = []
    for start, count in _iter_padded_batches(n, batch_size):
        chunk = latents[start : start + count]
        if count < batch_size:
            pad = np.repeat(chunk[-1:], batch_size - count, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        recon = _decode_batch(vae, jnp.asarray(chunk.astype(np.float32)))
        recon_u8 = np.clip((np.asarray(recon) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        out.append(recon_u8[:count])
    return np.concatenate(out, axis=0)


def reconstruct(vae: VAE, frames: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Full round-trip ``decode(encode(frames))`` → uint8 frames.

    Convenience for the evaluation path. Equivalent to
    ``decode_latents(vae, encode_frames(vae, frames))`` but fuses the two batch
    loops so intermediate latents for the whole set are never all held at once.
    """
    frames = np.asarray(frames)
    n = frames.shape[0]
    out: list[np.ndarray] = []
    for start, count in _iter_padded_batches(n, batch_size):
        chunk = frames[start : start + count]
        if count < batch_size:
            pad = np.repeat(chunk[-1:], batch_size - count, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunk_f32 = chunk.astype(np.float32) / 127.5 - 1.0
        mu = _encode_batch(vae, jnp.asarray(chunk_f32))
        recon = _decode_batch(vae, mu)
        recon_u8 = np.clip((np.asarray(recon) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        out.append(recon_u8[:count])
    return np.concatenate(out, axis=0)


def encode_latent_stats(
    vae: VAE, frames: np.ndarray, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mu, log_sigma_sq)`` for frames — used by KL-per-channel diagnostics.

    Unlike :func:`encode_frames` (which returns only ``mu``), this exposes both
    Gaussian parameters so the caller can compute the per-channel KL budget.
    """
    frames = np.asarray(frames)
    n = frames.shape[0]
    mus: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    for start, count in _iter_padded_batches(n, batch_size):
        chunk = frames[start : start + count]
        if count < batch_size:
            pad = np.repeat(chunk[-1:], batch_size - count, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunk_f32 = chunk.astype(np.float32) / 127.5 - 1.0
        mu, logvar = vae.encode(jnp.asarray(chunk_f32))
        mus.append(np.asarray(mu)[:count])
        logvars.append(np.asarray(logvar)[:count])
    return (np.concatenate(mus, axis=0), np.concatenate(logvars, axis=0))


# ============================================================
# toylib_projects.wm.vision_encoder.metrics - /Users/anuj/Desktop/code/toylib/toylib_projects/wm/vision_encoder/metrics.py
# ============================================================

"""Vision-encoder (VAE) specific metrics.

The domain-agnostic ``Metric`` / ``Loss`` / ``VisualizationMetric`` scaffolding
lives in ``toylib_projects.wm.metrics``. This module holds the image metrics
that are specific to the VAE codec:

  - ``ReconstructionVisualization`` — side-by-side input vs. reconstruction,
    implementing the ``Metric`` protocol (runs inside the JIT eval step via aux).
  - ``PriorSamplingVisualization``  — images decoded from random prior latents,
    implementing the ``VisualizationMetric`` protocol (runs outside JIT).

Plus the standalone evaluation metrics used by ``vision_encoder.evaluate`` (and
callable directly from a notebook — plain numpy in / numpy out, no extra deps):

  - Reconstruction fidelity: ``psnr`` / ``ssim`` (and ``*_per_frame`` variants).
  - Physics via known state (no trained detector): ``ball_region_psnr`` /
    ``paddle_region_psnr`` measure reconstruction fidelity in the small patch
    where the RAM state says the ball / paddle is. See ``§8.2`` of
    ``docs/designs/vision_codec.md`` and the coordinate map ``ram_to_pixel`` in
    ``datagen/preprocess_frames.py``.
  - Latent diagnostics: ``kl_per_channel``.
"""


@dataclasses.dataclass
class ReconstructionVisualization:
    """Return input frames and their VAE reconstructions from the eval forward pass.

    Implements the standard ``Metric`` protocol so it runs inside the JIT-compiled
    eval step. The eval ``forward_fn`` must include ``"recon"`` in the returned
    ``aux`` dict (float32, ``[-1, 1]``). Both outputs are converted to uint8
    ``[0, 255]`` before being returned for logging.

    Args:
        recon_aux_key: Key in ``aux`` where the eval forward_fn stores reconstructed
            frames as float32 in ``[-1, 1]``.
        num_images: How many images from the batch to return.
    """

    recon_aux_key: str = "recon"
    num_images: int = 8
    gap_px: int = 4

    def __call__(
        self, loss: float, aux: jt.PyTree, batch: jt.PyTree
    ) -> dict[str, jt.Array]:
        del loss
        inputs = batch[: self.num_images]
        recon_f32 = aux[self.recon_aux_key][: self.num_images]
        recons = ((recon_f32 + 1.0) * 127.5).clip(0, 255).astype(jnp.uint8)
        n, h = inputs.shape[:2]
        gap = jnp.full((n, h, self.gap_px, 3), 128, dtype=jnp.uint8)
        comparison = jnp.concatenate([inputs, gap, recons], axis=2)
        return {"recon_comparison": comparison}


@dataclasses.dataclass
class PriorSamplingVisualization:
    """Log images decoded from randomly sampled latents.

    The PRNG key is derived from ``seed`` and held fixed across evals so
    outputs are directly comparable over the course of training.

    Args:
        sample_fn: ``(model, key, n) -> images`` where ``images`` is
            ``(n, H, W, 3)`` uint8.
        num_samples: Number of images to generate.
        seed: Fixed seed for the sampling key.
    """

    sample_fn: typing.Callable[..., np.ndarray]
    num_samples: int = 16
    seed: int = 42

    def __call__(self, model: typing.Any) -> dict[str, np.ndarray]:
        key = jax.random.key(self.seed)
        samples = self.sample_fn(model, key, self.num_samples)
        return {"prior_samples": np.asarray(samples)}


def _as_float(x: np.ndarray, max_val: float) -> np.ndarray:
    """To float64 in ``[0, 1]`` given the input's ``max_val`` (255 for uint8)."""
    return np.asarray(x, dtype=np.float64) / max_val


def psnr_per_frame(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> np.ndarray:
    """Per-frame PSNR (dB). Inputs ``(N, H, W, C)``; returns ``(N,)``.

    ``+inf`` where a frame reconstructs exactly (``mse == 0``).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2, axis=(1, 2, 3))
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(max_val) - np.log10(mse)


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """Scalar PSNR (dB) over all pixels of one or many frames.

    Accepts a single frame ``(H, W, C)`` or a batch ``(N, H, W, C)``. Computed
    from the *pooled* MSE (not the mean of per-frame PSNRs) so it matches the
    textbook single-image definition when given one frame.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))


def _box2d_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Sliding-window (valid) sum over the trailing two axes.

    ``x`` has shape ``(..., H, W)``; returns ``(..., H-w+1, W-w+1)``. Uses a
    cumulative-sum integral trick, so it's exact and O(HW) regardless of ``w``.
    """
    cs = np.cumsum(x, axis=-2)
    zeros = np.zeros_like(cs[..., :1, :])
    cs = np.concatenate([zeros, cs], axis=-2)
    row = cs[..., w:, :] - cs[..., :-w, :]
    cs = np.cumsum(row, axis=-1)
    zeros = np.zeros_like(cs[..., :, :1])
    cs = np.concatenate([zeros, cs], axis=-1)
    return cs[..., :, w:] - cs[..., :, :-w]


def _ssim_maps(
    x: np.ndarray, y: np.ndarray, window: int, data_range: float
) -> np.ndarray:
    """Per-window SSIM map over the trailing two axes (Wang et al. 2004)."""
    n = window * window
    mux = _box2d_sum(x, window) / n
    muy = _box2d_sum(y, window) / n
    muxx = _box2d_sum(x * x, window) / n
    muyy = _box2d_sum(y * y, window) / n
    muxy = _box2d_sum(x * y, window) / n
    cov_norm = n / (n - 1)
    vx = cov_norm * (muxx - mux * mux)
    vy = cov_norm * (muyy - muy * muy)
    vxy = cov_norm * (muxy - mux * muy)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mux * muy + c1) * (2 * vxy + c2)
    den = (mux * mux + muy * muy + c1) * (vx + vy + c2)
    return num / den


def _prep_ssim(a: np.ndarray, b: np.ndarray, max_val: float):
    """Normalize + reshape to ``(N, C, H, W)`` float in ``[0, 1]``."""
    a = _as_float(a, max_val)
    b = _as_float(b, max_val)
    if a.ndim == 3:
        a, b = (a[None], b[None])
    return (np.moveaxis(a, -1, 1), np.moveaxis(b, -1, 1))


def ssim_per_frame(
    a: np.ndarray, b: np.ndarray, max_val: float = 255.0, window: int = 7
) -> np.ndarray:
    """Per-frame mean SSIM. Inputs ``(N, H, W, C)``; returns ``(N,)``.

    Uniform ``window×window`` filter, unbiased (co)variance, averaged over the
    valid region and channels. Suitable for frames down to ``window`` px.
    """
    a, b = _prep_ssim(a, b, max_val)
    maps = _ssim_maps(a, b, window, data_range=1.0)
    return maps.mean(axis=(1, 2, 3))


def ssim(
    a: np.ndarray, b: np.ndarray, max_val: float = 255.0, window: int = 7
) -> float:
    """Scalar mean SSIM over one frame ``(H, W, C)`` or a batch ``(N, H, W, C)``."""
    return float(np.mean(ssim_per_frame(a, b, max_val=max_val, window=window)))


def _region_slice(cx: float, cy: float, size: int, h: int, w: int):
    """Top-left ``(y0, x0)`` of a ``size×size`` box centered on ``(cx, cy)``.

    The box is clamped to stay fully inside an ``h×w`` frame, so a ball near the
    edge still yields a full-size patch (biased toward the edge, never partial).
    """
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x0 = max(0, min(x0, w - size))
    y0 = max(0, min(y0, h - size))
    return (y0, x0)


def region_psnr(
    input_frame: np.ndarray,
    recon_frame: np.ndarray,
    cx: float,
    cy: float,
    size: int = 16,
    max_val: float = 255.0,
) -> float:
    """PSNR inside a ``size×size`` box centered on ``(cx, cy)`` of one frame.

    Both frames ``(H, W, C)``. ``(cx, cy)`` is a ``(column, row)`` pixel position
    (e.g. from :func:`preprocess_frames.ram_to_pixel`). Measures how faithfully
    the VAE reconstructed the patch that contains a small object — high when the
    object is preserved in place, low when it's blurred away or displaced.
    """
    h, w = input_frame.shape[:2]
    y0, x0 = _region_slice(cx, cy, size, h, w)
    a = input_frame[y0 : y0 + size, x0 : x0 + size]
    b = recon_frame[y0 : y0 + size, x0 : x0 + size]
    return psnr(a, b, max_val=max_val)


def _centers_from_state(
    ram_x: np.ndarray, ram_y: np.ndarray, config: PreprocessConfig
) -> list[typing.Optional[tuple[float, float]]]:
    """Map per-frame ``(ram_x, ram_y)`` state to pixel centers via ``ram_to_pixel``."""
    return [ram_to_pixel(float(x), float(y), config) for x, y in zip(ram_x, ram_y)]


def ball_region_psnr_per_frame(
    inputs: np.ndarray,
    recons: np.ndarray,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    config: PreprocessConfig,
    size: int = 16,
    max_val: float = 255.0,
) -> np.ndarray:
    """Per-frame PSNR in the box around the (known) ball position.

    Uses ``ball_x`` / ``ball_y`` RAM state mapped through ``config`` to locate the
    ball; ``np.nan`` for frames whose ball falls outside the crop (e.g. between
    lives — no ball on screen). ``size`` defaults to 16px, generous enough to
    tolerate small RAM-address / coordinate-mapping error.
    """
    inputs = np.asarray(inputs)
    recons = np.asarray(recons)
    centers = _centers_from_state(ball_x, ball_y, config)
    out = np.full(len(centers), np.nan, dtype=np.float64)
    for i, c in enumerate(centers):
        if c is None:
            continue
        out[i] = region_psnr(
            inputs[i], recons[i], c[0], c[1], size=size, max_val=max_val
        )
    return out


def paddle_region_psnr_per_frame(
    inputs: np.ndarray,
    recons: np.ndarray,
    paddle_x: np.ndarray,
    config: PreprocessConfig,
    size: int = 24,
    paddle_row_frac: float = 0.92,
    max_val: float = 255.0,
) -> np.ndarray:
    """Per-frame PSNR in the box around the (known) paddle position.

    The paddle sits in a fixed horizontal band near the bottom, so only its
    ``x`` is state-dependent: we map ``paddle_x`` to a column and fix the row at
    ``paddle_row_frac`` of the frame height. ``size`` is a little larger than for
    the ball since the paddle is wider. ``np.nan`` if ``paddle_x`` maps outside
    the crop.
    """
    inputs = np.asarray(inputs)
    recons = np.asarray(recons)
    h = inputs.shape[1]
    py = paddle_row_frac * h
    config_row_native = config.crop_top + paddle_row_frac * (
        config.crop_bottom - config.crop_top
    )
    out = np.full(len(paddle_x), np.nan, dtype=np.float64)
    for i, px_ram in enumerate(paddle_x):
        mapped = ram_to_pixel(float(px_ram), config_row_native, config)
        if mapped is None:
            continue
        out[i] = region_psnr(
            inputs[i], recons[i], mapped[0], py, size=size, max_val=max_val
        )
    return out


def kl_per_channel(mu: np.ndarray, log_sigma_sq: np.ndarray) -> np.ndarray:
    """Mean KL budget (nats) per latent channel.

    ``KL_c = 0.5 · mean_{N,h,w}( μ² + σ² − log σ² − 1 )`` for each channel ``c``.
    A healthy KL-VAE spreads budget across channels; a channel near ``0`` is
    "dead" (unused capacity / posterior collapse). Inputs ``(N, h, w, C)``;
    returns ``(C,)``.
    """
    mu = np.asarray(mu, dtype=np.float64)
    log_sigma_sq = np.asarray(log_sigma_sq, dtype=np.float64)
    sigma_sq = np.exp(log_sigma_sq)
    per_elem = 0.5 * (mu**2 + sigma_sq - log_sigma_sq - 1.0)
    return per_elem.mean(axis=(0, 1, 2))


# ============================================================
# None - ../wm/vision_encoder/evaluate.py
# ============================================================

"""Inference-based evaluation of a trained Track A1 KL-VAE on a held-out set.

Loads a frozen VAE checkpoint, reconstructs a large evaluation split
(``vae_test.h5``), and computes the reconstruction + physics metrics from
``docs/designs/vision_codec.md`` §8:

  - **Reconstruction fidelity** — PSNR, SSIM (whole frame).
  - **Physics (region-based)** — ``ball_region_psnr`` / ``paddle_region_psnr``:
    PSNR in the small patch where the stored RAM state says the ball / paddle
    is. No trained detector; robust to blur/displacement (see §8.2 rationale in
    ``metrics.py``).
  - **Latent diagnostics** — KL budget per channel (dead-channel check).
  - **Baselines** — identity / mean-frame / bilinear-8× for context (does the
    codec beat naive compression?).

Everything is reported **overall + stratified by game mode and score bucket**,
and checked against a (128px-scaled, region-reframed) stage gate.

Two entry points, same core:

  - ``evaluate_checkpoint(...)`` — returns a results dict; call it from a
    notebook / Colab.
  - ``main()`` — CLI wrapper that prints tables and writes the JSON.

Usage (CLI, from ``toylib_projects/wm/``)::

uv run python -m vision_encoder.evaluate     --checkpoint-dir=gs://tinystories-checkpoints/wm-vae/20260702T002831/     --checkpoint_step=399000     --test-path=data/compiled/vae_test.h5     --output=eval_report.json

The checkpoint step defaults to the latest; ``--base-ch`` / ``--latent-channels``
must match the trained model.
"""


@dataclasses.dataclass(frozen=True)
class StageGate:
    """Pass/fail thresholds for the codec stage gate.

    The design doc's §8.4 gate is written for 64px frames with pixel-space ball
    position MSE + detection rate. Because we (a) train at 128px and (b) replaced
    the brittle detector with region-reconstruction PSNR, the physics thresholds
    are reframed as minimum region PSNRs. These region PSNR values are
    **heuristic starting points** — calibrate against the identity/bilinear
    baselines the evaluator prints before treating them as hard gates.
    """

    min_ssim: float = 0.85
    min_ball_region_psnr: float = 20.0
    min_paddle_region_psnr: float = 20.0
    min_kl_per_channel: float = 0.01


def bilinear_baseline(frames: np.ndarray, factor: int = 8) -> np.ndarray:
    """Downscale by ``factor`` then upscale back — naive-compression baseline.

    Matches the VAE's 8× spatial compression so the comparison is fair: if the
    codec doesn't beat this, it isn't earning its latent budget. Per-frame PIL
    bilinear resize.
    """
    frames = np.asarray(frames)
    n, h, w, _ = frames.shape
    small = (max(1, w // factor), max(1, h // factor))
    out = np.empty_like(frames)
    for i in range(n):
        img = Image.fromarray(frames[i])
        down = img.resize(small, Image.Resampling.BILINEAR)
        out[i] = np.asarray(down.resize((w, h), Image.Resampling.BILINEAR))
    return out


def mean_frame_baseline(frames: np.ndarray) -> np.ndarray:
    """Every reconstruction = the dataset mean frame — the variance floor."""
    frames = np.asarray(frames)
    mean = frames.mean(axis=0, keepdims=True)
    return np.broadcast_to(mean, frames.shape).astype(np.uint8)


def compute_per_frame_metrics(
    inputs: np.ndarray,
    recons: np.ndarray,
    source: dict[str, np.ndarray],
    config: PreprocessConfig,
    *,
    ball_region_size: int = 16,
    paddle_region_size: int = 24,
    ssim_window: int = 7,
) -> dict[str, np.ndarray]:
    """Per-frame metric arrays (``(N,)`` each) for a set of reconstructions.

    Region metrics are only produced when the required state keys are present in
    ``source`` (``ball_x``/``ball_y`` for the ball, ``paddle_x`` for the paddle).
    """
    out: dict[str, np.ndarray] = {
        "psnr": psnr_per_frame(inputs, recons),
        "ssim": ssim_per_frame(inputs, recons, window=ssim_window),
    }
    if "ball_x" in source and "ball_y" in source:
        out["ball_region_psnr"] = ball_region_psnr_per_frame(
            inputs,
            recons,
            source["ball_x"],
            source["ball_y"],
            config,
            size=ball_region_size,
        )
    if "paddle_x" in source:
        out["paddle_region_psnr"] = paddle_region_psnr_per_frame(
            inputs, recons, source["paddle_x"], config, size=paddle_region_size
        )
    return out


def _finite_mean(x: np.ndarray) -> float:
    """Mean over finite entries only (drops ``inf`` from exact frames, ``nan``
    from region metrics with no object). Returns ``nan`` if nothing is finite."""
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(x)
    return float(x[mask].mean()) if mask.any() else float("nan")


def _aggregate(per_frame: dict[str, np.ndarray]) -> dict[str, float]:
    """Overall finite-mean of each per-frame metric."""
    return {k: _finite_mean(v) for k, v in per_frame.items()}


def _stratify(
    per_frame: dict[str, np.ndarray],
    keys: np.ndarray,
    key_labels: dict[typing.Any, str] | None = None,
) -> pd.DataFrame:
    """Finite-mean of each metric grouped by an integer stratum key.

    Returns a DataFrame indexed by stratum label with one column per metric plus
    an ``n`` count column.
    """
    rows = []
    for k in np.unique(keys):
        sel = keys == k
        label = key_labels.get(k, str(k)) if key_labels else str(k)
        row = {"stratum": label, "n": int(sel.sum())}
        for name, vals in per_frame.items():
            row[name] = _finite_mean(vals[sel])
        rows.append(row)
    return pd.DataFrame(rows).set_index("stratum")


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    test_path: str | Path,
    step: int | None = None,
    *,
    base_ch: int = 64,
    latent_channels: int = 4,
    batch_size: int = 64,
    max_frames: int | None = None,
    ball_region_size: int = 16,
    paddle_region_size: int = 24,
    score_buckets: tuple[int, ...] = DEFAULT_SCORE_BUCKETS,
    include_baselines: bool = True,
    baseline_max_frames: int = 2000,
    stage_gate: StageGate = StageGate(),
) -> dict[str, typing.Any]:
    """Evaluate a VAE checkpoint on a compiled test split.

    Loads frames + state, reconstructs, and computes overall / per-mode /
    per-score-bucket metrics, latent KL diagnostics, baselines, and a stage-gate
    verdict. Returns a nested, JSON-serializable results dict (also the return
    value used by :func:`main`).

    Notebook-friendly: everything needed (frames, recons, per-frame arrays) is
    also returned under ``"arrays"`` so you can plot or drill in further.
    """
    frames, source, config = load_frames(test_path, n=max_frames)
    vae = load_vae(
        checkpoint_dir, step, base_ch=base_ch, latent_channels=latent_channels
    )
    recons = reconstruct(vae, frames, batch_size=batch_size)
    per_frame = compute_per_frame_metrics(
        frames,
        recons,
        source,
        config,
        ball_region_size=ball_region_size,
        paddle_region_size=paddle_region_size,
    )
    overall = _aggregate(per_frame)
    mu, log_sigma_sq = encode_latent_stats(vae, frames, batch_size=batch_size)
    kl_c = kl_per_channel(mu, log_sigma_sq)
    overall["kl_per_channel_min"] = float(kl_c.min())
    overall["kl_per_channel_mean"] = float(kl_c.mean())
    by_mode = None
    by_score = None
    if "mode" in source:
        by_mode = _stratify(per_frame, np.asarray(source["mode"]))
    if "score" in source:
        buckets = np.array(
            [_score_bucket(int(s), score_buckets) for s in source["score"]]
        )
        edges = [0, *score_buckets]
        labels = {
            i: f"[{edges[i]},{edges[i + 1]})"
            if i < len(score_buckets)
            else f">={score_buckets[-1]}"
            for i in range(len(score_buckets) + 1)
        }
        by_score = _stratify(per_frame, buckets, labels)
    baselines: dict[str, dict[str, float]] = {}
    if include_baselines:
        m = min(len(frames), baseline_max_frames)
        fb, rb = (frames[:m], recons[:m])
        sb = {k: v[:m] for k, v in source.items()}
        baseline_recons = {
            "identity": fb,
            "mean_frame": mean_frame_baseline(fb),
            "bilinear_8x": bilinear_baseline(fb, factor=8),
            "vae": rb,
        }
        for name, rec in baseline_recons.items():
            baselines[name] = _aggregate(
                compute_per_frame_metrics(
                    fb,
                    rec,
                    sb,
                    config,
                    ball_region_size=ball_region_size,
                    paddle_region_size=paddle_region_size,
                )
            )
    gate = _check_stage_gate(overall, stage_gate)
    results: dict[str, typing.Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "step": step if step is not None else latest_step(checkpoint_dir),
        "test_path": str(test_path),
        "n_frames": int(len(frames)),
        "config": dataclasses.asdict(config),
        "overall": overall,
        "kl_per_channel": kl_c.tolist(),
        "by_mode": None if by_mode is None else by_mode.to_dict(orient="index"),
        "by_score_bucket": None
        if by_score is None
        else by_score.to_dict(orient="index"),
        "baselines": baselines,
        "stage_gate": gate,
        "arrays": {
            "frames": frames,
            "recons": recons,
            "per_frame": per_frame,
            "kl_per_channel": kl_c,
        },
    }
    return results


def _check_stage_gate(
    overall: dict[str, float], gate: StageGate
) -> dict[str, typing.Any]:
    """Compare overall metrics to the gate; return per-criterion + overall pass."""
    checks = {
        "ssim": (overall.get("ssim", float("nan")), gate.min_ssim),
        "ball_region_psnr": (
            overall.get("ball_region_psnr", float("nan")),
            gate.min_ball_region_psnr,
        ),
        "paddle_region_psnr": (
            overall.get("paddle_region_psnr", float("nan")),
            gate.min_paddle_region_psnr,
        ),
        "kl_per_channel_min": (
            overall.get("kl_per_channel_min", float("nan")),
            gate.min_kl_per_channel,
        ),
    }
    detail = {}
    all_pass = True
    for name, (value, threshold) in checks.items():
        passed = bool(np.isfinite(value) and value >= threshold)
        detail[name] = {"value": value, "threshold": threshold, "pass": passed}
        all_pass = all_pass and passed
    detail["all_pass"] = all_pass
    return detail


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "  n/a" if np.isnan(v) else "  inf"
    return f"{v:7.3f}"


def print_report(results: dict[str, typing.Any]) -> None:
    """Human-readable dump of an ``evaluate_checkpoint`` result."""
    print("=" * 64)
    print(f"VAE evaluation — step {results['step']} — {results['n_frames']:,} frames")
    print(f"  checkpoint: {results['checkpoint_dir']}")
    print(f"  test set:   {results['test_path']}")
    print("=" * 64)
    print("\nOverall metrics:")
    for k, v in results["overall"].items():
        print(f"  {k:<24} {_fmt(v)}")
    if results["baselines"]:
        print("\nBaselines (subsample) — PSNR / SSIM / ball / paddle region PSNR:")
        cols = ["psnr", "ssim", "ball_region_psnr", "paddle_region_psnr"]
        header = "  {:<14}".format("recon") + "".join((f"{c:>20}" for c in cols))
        print(header)
        for name, m in results["baselines"].items():
            row = "  {:<14}".format(name) + "".join(
                (f"{_fmt(m.get(c, float('nan'))):>20}" for c in cols)
            )
            print(row)
    if results["by_mode"]:
        print("\nBy game mode:")
        print(
            pd.DataFrame.from_dict(results["by_mode"], orient="index")
            .round(3)
            .to_string()
        )
    if results["by_score_bucket"]:
        print("\nBy score bucket:")
        print(
            pd.DataFrame.from_dict(results["by_score_bucket"], orient="index")
            .round(3)
            .to_string()
        )
    print("\nStage gate:")
    for name, d in results["stage_gate"].items():
        if name == "all_pass":
            continue
        mark = "PASS" if d["pass"] else "FAIL"
        print(f"  [{mark}] {name:<22} {_fmt(d['value'])}  (>= {d['threshold']})")
    verdict = "PASS" if results["stage_gate"]["all_pass"] else "FAIL"
    print(f"\n  OVERALL STAGE GATE: {verdict}")
    print("=" * 64)


def _json_safe(results: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Drop the heavy ``arrays`` and coerce numpy → python for JSON dumping."""
    out = {k: v for k, v in results.items() if k != "arrays"}
    return json.loads(
        json.dumps(
            out, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)
        )
    )
