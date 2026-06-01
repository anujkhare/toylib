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
uv run python -m datagen.generate_vision_enc_data \
    --target-size 128 \
    --n-train 200000 --n-val 20000 --n-test 20000 \
    --input-fps 6 \
    --base-seed 0

# Side-by-side run on a smoke dataset
uv run python -m datagen.generate_vision_enc_data \
    --input-root /tmp/wm_smoke \
    --output-root /tmp/wm_vae_smoke
```
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, NamedTuple

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm

from .preprocess_frames import PreprocessConfig, preprocess_frames

# Stage 1 emulator runs at 60Hz native. Override only if that ever changes.
NATIVE_FPS = 60

# Default score-bucket boundaries (half-open lower edges; +inf last).
# Picked to roughly partition pristine / early / mid / late / post-clear.
DEFAULT_SCORE_BUCKETS = (0, 100, 500, 2000)

# Default split fractions (episode-level).
DEFAULT_TRAIN_FRAC = 0.80
DEFAULT_VAL_FRAC = 0.10
# test_frac = 1.0 - train - val


class FrameSample(NamedTuple):
    """One selected source-frame, fully addressable into the raw shard tree."""

    shard_idx: int  # index into the discovered shard list
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


# ────────────────────────────────────────────────────────────────────────────
# Discovery / split assignment
# ────────────────────────────────────────────────────────────────────────────


def _find_shards(root: Path) -> list[Path]:
    """All `episodes_shard_*.h5` under root (recursive)."""
    return sorted(root.rglob("episodes_shard_*.h5"))


def _assign_split(
    episode_key: tuple[int, str],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> str:
    """Deterministic train/val/test from a stable per-episode key."""
    # Use a seeded RNG seeded by hash of (seed, shard_idx, episode_name).
    h = (seed, episode_key[0], episode_key[1])
    rng = np.random.default_rng(abs(hash(h)) % (2**32))
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


# ────────────────────────────────────────────────────────────────────────────
# Candidate enumeration
# ────────────────────────────────────────────────────────────────────────────


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

    # Read the small per-frame state arrays once; frames are read later in bulk.
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
                    shard_idx,
                    ep_name,
                    f[ep_name],
                    stride,
                    score_buckets,
                ):
                    out[split].append(cand)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Stratified sub-sampling
# ────────────────────────────────────────────────────────────────────────────


def _stratified_sample(
    candidates: list[FrameSample],
    n_target: int,
    rng: np.random.Generator,
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
        strata[(c.mode, c.difficulty, c.score_bucket)].append(c)

    n_strata = len(strata)
    per_stratum = n_target // n_strata
    selected: list[FrameSample] = []
    remainders: list[list[FrameSample]] = []

    for stratum_cands in strata.values():
        if len(stratum_cands) <= per_stratum:
            selected.extend(stratum_cands)
            # No leftovers to top up from this stratum.
            continue
        idx = rng.choice(len(stratum_cands), size=per_stratum, replace=False)
        picked = [stratum_cands[i] for i in idx]
        selected.extend(picked)
        # Leftovers we may draw from to top up.
        picked_set = set(int(i) for i in idx)
        remainders.append(
            [c for i, c in enumerate(stratum_cands) if i not in picked_set]
        )

    # Top up to n_target if we're short (because some strata were small).
    if len(selected) < n_target and remainders:
        pool = [c for r in remainders for c in r]
        if pool:
            need = n_target - len(selected)
            top_up_n = min(need, len(pool))
            idx = rng.choice(len(pool), size=top_up_n, replace=False)
            selected.extend(pool[i] for i in idx)

    rng.shuffle(selected)
    return selected[:n_target]


# ────────────────────────────────────────────────────────────────────────────
# Materialization (read raw frames, preprocess, write to output)
# ────────────────────────────────────────────────────────────────────────────


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
    h, w = preproc.target_h, preproc.target_w
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

        # Group samples by (shard, episode) so we touch each HDF5 group at most
        # once, which is much faster than per-frame opens.
        by_episode: dict[tuple[int, str], list[tuple[int, FrameSample]]] = defaultdict(
            list
        )
        for out_i, s in enumerate(samples):
            by_episode[(s.shard_idx, s.episode_name)].append((out_i, s))

        # Pre-build a stable mapping from episode_name -> per-shard episode index.
        shard_ep_index: dict[tuple[int, str], int] = {}
        with_ep_idx_cache: dict[int, list[str]] = {}

        for (shard_idx, ep_name), pairs in tqdm(
            by_episode.items(), desc=f"materializing {output_path.name}"
        ):
            shard_path = shards[shard_idx]
            with h5py.File(shard_path, "r") as f:
                # Cache the per-shard episode-name → index map (used for `episode_idx`
                # bookkeeping; faster than re-listing keys every loop).
                if shard_idx not in with_ep_idx_cache:
                    with_ep_idx_cache[shard_idx] = sorted(f.keys())
                ep_names = with_ep_idx_cache[shard_idx]

                grp = f[ep_name]
                # Read just the frame indices we need (fancy indexing on axis 0).
                wanted_idx = np.array([s.frame_idx for _, s in pairs], dtype=np.int64)
                # h5py requires sorted indices for fancy indexing; sort + invert.
                order = np.argsort(wanted_idx)
                sorted_idx = wanted_idx[order]
                raw_frames = grp["frames"][sorted_idx.tolist()]
                # Un-sort back to the original pair order.
                inverse = np.argsort(order)
                raw_frames = raw_frames[inverse]

                # Preprocess in a single Python loop (a few thousand frames at a time).
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

        # Flush per-frame source arrays in one go.
        for k, arr in src_arrays.items():
            src.create_dataset(k, data=arr)


# ────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ────────────────────────────────────────────────────────────────────────────


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
            f"No episodes_shard_*.h5 found under {input_root}. "
            "Run Stage 1 generation first (datagen.run_stage1)."
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
        shards,
        stride,
        score_buckets,
        base_seed,
        train_frac,
        val_frac,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile a vision-encoder training dataset from Stage 1 raw shards.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/raw"),
        help="Root containing mode_MM_diff_D/episodes_shard_*.h5 (default: data/raw).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/compiled"),
        help="Output directory for vae_{train,val,test}.h5 (default: data/compiled).",
    )
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=10_000)
    parser.add_argument("--n-test", type=int, default=10_000)
    parser.add_argument(
        "--input-fps",
        type=int,
        default=NATIVE_FPS,
        help=(
            f"Effective sampling frame-rate (Hz) over Stage 1 raw. Default "
            f"{NATIVE_FPS} = use every raw frame. Must divide {NATIVE_FPS} evenly. "
            f"Lower values down-sample temporally (e.g. 6 = every 10th frame)."
        ),
    )
    parser.add_argument(
        "--score-buckets",
        type=int,
        nargs="+",
        default=list(DEFAULT_SCORE_BUCKETS),
        help=(
            "Half-open lower bounds for score strata used in sampling. "
            f"Default {list(DEFAULT_SCORE_BUCKETS)} gives 5 buckets "
            "(pristine / early / mid / late / post-clear)."
        ),
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=128,
        help="Output frame height/width (default 128 → 128×128).",
    )
    parser.add_argument("--crop-top", type=int, default=32)
    parser.add_argument("--crop-bottom", type=int, default=192)
    parser.add_argument("--crop-left", type=int, default=0)
    parser.add_argument("--crop-right", type=int, default=160)
    parser.add_argument(
        "--resize-filter",
        default="lanczos",
        choices=["lanczos", "bilinear", "bicubic", "nearest", "box"],
    )
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--base-seed", type=int, default=0)

    args = parser.parse_args()

    preproc = PreprocessConfig(
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        crop_left=args.crop_left,
        crop_right=args.crop_right,
        target_h=args.target_size,
        target_w=args.target_size,
        resize_filter=args.resize_filter,
    )
    preproc.validate()

    compile_vision_enc_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        input_fps=args.input_fps,
        score_buckets=tuple(args.score_buckets),
        preproc=preproc,
        base_seed=args.base_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
