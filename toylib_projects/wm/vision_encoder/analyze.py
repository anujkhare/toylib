"""Reporting helpers for VAE training: parameter counts, dataset sizing, and
capacity analysis.

Mirrors the shape of ``toylib_projects/tinystories/analyze.py`` so the train
script has the same ergonomics: build the experiment, init state, then call
``print_*`` helpers to dump a human-readable summary to stdout (and to the
WandB run config via the standard logger plumbing).

What's different from tinystories
---------------------------------

**No Chinchilla.** Chinchilla's "~20 tokens per parameter" rule is calibrated
to next-token cross-entropy on natural language. For an image VAE trained
with per-pixel reconstruction loss, the relevant scaling unit is
**pixel-supervisions** (``samples × H × W × C``), not tokens, and the
rule-of-thumb sweet spot is much looser (autoencoders converge well below
the Chinchilla ratio because each sample provides H*W*C parallel scalar
supervisions). So we report capacity in pixel-supervisions per parameter and
flag whether we land in the typical "well-fed" range (100–1,000 supervisions
per param across the whole run).

**Walkthrough milestones.** The Track A1 walkthrough lists three concrete
training targets (Milestones 4 / 5 / 6). We surface them next to the
current ``max_steps`` so you can see at a glance whether the configured run
matches a milestone.
"""

from __future__ import annotations

import math
import typing

import jax
import jaxtyping as jt
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────
# Parameter-tree introspection
# ──────────────────────────────────────────────────────────────────────────


def get_tree_stats(model: jt.PyTree) -> pd.DataFrame:
    """One row per array leaf in the model pytree, with ``params``, ``n_bytes``,
    ``dtype``, ``path``, and ``level_<i>`` columns for grouping at any depth."""
    results = []
    leaf_stats = [
        (k, v.shape, v.dtype) for k, v in jax.tree_util.tree_leaves_with_path(model)
    ]

    for path, shape, dtype in leaf_stats:
        path = [str(p) for p in path]
        count = math.prod(shape)
        nbytes = count * dtype.itemsize
        row = {
            "params": count,
            "n_bytes": nbytes,
            "dtype": str(dtype),
            "path": "/".join(path),
        }
        for i, p in enumerate(path):
            row[f"level_{i}"] = p
        results.append(row)
    return pd.DataFrame(results)


def print_param_sizes(
    model: jt.PyTree, depth: int = 1, size_denom: int = 1
) -> tuple[pd.DataFrame, int, int]:
    """Print and return a grouped param-count / byte table.

    Same surface as ``tinystories.analyze.print_param_sizes`` so train scripts
    can use it interchangeably. ``depth`` controls how many path components
    to group by (depth=1 = encoder vs decoder; depth=2 = per sub-block).
    """
    df_stats = get_tree_stats(model)
    if len(df_stats) == 0:
        print("Model has no parameters.")
        return pd.DataFrame(), 0, 0
    df_stats.loc[:, "n_bytes_divided"] = df_stats["n_bytes"] / size_denom
    total_params = int(df_stats["params"].sum())
    total_bytes = float(df_stats["n_bytes_divided"].sum())
    print(f"Total Parameters: {total_params:,}. Bytes: ({total_bytes:,.2f})")
    level_cols = [f"level_{i}" for i in range(depth)]
    grouped = (
        df_stats.fillna("")
        .groupby(level_cols + ["dtype"])
        .sum()[["params", "n_bytes_divided"]]
        .reset_index()
    )
    return grouped, total_params, total_bytes


# ──────────────────────────────────────────────────────────────────────────
# Dataset / training stats
# ──────────────────────────────────────────────────────────────────────────


def print_dataset_stats(dataset, name: str = "train") -> dict[str, int]:
    """Print shape, count, and pixel volume for a ``Hdf5FramesDataset``."""
    H, W, C = dataset.frame_shape
    n_frames = dataset.num_frames
    pixels_per_frame = H * W * C
    total_pixels = n_frames * pixels_per_frame
    print("------------------------------")
    print(f"{name.capitalize()} dataset:")
    print("------------------------------")
    print(f"  Frame shape:        {H}x{W}x{C} ({pixels_per_frame:,} pixels/frame)")
    print(f"  Frames:             {n_frames:,}")
    print(f"  Batch size:         {dataset.batch_size:,}")
    print(f"  Batches per epoch:  {len(dataset):,}")
    print(f"  Total pixels:       {total_pixels:,}")
    print(f"  Raw uint8 bytes:    {total_pixels:,} ({total_pixels / 1e9:.2f} GB)")
    return {
        "n_frames": n_frames,
        "pixels_per_frame": pixels_per_frame,
        "total_pixels": total_pixels,
        "batches_per_epoch": len(dataset),
    }


def print_training_estimate(exp) -> dict[str, float]:
    """Total samples / epochs / pixel-supervisions for the configured run."""
    ds = exp.train_task.dataset
    max_steps = exp.training_config.max_steps
    batch_size = ds.batch_size
    pixels_per_frame = math.prod(ds.frame_shape)
    n_frames = ds.num_frames

    samples_seen = max_steps * batch_size
    epochs = samples_seen / max(n_frames, 1)
    pixel_supervisions = samples_seen * pixels_per_frame

    print("------------------------------")
    print("Training estimate:")
    print("------------------------------")
    print(f"  Max steps:                 {max_steps:,}")
    print(f"  Microbatches per step:     {exp.training_config.num_microbatches:,}")
    print(f"  Samples per step:          {batch_size:,}")
    print(f"  Total samples seen:        {samples_seen:,}")
    print(f"  Equivalent epochs:         {epochs:.2f}")
    print(f"  Total pixel-supervisions:  {pixel_supervisions:,.0f}")
    return {
        "max_steps": max_steps,
        "samples_seen": samples_seen,
        "epochs": epochs,
        "pixel_supervisions": pixel_supervisions,
    }


def print_capacity_analysis(model: jt.PyTree, exp) -> dict[str, float]:
    """Pixel-supervisions per parameter — the VAE analogue of the Chinchilla ratio.

    For an autoencoder, each training sample provides ``H*W*C`` independent
    scalar supervisions (one per pixel), not a single token. So the
    cross-entropy-trained Chinchilla rule (~20 tokens/param) doesn't apply
    directly. Empirical sweet spot for ImageNet-scale VAEs: roughly
    ``100–1,000 pixel-supervisions per parameter`` across the full run.
    Below 100 you'll typically still be improving; above ~10k you're
    starting to memorize.
    """
    ds = exp.train_task.dataset
    n_params = sum(int(math.prod(t.shape)) for t in jax.tree.leaves(model))
    samples_seen = exp.training_config.max_steps * ds.batch_size
    pixels_per_sample = math.prod(ds.frame_shape)
    pixel_supervisions = samples_seen * pixels_per_sample
    ratio = pixel_supervisions / max(n_params, 1)

    print("------------------------------")
    print("Capacity analysis (VAE):")
    print("------------------------------")
    print(f"  Model parameters:                {n_params:,}")
    print(f"  Pixel-supervisions over run:     {pixel_supervisions:,.0f}")
    print(f"  Pixel-supervisions per param:    {ratio:,.1f}")
    if ratio < 30:
        verdict = "UNDER-trained — increase max_steps or data; expect poor reconstruction"
    elif ratio < 100:
        verdict = "below typical sweet spot (100–1000); may not converge fully"
    elif ratio <= 10_000:
        verdict = "in the typical well-trained range for image VAEs"
    else:
        verdict = "well above typical — likely memorizing; KL term should still regularize"
    print(f"  Verdict:                         {verdict}")
    return {
        "n_params": n_params,
        "pixel_supervisions": pixel_supervisions,
        "supervisions_per_param": ratio,
    }


def print_walkthrough_milestones(exp) -> None:
    """Compare ``max_steps`` against the Track A1 walkthrough milestones."""
    max_steps = exp.training_config.max_steps
    milestones = [
        ("M4 overfit test (300 steps)",   300,    "L_rec < 0.05 on 8 fixed frames"),
        ("M5 + perceptual + aux (+1k)",  1_300,   "extends M4 with VGG + ball MLP"),
        ("M6 full A1 run (50k–100k)",    75_000,  "stage-gate eval (SSIM > 0.85, ball detect > 95%)"),
    ]
    print("------------------------------")
    print(f"Walkthrough milestones (your max_steps = {max_steps:,}):")
    print("------------------------------")
    closest = min(milestones, key=lambda m: abs(m[1] - max_steps))
    for label, target, goal in milestones:
        marker = "→" if (label, target, goal) == closest else " "
        print(f"  {marker} {label:<35} target {target:>7,} steps  ({goal})")


# ──────────────────────────────────────────────────────────────────────────
# One-call wrapper
# ──────────────────────────────────────────────────────────────────────────


def print_all(exp, *, depth: int = 2) -> None:
    """Convenience: param sizes + dataset stats + training estimate + capacity."""
    grouped, total_params, _ = print_param_sizes(exp.model, depth=depth)
    print()
    print(grouped.to_string(index=False))
    print()
    print_dataset_stats(exp.train_task.dataset, name="train")
    if exp.eval_task is not None:
        print()
        print_dataset_stats(exp.eval_task.dataset, name="val")
    print()
    print_training_estimate(exp)
    print()
    print_capacity_analysis(exp.model, exp)
    print()
    print_walkthrough_milestones(exp)
