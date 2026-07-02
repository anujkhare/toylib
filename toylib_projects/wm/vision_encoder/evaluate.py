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

uv run python -m vision_encoder.evaluate \
    --checkpoint-dir=gs://tinystories-checkpoints/wm-vae/20260702T002831/ \
    --checkpoint_step=399000 \
    --test-path=data/compiled/vae_test.h5 \
    --output=eval_report.json

The checkpoint step defaults to the latest; ``--base-ch`` / ``--latent-channels``
must match the trained model.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import typing
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from toylib_projects.wm.datagen import preprocess_frames as pp_lib
from toylib_projects.wm.datagen.generate_vision_enc_data import (
    DEFAULT_SCORE_BUCKETS,
    _score_bucket,
)
from toylib_projects.wm.vision_encoder import inference as inference_lib
from toylib_projects.wm.vision_encoder import metrics as metrics_lib


# ──────────────────────────────────────────────────────────────────────────
# Stage-gate thresholds
# ──────────────────────────────────────────────────────────────────────────


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
    min_kl_per_channel: float = 0.01  # "no dead channels"


# ──────────────────────────────────────────────────────────────────────────
# Baseline reconstructions (context for the real numbers)
# ──────────────────────────────────────────────────────────────────────────


def bilinear_baseline(frames: np.ndarray, factor: int = 8) -> np.ndarray:
    """Downscale by ``factor`` then upscale back — naive-compression baseline.

    Matches the VAE's 8× spatial compression so the comparison is fair: if the
    codec doesn't beat this, it isn't earning its latent budget. Per-frame PIL
    bilinear resize.
    """
    frames = np.asarray(frames)
    n, h, w, _ = frames.shape
    small = (max(1, w // factor), max(1, h // factor))  # PIL (W, H)
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


# ──────────────────────────────────────────────────────────────────────────
# Core metric computation
# ──────────────────────────────────────────────────────────────────────────


def compute_per_frame_metrics(
    inputs: np.ndarray,
    recons: np.ndarray,
    source: dict[str, np.ndarray],
    config: pp_lib.PreprocessConfig,
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
        "psnr": metrics_lib.psnr_per_frame(inputs, recons),
        "ssim": metrics_lib.ssim_per_frame(inputs, recons, window=ssim_window),
    }
    if "ball_x" in source and "ball_y" in source:
        out["ball_region_psnr"] = metrics_lib.ball_region_psnr_per_frame(
            inputs,
            recons,
            source["ball_x"],
            source["ball_y"],
            config,
            size=ball_region_size,
        )
    if "paddle_x" in source:
        out["paddle_region_psnr"] = metrics_lib.paddle_region_psnr_per_frame(
            inputs,
            recons,
            source["paddle_x"],
            config,
            size=paddle_region_size,
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


# ──────────────────────────────────────────────────────────────────────────
# Top-level evaluation
# ──────────────────────────────────────────────────────────────────────────


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
    # ── Load data + model, reconstruct ──────────────────────────────────
    frames, source, config = inference_lib.load_frames(test_path, n=max_frames)
    vae = inference_lib.load_vae(
        checkpoint_dir, step, base_ch=base_ch, latent_channels=latent_channels
    )
    recons = inference_lib.reconstruct(vae, frames, batch_size=batch_size)

    # ── Core metrics ────────────────────────────────────────────────────
    per_frame = compute_per_frame_metrics(
        frames,
        recons,
        source,
        config,
        ball_region_size=ball_region_size,
        paddle_region_size=paddle_region_size,
    )
    overall = _aggregate(per_frame)

    # ── Latent KL diagnostics ───────────────────────────────────────────
    mu, log_sigma_sq = inference_lib.encode_latent_stats(
        vae, frames, batch_size=batch_size
    )
    kl_c = metrics_lib.kl_per_channel(mu, log_sigma_sq)
    overall["kl_per_channel_min"] = float(kl_c.min())
    overall["kl_per_channel_mean"] = float(kl_c.mean())

    # ── Stratification ──────────────────────────────────────────────────
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
            i: (
                f"[{edges[i]},{edges[i + 1]})"
                if i < len(score_buckets)
                else f">={score_buckets[-1]}"
            )
            for i in range(len(score_buckets) + 1)
        }
        by_score = _stratify(per_frame, buckets, labels)

    # ── Baselines ───────────────────────────────────────────────────────
    baselines: dict[str, dict[str, float]] = {}
    if include_baselines:
        m = min(len(frames), baseline_max_frames)
        fb, rb = frames[:m], recons[:m]
        sb = {k: v[:m] for k, v in source.items()}
        baseline_recons = {
            "identity": fb,
            "mean_frame": mean_frame_baseline(fb),
            "bilinear_8x": bilinear_baseline(fb, factor=8),
            "vae": rb,  # VAE on the same subsample, for apples-to-apples
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

    # ── Stage-gate verdict ──────────────────────────────────────────────
    gate = _check_stage_gate(overall, stage_gate)

    results: dict[str, typing.Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "step": step if step is not None else inference_lib.latest_step(checkpoint_dir),
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
        # Heavy arrays for notebook drill-down (not written to the JSON report).
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


# ──────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────


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
        header = "  {:<14}".format("recon") + "".join(f"{c:>20}" for c in cols)
        print(header)
        for name, m in results["baselines"].items():
            row = "  {:<14}".format(name) + "".join(
                f"{_fmt(m.get(c, float('nan'))):>20}" for c in cols
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


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def main() -> dict[str, typing.Any]:
    # Accept both hyphen and underscore spellings for every multi-word flag
    # (e.g. --checkpoint-step and --checkpoint_step) so either convention works.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", required=True)
    parser.add_argument("--checkpoint-step", "--checkpoint_step", type=int, default=None)
    parser.add_argument("--test-path", "--test_path", type=Path, required=True)
    parser.add_argument("--base-ch", "--base_ch", type=int, default=64)
    parser.add_argument("--latent-channels", "--latent_channels", type=int, default=4)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=64)
    parser.add_argument(
        "--max-frames",
        "--max_frames",
        type=int,
        default=None,
        help="Cap frames evaluated (default: whole test set).",
    )
    parser.add_argument("--ball-region-size", "--ball_region_size", type=int, default=16)
    parser.add_argument("--paddle-region-size", "--paddle_region_size", type=int, default=24)
    parser.add_argument("--no-baselines", "--no_baselines", action="store_true")
    parser.add_argument("--baseline-max-frames", "--baseline_max_frames", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON report here (heavy arrays excluded).",
    )
    args = parser.parse_args()

    results = evaluate_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        test_path=args.test_path,
        step=args.checkpoint_step,
        base_ch=args.base_ch,
        latent_channels=args.latent_channels,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        ball_region_size=args.ball_region_size,
        paddle_region_size=args.paddle_region_size,
        include_baselines=not args.no_baselines,
        baseline_max_frames=args.baseline_max_frames,
    )
    print_report(results)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(_json_safe(results), f, indent=2)
        print(f"\nWrote JSON report → {args.output}")
    return results


if __name__ == "__main__":
    main()
