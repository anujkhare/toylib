"""Tests for preprocess_frames and the vision-encoder compiler."""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pytest

from .generate_vision_enc_data import (
    DEFAULT_SCORE_BUCKETS,
    _score_bucket,
    _stratified_sample,
    compile_vision_enc_dataset,
)
from .generate_vision_enc_data import FrameSample
from .preprocess_frames import (
    NATIVE_H,
    NATIVE_W,
    PreprocessConfig,
    preprocess_frame,
    preprocess_frames,
)


# ────────────────────────────────────────────────────────────────────────────
# preprocess_frames
# ────────────────────────────────────────────────────────────────────────────


def test_default_config_round_trip() -> None:
    raw = np.zeros((NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
    out = preprocess_frame(raw)
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8


def test_custom_target_size() -> None:
    raw = np.random.default_rng(0).integers(0, 256, (NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
    cfg = PreprocessConfig(target_h=64, target_w=64)
    out = preprocess_frame(raw, cfg)
    assert out.shape == (64, 64, 3)


def test_batch_preserves_shape() -> None:
    raw = np.zeros((7, NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
    out = preprocess_frames(raw)
    assert out.shape == (7, 128, 128, 3)


def test_no_op_when_dims_match() -> None:
    """If the cropped shape already equals target, the resize is skipped (but image is still correct)."""
    cfg = PreprocessConfig(
        crop_top=32, crop_bottom=192, crop_left=0, crop_right=160,
        target_h=160, target_w=160,
    )
    raw = np.arange(NATIVE_H * NATIVE_W * 3, dtype=np.uint8).reshape(NATIVE_H, NATIVE_W, 3)
    out = preprocess_frame(raw, cfg)
    assert out.shape == (160, 160, 3)
    # Cropped region must be byte-identical to the corresponding slice of raw.
    expected = raw[32:192, 0:160, :]
    np.testing.assert_array_equal(out, expected)


def test_crops_remove_scoreboard() -> None:
    """The default crop must drop the top scoreboard rows."""
    cfg = PreprocessConfig()
    # Synthetic frame: row index encoded in red channel. After default crop
    # (rows 32..192) and resize back, the *original row 0* color must be absent.
    raw = np.zeros((NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
    for r in range(NATIVE_H):
        raw[r, :, 0] = r
    out = preprocess_frame(raw, cfg)
    # Resize is LANCZOS so some color leakage is possible at the boundary,
    # but red values <32 should not dominate.
    assert (out[..., 0] >= 32).mean() > 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crop_top": -1},
        {"crop_top": 100, "crop_bottom": 50},
        {"crop_right": NATIVE_W + 10},
        {"target_h": 0},
        {"resize_filter": "bogus"},
    ],
)
def test_invalid_configs_raise(kwargs: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        cfg = PreprocessConfig(**kwargs)
        cfg.validate()


# ────────────────────────────────────────────────────────────────────────────
# Sampler internals
# ────────────────────────────────────────────────────────────────────────────


def test_score_bucket_boundaries() -> None:
    boundaries = (0, 100, 500, 2000)
    # score < 0     → 0
    # score == 0    → 1 (since boundaries[0] == 0, score < 0 is bucket 0; score == 0 falls into bucket where score < 100, i.e. 1)
    # score == 100  → 2
    # score == 5000 → 4 (last)
    assert _score_bucket(-1, boundaries) == 0
    assert _score_bucket(0, boundaries) == 1
    assert _score_bucket(99, boundaries) == 1
    assert _score_bucket(100, boundaries) == 2
    assert _score_bucket(499, boundaries) == 2
    assert _score_bucket(500, boundaries) == 3
    assert _score_bucket(5000, boundaries) == 4


def _fake_candidate(mode: int, diff: int, score: int, frame_idx: int) -> FrameSample:
    return FrameSample(
        shard_idx=0,
        episode_name="episode_000000",
        frame_idx=frame_idx,
        mode=mode,
        difficulty=diff,
        paddle_x=0.0, ball_x=0.0, ball_y=0.0, score=score, lives=5,
        score_bucket=_score_bucket(score, DEFAULT_SCORE_BUCKETS),
    )


def test_stratified_sample_covers_all_strata() -> None:
    """Every (mode, diff, score_bucket) stratum that has candidates must contribute."""
    rng = np.random.default_rng(0)
    candidates = []
    # 2 modes × 2 diffs × 3 score buckets, 20 candidates per stratum.
    for m in (0, 8):
        for d in (0, 1):
            for s in (50, 300, 1500):  # buckets 1, 2, 3
                for t in range(20):
                    candidates.append(_fake_candidate(m, d, s, t))

    out = _stratified_sample(candidates, n_target=48, rng=rng)
    assert len(out) == 48
    strata_hit = {(c.mode, c.difficulty, c.score_bucket) for c in out}
    # 2 × 2 × 3 = 12 strata
    assert len(strata_hit) == 12, strata_hit


def test_stratified_sample_clips_to_available() -> None:
    """Asking for more than exists returns everything we have."""
    rng = np.random.default_rng(0)
    candidates = [_fake_candidate(0, 0, 50, t) for t in range(5)]
    out = _stratified_sample(candidates, n_target=100, rng=rng)
    assert len(out) == 5


# ────────────────────────────────────────────────────────────────────────────
# End-to-end compile
# ────────────────────────────────────────────────────────────────────────────


def _write_fake_raw_shard(
    path: Path,
    mode: int,
    difficulty: int,
    n_episodes: int = 12,
    length: int = 30,
) -> Path:
    """Tiny synthetic Stage 1 shard with the schema generate_vision_enc_data expects."""
    rng = np.random.default_rng((mode + 1) * 7 + difficulty)
    with h5py.File(path, "w") as f:
        for ep_idx in range(n_episodes):
            grp = f.create_group(f"episode_{ep_idx:06d}")
            grp.attrs["length"] = np.int32(length)
            grp.attrs["mode"] = np.int32(mode)
            grp.attrs["difficulty"] = np.int32(difficulty)
            frames = rng.integers(0, 256, (length, NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
            grp.create_dataset("frames", data=frames, chunks=(1, NATIVE_H, NATIVE_W, 3), **hdf5plugin.LZ4())
            grp.create_dataset("actions", data=np.zeros(length, dtype=np.int32))
            states = grp.create_group("states")
            # Make score climb so we get multiple score buckets per episode.
            states.create_dataset("score", data=(np.arange(length) * 50).astype(np.int32))
            states.create_dataset("lives", data=np.full(length, 5, dtype=np.int32))
            states.create_dataset("bricks_remaining", data=np.zeros(length, dtype=np.int32))
            states.create_dataset("paddle_x", data=np.linspace(60, 180, length, dtype=np.float32))
            states.create_dataset("ball_x", data=np.linspace(0, 160, length, dtype=np.float32))
            states.create_dataset("ball_y", data=np.linspace(0, 200, length, dtype=np.float32))
    return path


def test_compile_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for mode in (0, 8):
        for diff in (0, 1):
            d = raw / f"mode_{mode:02d}_diff_{diff}"
            d.mkdir(parents=True)
            _write_fake_raw_shard(d / "episodes_shard_0000.h5", mode=mode, difficulty=diff)

    out_root = tmp_path / "compiled"
    preproc = PreprocessConfig(target_h=32, target_w=32)  # tiny for speed

    written = compile_vision_enc_dataset(
        input_root=raw,
        output_root=out_root,
        n_train=20,
        n_val=4,
        n_test=4,
        input_fps=60,
        score_buckets=DEFAULT_SCORE_BUCKETS,
        preproc=preproc,
        base_seed=0,
        train_frac=0.80,
        val_frac=0.10,
    )

    assert set(written.keys()) == {"train", "val", "test"}
    # With 4 shards × 12 episodes = 48 episodes and an 80/10/10 split, every
    # split should contain at least one episode.
    for split, p in written.items():
        assert p.exists(), f"{split}: expected output at {p} but it wasn't written"
        with h5py.File(p, "r") as f:
            n = int(f.attrs["n_frames"])
            assert n > 0, f"{split}: empty"
            assert f["frames"].shape == (n, 32, 32, 3)
            assert f["frames"].dtype == np.uint8
            for k in ("shard_idx", "episode_idx", "frame_idx", "mode", "difficulty",
                      "paddle_x", "ball_x", "ball_y", "score", "lives"):
                assert f[f"source/{k}"].shape == (n,), f"{split}/{k}"
            # Provenance attrs.
            assert "config_json" in f.attrs
            assert "created_utc" in f.attrs
