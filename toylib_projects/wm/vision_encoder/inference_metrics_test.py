"""Tests for the Plan-A evaluation stack (no checkpoint required).

Covers the three pieces that make up the Colab-callable inference/metrics path:

  - ``datagen.preprocess_frames.ram_to_pixel`` — the state→pixel coordinate map
    the region metrics rely on.
  - ``vision_encoder.metrics`` — psnr / ssim / region-psnr / kl-per-channel,
    exercised with synthetic frames where the answer is known analytically.
  - ``vision_encoder.inference`` — encode/decode/reconstruct batching contracts,
    exercised with a tiny randomly-initialised VAE (no trained checkpoint).

Everything here runs on synthetic data so it needs neither a checkpoint nor a
compiled ``.h5`` dataset.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from toylib_projects.wm.datagen.preprocess_frames import (
    PreprocessConfig,
    ram_to_pixel,
)
from toylib_projects.wm.vision_encoder import inference as inf
from toylib_projects.wm.vision_encoder import metrics
from toylib_projects.wm.vision_encoder import model as model_lib


# ──────────────────────────────────────────────────────────────────────────
# ram_to_pixel
# ──────────────────────────────────────────────────────────────────────────


class TestRamToPixel:
    def test_top_left_of_crop_maps_to_origin(self) -> None:
        cfg = PreprocessConfig()  # crop [32,192)x[0,160) → 128x128
        px, py = ram_to_pixel(cfg.crop_left, cfg.crop_top, cfg)
        assert px == pytest.approx(0.0)
        assert py == pytest.approx(0.0)

    def test_scales_into_target_grid(self) -> None:
        cfg = PreprocessConfig()
        # A native point halfway down/across the crop lands at the grid centre.
        mid_col = cfg.crop_left + (cfg.crop_right - cfg.crop_left) / 2
        mid_row = cfg.crop_top + (cfg.crop_bottom - cfg.crop_top) / 2
        px, py = ram_to_pixel(mid_col, mid_row, cfg)
        assert px == pytest.approx(cfg.target_w / 2)
        assert py == pytest.approx(cfg.target_h / 2)

    def test_out_of_crop_returns_none(self) -> None:
        cfg = PreprocessConfig()
        # A row inside the cropped-away scoreboard strip (< crop_top).
        assert ram_to_pixel(80, 0, cfg) is None
        # A column past the right edge of the crop.
        assert ram_to_pixel(cfg.crop_right + 5, 100, cfg) is None

    def test_no_resize_is_identity_shift(self) -> None:
        # crop only, no resize: target == crop size → pure translation.
        cfg = PreprocessConfig(
            crop_top=10, crop_bottom=110, crop_left=20, crop_right=120,
            target_h=100, target_w=100,
        )
        px, py = ram_to_pixel(50, 60, cfg)
        assert px == pytest.approx(30.0)  # 50 - 20
        assert py == pytest.approx(50.0)  # 60 - 10


# ──────────────────────────────────────────────────────────────────────────
# PSNR / SSIM
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


class TestPsnr:
    def test_identity_is_inf(self, rng) -> None:
        a = rng.integers(0, 256, size=(4, 16, 16, 3), dtype=np.uint8)
        assert metrics.psnr(a, a) == float("inf")
        assert np.all(np.isinf(metrics.psnr_per_frame(a, a)))

    def test_degrades_with_noise(self, rng) -> None:
        a = rng.integers(0, 256, size=(4, 16, 16, 3), dtype=np.uint8)
        small = np.clip(a.astype(np.int16) + 5, 0, 255).astype(np.uint8)
        large = np.clip(a.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        assert metrics.psnr(a, small) > metrics.psnr(a, large)

    def test_known_value(self) -> None:
        # Constant offset of d over all pixels → mse = d^2 → known PSNR.
        a = np.full((1, 8, 8, 3), 100, dtype=np.uint8)
        b = np.full((1, 8, 8, 3), 110, dtype=np.uint8)
        expected = 10.0 * np.log10(255.0**2 / 100.0)
        assert metrics.psnr(a, b) == pytest.approx(expected)


class TestSsim:
    def test_identity_is_one(self, rng) -> None:
        a = rng.integers(0, 256, size=(3, 32, 32, 3), dtype=np.uint8)
        assert metrics.ssim(a, a) == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(metrics.ssim_per_frame(a, a), 1.0, atol=1e-6)

    def test_bounded_and_degrades(self, rng) -> None:
        a = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
        noisy = np.clip(
            a.astype(np.int16) + rng.integers(-60, 60, size=a.shape), 0, 255
        ).astype(np.uint8)
        s = metrics.ssim(a, noisy)
        assert -1.0 <= s <= 1.0
        assert s < metrics.ssim(a, a)

    def test_per_frame_shape(self, rng) -> None:
        a = rng.integers(0, 256, size=(5, 20, 20, 3), dtype=np.uint8)
        b = rng.integers(0, 256, size=(5, 20, 20, 3), dtype=np.uint8)
        assert metrics.ssim_per_frame(a, b).shape == (5,)


# ──────────────────────────────────────────────────────────────────────────
# Region metrics
# ──────────────────────────────────────────────────────────────────────────


def _frame_with_dot(h, w, cx, cy, val=255) -> np.ndarray:
    """Black frame with a small bright 3x3 dot centred at (cx, cy)."""
    f = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0 = int(cy) - 1, int(cx) - 1
    f[y0 : y0 + 3, x0 : x0 + 3] = val
    return f


class TestRegionPsnr:
    def test_perfect_recon_is_inf(self) -> None:
        f = _frame_with_dot(64, 64, 30, 30)
        assert metrics.region_psnr(f, f, 30, 30, size=16) == float("inf")

    def test_missing_object_scores_low(self) -> None:
        inp = _frame_with_dot(64, 64, 30, 30)
        recon_perfect = inp.copy()
        recon_missing = np.zeros_like(inp)  # object blurred away
        p_perfect = metrics.region_psnr(inp, recon_perfect, 30, 30, size=16)
        p_missing = metrics.region_psnr(inp, recon_missing, 30, 30, size=16)
        assert p_perfect > p_missing
        assert np.isfinite(p_missing)

    def test_ball_region_nan_when_out_of_crop(self) -> None:
        cfg = PreprocessConfig()
        inputs = np.zeros((2, 128, 128, 3), dtype=np.uint8)
        recons = np.zeros((2, 128, 128, 3), dtype=np.uint8)
        # frame 0 ball in play; frame 1 ball off-screen (row above crop_top).
        ball_x = np.array([80.0, 80.0])
        ball_y = np.array([100.0, 0.0])
        out = metrics.ball_region_psnr_per_frame(
            inputs, recons, ball_x, ball_y, cfg, size=16
        )
        assert out.shape == (2,)
        assert np.isfinite(out[0]) or np.isinf(out[0])  # in-crop → a value
        assert np.isnan(out[1])  # off-screen → nan

    def test_paddle_region_runs(self) -> None:
        cfg = PreprocessConfig()
        inputs = np.zeros((2, 128, 128, 3), dtype=np.uint8)
        recons = inputs.copy()
        paddle_x = np.array([70.0, 90.0])
        out = metrics.paddle_region_psnr_per_frame(
            inputs, recons, paddle_x, cfg, size=24
        )
        assert out.shape == (2,)
        # identical frames → +inf inside the region.
        assert np.all(np.isinf(out))


# ──────────────────────────────────────────────────────────────────────────
# KL per channel
# ──────────────────────────────────────────────────────────────────────────


class TestKlPerChannel:
    def test_zero_at_prior(self) -> None:
        # mu = 0, log_sigma_sq = 0 (sigma^2 = 1) → KL = 0 exactly.
        mu = np.zeros((4, 4, 4, 3))
        lv = np.zeros((4, 4, 4, 3))
        kl = metrics.kl_per_channel(mu, lv)
        assert kl.shape == (3,)
        assert np.allclose(kl, 0.0)

    def test_positive_away_from_prior(self) -> None:
        mu = np.ones((2, 4, 4, 3))
        lv = np.zeros((2, 4, 4, 3))
        kl = metrics.kl_per_channel(mu, lv)
        # KL = 0.5*(1 + 1 - 0 - 1) = 0.5 per channel.
        assert np.allclose(kl, 0.5)


# ──────────────────────────────────────────────────────────────────────────
# Inference batching (tiny randomly-initialised VAE — no checkpoint)
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_vae() -> model_lib.VAE:
    cfg = model_lib.ModelConfig(base_ch=32, latent_channels=2)
    vae = model_lib.VAE(config=cfg, key=jax.random.key(0))
    vae.init()
    return vae


@pytest.fixture
def tiny_frames() -> np.ndarray:
    rng = np.random.default_rng(1)
    # 32x32 input → 4x4 latent (8x compression). 10 frames exercises padding
    # of the final short batch when batch_size doesn't divide N.
    return rng.integers(0, 256, size=(10, 32, 32, 3), dtype=np.uint8)


class TestInferenceRoundTrip:
    def test_encode_shape_and_dtype(self, tiny_vae, tiny_frames) -> None:
        z = inf.encode_frames(tiny_vae, tiny_frames, batch_size=4)
        assert z.shape == (10, 4, 4, 2)
        assert z.dtype == np.float32

    def test_decode_shape_and_dtype(self, tiny_vae, tiny_frames) -> None:
        z = inf.encode_frames(tiny_vae, tiny_frames, batch_size=4)
        rec = inf.decode_latents(tiny_vae, z, batch_size=4)
        assert rec.shape == tiny_frames.shape
        assert rec.dtype == np.uint8

    def test_reconstruct_matches_decode_of_encode(self, tiny_vae, tiny_frames) -> None:
        fused = inf.reconstruct(tiny_vae, tiny_frames, batch_size=4)
        staged = inf.decode_latents(
            tiny_vae, inf.encode_frames(tiny_vae, tiny_frames, batch_size=4),
            batch_size=4,
        )
        np.testing.assert_array_equal(fused, staged)

    def test_batch_size_invariance(self, tiny_vae, tiny_frames) -> None:
        # Padding the short final batch must not change results vs. a size that
        # divides N evenly.
        z_a = inf.encode_frames(tiny_vae, tiny_frames, batch_size=4)  # 4,4,2
        z_b = inf.encode_frames(tiny_vae, tiny_frames, batch_size=5)  # divides 10
        np.testing.assert_allclose(z_a, z_b, atol=1e-5)

    def test_encode_is_deterministic(self, tiny_vae, tiny_frames) -> None:
        z1 = inf.encode_frames(tiny_vae, tiny_frames, batch_size=4)
        z2 = inf.encode_frames(tiny_vae, tiny_frames, batch_size=4)
        np.testing.assert_array_equal(z1, z2)  # z = mu, no sampling noise

    def test_latent_stats_shapes(self, tiny_vae, tiny_frames) -> None:
        mu, log_sigma_sq = inf.encode_latent_stats(
            tiny_vae, tiny_frames, batch_size=4
        )
        assert mu.shape == (10, 4, 4, 2)
        assert log_sigma_sq.shape == (10, 4, 4, 2)
        # mu from encode_latent_stats matches encode_frames.
        z = inf.encode_frames(tiny_vae, tiny_frames, batch_size=4)
        np.testing.assert_allclose(mu, z, atol=1e-5)
