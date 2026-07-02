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

import dataclasses
import typing

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np

from toylib_projects.wm.datagen import preprocess_frames as pp_lib


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
        self,
        loss: float,
        aux: jt.PyTree,
        batch: jt.PyTree,
    ) -> dict[str, jt.Array]:
        del loss
        inputs = batch[: self.num_images]  # (N, H, W, 3) uint8
        recon_f32 = aux[self.recon_aux_key][: self.num_images]  # float32 [-1, 1]
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

    def __call__(
        self,
        model: typing.Any,
    ) -> dict[str, np.ndarray]:
        key = jax.random.key(self.seed)
        samples = self.sample_fn(model, key, self.num_samples)
        return {"prior_samples": np.asarray(samples)}


# ══════════════════════════════════════════════════════════════════════════
# Standalone evaluation metrics (numpy — usable directly from a notebook)
# ══════════════════════════════════════════════════════════════════════════
#
# These operate on plain uint8 frames (``[0, 255]``) or float frames with an
# explicit ``max_val``. Per-frame variants return one value per frame so the
# evaluator can stratify by mode / score bucket cheaply; the scalar wrappers
# just average.


def _as_float(x: np.ndarray, max_val: float) -> np.ndarray:
    """To float64 in ``[0, 1]`` given the input's ``max_val`` (255 for uint8)."""
    return np.asarray(x, dtype=np.float64) / max_val


# ──────────────────────────────────────────────────────────────────────────
# PSNR
# ──────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────
# SSIM (pure-numpy, uniform window via integral images — no scipy/skimage)
# ──────────────────────────────────────────────────────────────────────────


def _box2d_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Sliding-window (valid) sum over the trailing two axes.

    ``x`` has shape ``(..., H, W)``; returns ``(..., H-w+1, W-w+1)``. Uses a
    cumulative-sum integral trick, so it's exact and O(HW) regardless of ``w``.
    """
    # Cumulative sum with a leading zero, along the row axis, then window-diff.
    cs = np.cumsum(x, axis=-2)
    zeros = np.zeros_like(cs[..., :1, :])
    cs = np.concatenate([zeros, cs], axis=-2)
    row = cs[..., w:, :] - cs[..., :-w, :]
    # Same along the column axis.
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
    # Unbiased variance/covariance (matches skimage's cov_norm = N/(N-1)).
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
    if a.ndim == 3:  # single frame → add batch axis
        a, b = a[None], b[None]
    # (N, H, W, C) → (N, C, H, W) so the box filter runs over (H, W).
    return np.moveaxis(a, -1, 1), np.moveaxis(b, -1, 1)


def ssim_per_frame(
    a: np.ndarray,
    b: np.ndarray,
    max_val: float = 255.0,
    window: int = 7,
) -> np.ndarray:
    """Per-frame mean SSIM. Inputs ``(N, H, W, C)``; returns ``(N,)``.

    Uniform ``window×window`` filter, unbiased (co)variance, averaged over the
    valid region and channels. Suitable for frames down to ``window`` px.
    """
    a, b = _prep_ssim(a, b, max_val)
    maps = _ssim_maps(a, b, window, data_range=1.0)  # (N, C, h', w')
    return maps.mean(axis=(1, 2, 3))


def ssim(
    a: np.ndarray,
    b: np.ndarray,
    max_val: float = 255.0,
    window: int = 7,
) -> float:
    """Scalar mean SSIM over one frame ``(H, W, C)`` or a batch ``(N, H, W, C)``."""
    return float(np.mean(ssim_per_frame(a, b, max_val=max_val, window=window)))


# ──────────────────────────────────────────────────────────────────────────
# Region-based physics metrics (use known RAM state, no detector)
# ──────────────────────────────────────────────────────────────────────────


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
    return y0, x0


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
    ram_x: np.ndarray,
    ram_y: np.ndarray,
    config: pp_lib.PreprocessConfig,
) -> list[typing.Optional[tuple[float, float]]]:
    """Map per-frame ``(ram_x, ram_y)`` state to pixel centers via ``ram_to_pixel``."""
    return [
        pp_lib.ram_to_pixel(float(x), float(y), config) for x, y in zip(ram_x, ram_y)
    ]


def ball_region_psnr_per_frame(
    inputs: np.ndarray,
    recons: np.ndarray,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    config: pp_lib.PreprocessConfig,
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
    config: pp_lib.PreprocessConfig,
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
    # ram_to_pixel needs a row too; pass the paddle row so it isn't rejected as
    # out-of-crop. We only use the returned column.
    config_row_native = config.crop_top + (
        paddle_row_frac * (config.crop_bottom - config.crop_top)
    )
    out = np.full(len(paddle_x), np.nan, dtype=np.float64)
    for i, px_ram in enumerate(paddle_x):
        mapped = pp_lib.ram_to_pixel(float(px_ram), config_row_native, config)
        if mapped is None:
            continue
        out[i] = region_psnr(
            inputs[i], recons[i], mapped[0], py, size=size, max_val=max_val
        )
    return out


# ──────────────────────────────────────────────────────────────────────────
# Latent diagnostics
# ──────────────────────────────────────────────────────────────────────────


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
