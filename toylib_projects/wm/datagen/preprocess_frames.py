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

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

# Native ALE/Breakout frame dimensions; expected input shape from Stage 1.
NATIVE_H = 210
NATIVE_W = 160

# Pillow resize-filter enum is positional/integer in older versions, so we map
# friendly names through `Image.Resampling` (Pillow ≥ 9.1).
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
    crop_bottom: int = 192   # exclusive — keeps rows 32..191 = 160 rows
    crop_left: int = 0
    crop_right: int = 160    # exclusive — keeps full width
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
        if not (0 <= self.crop_top < self.crop_bottom <= NATIVE_H):
            raise ValueError(
                f"crop_top/crop_bottom must satisfy 0 <= top < bottom <= {NATIVE_H}, "
                f"got [{self.crop_top}, {self.crop_bottom})"
            )
        if not (0 <= self.crop_left < self.crop_right <= NATIVE_W):
            raise ValueError(
                f"crop_left/crop_right must satisfy 0 <= left < right <= {NATIVE_W}, "
                f"got [{self.crop_left}, {self.crop_right})"
            )
        if self.target_h <= 0 or self.target_w <= 0:
            raise ValueError(f"target_h/target_w must be positive, got {self.target_h}×{self.target_w}")
        if self.resize_filter not in _RESIZE_FILTERS:
            raise ValueError(
                f"resize_filter must be one of {list(_RESIZE_FILTERS.keys())}, "
                f"got {self.resize_filter!r}"
            )


def preprocess_frame(frame: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
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
        config.crop_top : config.crop_bottom,
        config.crop_left : config.crop_right,
        :,
    ]
    img = Image.fromarray(cropped)
    if (config.target_h, config.target_w) != cropped.shape[:2]:
        img = img.resize(
            (config.target_w, config.target_h),  # PIL uses (W, H)
            _RESIZE_FILTERS[config.resize_filter],
        )
    return np.asarray(img, dtype=np.uint8)


def preprocess_frames(frames: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
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
