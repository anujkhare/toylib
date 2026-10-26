"""Visualization utilities for the vision-encoder (VAE) codec.

Companion to ``vision_encoder.metrics``: where that module *scores*
reconstructions, this one *shows* them. Everything here is plain numpy in /
numpy (or matplotlib ``Figure``) out — usable directly from a notebook / Colab,
no ``Experiment`` / ``Task`` scaffolding.

Two layers:

  - **Rendering-free helpers** (numpy only, no matplotlib import): build an
    input-vs-reconstruction comparison strip, extract the fixed-size patch around
    a known object (the ball), draw location boxes onto frames. These are cheap to
    unit-test and have no plotting dependency.
  - **Plotting helpers** (lazy ``matplotlib`` import): grid views of frames /
    reconstructions and of the ball patches.

Why a dedicated patch extractor
-------------------------------

Cropping the small patch around the ball is deceptively easy to get wrong. The
common bug is to zero-pad the frame so edge patches stay in bounds, then forget
that padding shifts every pixel: an object at column ``x`` now lives at column
``x + pad`` in the padded array, so gather indices must be shifted too. Miss that
and every patch is offset by ``pad`` pixels (and, worse, near-edge coordinates
silently wrap through negative indices). :func:`extract_patches` centers each
patch correctly and never wraps.

Coordinate mapping is delegated to :func:`preprocess_frames.ram_to_pixel` — the
single source of truth also used by the region metrics — so the patch a viz shows
is exactly the patch ``ball_region_psnr`` scored.

Conventions
-----------

- Frames are ``(N, H, W, 3)`` (a batch) or ``(H, W, 3)`` (single). uint8 in
  ``[0, 255]`` is the native form; float inputs are accepted and normalized (see
  :func:`to_uint8`).
- Pixel coordinates are ``(x=column, y=row)`` in the preprocessed frame grid,
  matching ``ram_to_pixel``'s return convention.
"""

from __future__ import annotations

import typing

import numpy as np

from toylib_projects.wm.datagen import preprocess_frames as pp_lib

if typing.TYPE_CHECKING:  # avoid importing matplotlib at module load
    import matplotlib.figure


# ══════════════════════════════════════════════════════════════════════════
# Image normalization
# ══════════════════════════════════════════════════════════════════════════


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Coerce an image / batch to uint8 ``[0, 255]``.

    Accepts, and auto-detects:
      - uint8 already → returned unchanged.
      - float in ``[-1, 1]`` (the VAE's native range, min < 0) → ``(x + 1) * 127.5``.
      - float in ``[0, 1]`` → ``x * 255``.

    Anything else is clipped into ``[0, 255]`` as-is. Always returns a fresh
    uint8 array.
    """
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    x = x.astype(np.float64)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if lo < 0.0:  # assume [-1, 1]
        x = (x + 1.0) * 127.5
    elif hi <= 1.0:  # assume [0, 1]
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════
# Coordinate mapping (state → preprocessed-frame pixels)
# ══════════════════════════════════════════════════════════════════════════


def state_pixel_centers(
    ram_x: np.ndarray,
    ram_y: np.ndarray,
    config: pp_lib.PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map per-frame RAM ``(x, y)`` state to preprocessed-frame pixel centers.

    Thin vectorized wrapper over :func:`preprocess_frames.ram_to_pixel` (looped
    for exactness — ``N`` is small for viz). Frames whose object falls outside the
    crop (e.g. between-lives frames with no ball) get ``nan`` centers and
    ``valid=False``.

    Parameters
    ----------
    ram_x, ram_y :
        ``(N,)`` native-frame column / row (e.g. ``ball_x`` / ``ball_y``).
    config :
        The dataset's ``PreprocessConfig`` (from ``inference.preproc_from_h5`` or
        ``load_frames``).

    Returns
    -------
    (px, py, valid) :
        ``px`` / ``py`` are ``(N,)`` float column / row in the preprocessed grid
        (``nan`` where out-of-crop); ``valid`` is a ``(N,)`` bool mask.
    """
    ram_x = np.asarray(ram_x)
    ram_y = np.asarray(ram_y)
    px = np.full(ram_x.shape, np.nan, dtype=np.float64)
    py = np.full(ram_x.shape, np.nan, dtype=np.float64)
    valid = np.zeros(ram_x.shape, dtype=bool)
    for i, (x, y) in enumerate(zip(ram_x.ravel(), ram_y.ravel())):
        mapped = pp_lib.ram_to_pixel(float(x), float(y), config)
        if mapped is not None:
            px.flat[i], py.flat[i] = mapped
            valid.flat[i] = True
    return px, py, valid


# ══════════════════════════════════════════════════════════════════════════
# Patch extraction (correctly centered, no wraparound)
# ══════════════════════════════════════════════════════════════════════════


def extract_patches(
    frames: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    size: int = 32,
    fill: int = 0,
) -> np.ndarray:
    """Crop a ``size×size`` patch centered on ``(cx, cy)`` from each frame.

    Padding-aware: frames are zero-padded (``fill``) by ``size // 2`` on each side
    so that a patch centered near an edge is filled with ``fill`` rather than
    wrapping around. The gather indices are shifted by the same pad amount, so the
    patch is centered exactly on the (rounded) ``(cx, cy)`` — this is the step the
    naive version forgets.

    Parameters
    ----------
    frames :
        ``(N, H, W, C)`` (or ``(H, W, C)`` for a single frame).
    cx, cy :
        ``(N,)`` pixel column / row of each patch center (e.g. from
        :func:`state_pixel_centers`). Rounded to the nearest pixel. ``nan`` centers
        (out-of-crop) are treated as ``(0, 0)`` and yield an all-``fill`` patch —
        pair with the ``valid`` mask from ``state_pixel_centers`` to skip them.
    size :
        Patch side length in pixels.
    fill :
        Constant value used for out-of-frame regions.

    Returns
    -------
    np.ndarray :
        ``(N, size, size, C)`` (same dtype as ``frames``); ``(size, size, C)`` if a
        single frame was passed.
    """
    frames = np.asarray(frames)
    single = frames.ndim == 3
    if single:
        frames = frames[None]
    if frames.ndim != 4:
        raise ValueError(f"frames must be (N, H, W, C) or (H, W, C), got {frames.shape}")

    n = frames.shape[0]
    cx = np.broadcast_to(np.asarray(cx, dtype=np.float64), (n,))
    cy = np.broadcast_to(np.asarray(cy, dtype=np.float64), (n,))

    pad = size // 2
    padded = np.pad(
        frames,
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=fill,
    )

    # nan → 0 so rounding/int-cast is safe; those rows land in the pad border and
    # come back as `fill` (and the caller has the validity mask anyway).
    cxi = np.nan_to_num(cx, nan=0.0)
    cyi = np.nan_to_num(cy, nan=0.0)
    # Clamp so the (padded) window is always fully in bounds.
    cxi = np.clip(np.round(cxi), 0, frames.shape[2] - 1).astype(np.int64)
    cyi = np.clip(np.round(cyi), 0, frames.shape[1] - 1).astype(np.int64)

    # padded[c : c+size] == original[c-pad : c+size-pad], i.e. centered on c.
    win = np.arange(size)
    ind_y = cyi[:, None, None] + win[None, :, None]  # (N, size, 1)
    ind_x = cxi[:, None, None] + win[None, None, :]  # (N, 1, size)
    ind_n = np.arange(n)[:, None, None]

    patches = padded[ind_n, ind_y, ind_x, :]  # (N, size, size, C)
    return patches[0] if single else patches


def ball_patches(
    inputs: np.ndarray,
    recons: np.ndarray,
    source: dict[str, np.ndarray],
    config: pp_lib.PreprocessConfig,
    size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patches around the (known) ball, cropped identically from inputs and recons.

    Convenience wrapper: maps ``source['ball_x'] / ['ball_y']`` to pixel centers
    and extracts matching patches from both stacks.

    Returns
    -------
    (input_patches, recon_patches, valid) :
        Two ``(N, size, size, C)`` stacks plus the ``(N,)`` bool mask of frames
        that actually had a ball in the crop.
    """
    px, py, valid = state_pixel_centers(source["ball_x"], source["ball_y"], config)
    return (
        extract_patches(inputs, px, py, size=size),
        extract_patches(recons, px, py, size=size),
        valid,
    )


# ══════════════════════════════════════════════════════════════════════════
# Overlays and strips (numpy — no matplotlib)
# ══════════════════════════════════════════════════════════════════════════


def draw_boxes(
    frames: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    size: int = 32,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Return a copy of ``frames`` with a hollow ``size×size`` box drawn at each center.

    Useful to show *where* a patch was cropped from. ``nan`` centers are skipped.
    ``frames`` is coerced to uint8 first.
    """
    out = to_uint8(frames).copy()
    single = out.ndim == 3
    if single:
        out = out[None]
    n, h, w = out.shape[:3]
    cx = np.broadcast_to(np.asarray(cx, dtype=np.float64), (n,))
    cy = np.broadcast_to(np.asarray(cy, dtype=np.float64), (n,))
    half = size // 2
    col = np.asarray(color, dtype=np.uint8)
    for i in range(n):
        if not (np.isfinite(cx[i]) and np.isfinite(cy[i])):
            continue
        x0 = int(round(cx[i])) - half
        y0 = int(round(cy[i])) - half
        x1, y1 = x0 + size, y0 + size
        xs0, xs1 = max(0, x0), min(w, x1)
        ys0, ys1 = max(0, y0), min(h, y1)
        if xs0 >= xs1 or ys0 >= ys1:
            continue
        t = thickness
        # top / bottom edges (only where the edge is inside the frame)
        if 0 <= y0 < h:
            out[i, y0 : y0 + t, xs0:xs1] = col
        if 0 <= y1 - t < h:
            out[i, y1 - t : y1, xs0:xs1] = col
        # left / right edges
        if 0 <= x0 < w:
            out[i, ys0:ys1, x0 : x0 + t] = col
        if 0 <= x1 - t < w:
            out[i, ys0:ys1, x1 - t : x1] = col
    return out[0] if single else out


def comparison_strip(
    inputs: np.ndarray,
    recons: np.ndarray,
    gap_px: int = 4,
    gap_value: int = 128,
) -> np.ndarray:
    """Stack input | gap | reconstruction horizontally per frame.

    Matplotlib-free (mirrors ``metrics.ReconstructionVisualization`` but as plain
    numpy). Both inputs coerced to uint8.

    Parameters
    ----------
    inputs, recons :
        ``(N, H, W, 3)`` (or single ``(H, W, 3)``).

    Returns
    -------
    np.ndarray :
        ``(N, H, 2*W + gap_px, 3)`` uint8 (or single ``(H, 2*W+gap_px, 3)``).
    """
    a = to_uint8(inputs)
    b = to_uint8(recons)
    single = a.ndim == 3
    if single:
        a, b = a[None], b[None]
    if a.shape != b.shape:
        raise ValueError(f"inputs {a.shape} and recons {b.shape} must match")
    n, h = a.shape[:2]
    gap = np.full((n, h, gap_px, 3), gap_value, dtype=np.uint8)
    out = np.concatenate([a, gap, b], axis=2)
    return out[0] if single else out


# ══════════════════════════════════════════════════════════════════════════
# Plotting (lazy matplotlib import)
# ══════════════════════════════════════════════════════════════════════════


def _subplots(nrows: int, ncols: int, scale: float, row_h: float):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(scale * ncols, row_h * nrows), squeeze=False
    )
    return fig, axes


def show_reconstructions(
    inputs: np.ndarray,
    recons: np.ndarray,
    n: int = 8,
    start: int = 0,
    *,
    titles: typing.Sequence[str] | None = None,
    scale: float = 2.0,
    row_labels: tuple[str, str] = ("input", "recon"),
) -> "matplotlib.figure.Figure":
    """Grid: top row = inputs, bottom row = reconstructions.

    Parameters
    ----------
    inputs, recons :
        ``(N, H, W, 3)`` frames (uint8 or float — coerced via :func:`to_uint8`).
    n :
        Number of columns (frames) to show, starting at ``start``.
    titles :
        Optional per-column titles (indexed by column, ``len >= n``). Defaults to
        the frame index ``#idx``.
    scale :
        Width in inches per column.
    row_labels :
        y-axis labels for the two rows.

    Returns
    -------
    matplotlib Figure (already laid out; call ``plt.show()`` or display it).
    """
    a = to_uint8(inputs)
    b = to_uint8(recons)
    n = int(min(n, len(a) - start))
    if n <= 0:
        raise ValueError(f"nothing to show: start={start}, len={len(a)}")

    fig, axes = _subplots(2, n, scale, row_h=2.25)
    for i in range(n):
        idx = start + i
        axes[0, i].imshow(a[idx])
        axes[1, i].imshow(b[idx])
        for r in (0, 1):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
        title = titles[idx] if titles is not None else f"#{idx}"
        axes[0, i].set_title(title, fontsize=8)
    axes[0, 0].set_ylabel(row_labels[0], fontsize=10)
    axes[1, 0].set_ylabel(row_labels[1], fontsize=10)
    fig.tight_layout()
    return fig


def show_ball_patches(
    inputs: np.ndarray,
    recons: np.ndarray,
    source: dict[str, np.ndarray],
    config: pp_lib.PreprocessConfig,
    n: int = 8,
    start: int = 0,
    *,
    size: int = 32,
    skip_invalid: bool = True,
    scale: float = 1.6,
) -> "matplotlib.figure.Figure":
    """Grid of the ball patch in input (top) vs reconstruction (bottom).

    Crops the ``size×size`` patch around the known ball position from both stacks
    (see :func:`ball_patches`) and lays them out like
    :func:`show_reconstructions`. By default frames with no ball in the crop are
    skipped so every column shows an actual ball.

    Parameters
    ----------
    size :
        Patch side length. Larger tolerates ball-position error; the ball is ~2 px
        at 128×128 so 16–32 is a good range.
    skip_invalid :
        If ``True`` (default), drop out-of-crop (ball-less) frames before taking
        the first ``n``; if ``False``, show them as black patches.
    """
    ip, rp, valid = ball_patches(inputs, recons, source, config, size=size)
    idxs = np.arange(len(ip))
    if skip_invalid:
        idxs = idxs[valid]
    idxs = idxs[start : start + n]
    if len(idxs) == 0:
        raise ValueError("no valid ball patches to show (all out-of-crop?)")

    titles = [f"#{i}" for i in idxs]
    return show_reconstructions(
        ip[idxs], rp[idxs], n=len(idxs), start=0,
        titles=titles, scale=scale, row_labels=("ball (in)", "ball (rec)"),
    )
