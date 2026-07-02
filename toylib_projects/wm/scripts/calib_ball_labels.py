"""Calibrate the RAM-label -> 128px-pixel affine by detecting the ball.

For a sample of frames, find the ball's pixel centroid (the only colored blob
in the central play band, between the bricks and the paddle) and least-squares
fit pixel = a * ram + b, separately for x and y. The R^2 of that fit tells us
directly how well the RAM label tracks the visible ball.

Run::

    uv run python -m toylib_projects.wm.scripts.calib_ball_labels
"""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

_DATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "vision_encoder"
    / "v0"
    / "frames_train.h5"
)

# Central play band in 128px coords (exclude brick rows on top, paddle at bottom,
# and the gray side walls left/right).
BAND_TOP, BAND_BOT = 45, 112
BAND_LEFT, BAND_RIGHT = 12, 116


def ball_centroid(frame: np.ndarray) -> tuple[float, float, int]:
    """Return (x, y, n_pixels) of the red ball blob in the central band."""
    band = frame[BAND_TOP:BAND_BOT, BAND_LEFT:BAND_RIGHT].astype(np.int16)
    r, g, b = band[..., 0], band[..., 1], band[..., 2]
    # The ball is red; gray walls have R≈G≈B, so require red to dominate.
    mask = (r > 100) & (r - g > 30) & (r - b > 30)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return -1.0, -1.0, 0
    return float(xs.mean() + BAND_LEFT), float(ys.mean() + BAND_TOP), len(xs)


def fit(ram: np.ndarray, pix: np.ndarray) -> tuple[float, float, float]:
    """Least-squares pix = a*ram + b; return (a, b, r2)."""
    A = np.vstack([ram, np.ones_like(ram)]).T
    (a, b), *_ = np.linalg.lstsq(A, pix, rcond=None)
    pred = a * ram + b
    ss_res = np.sum((pix - pred) ** 2)
    ss_tot = np.sum((pix - pix.mean()) ** 2)
    return float(a), float(b), float(1 - ss_res / ss_tot)


def main(n: int = 3000, seed: int = 0) -> None:
    f = h5py.File(_DATA, "r")
    frames = f["frames"]
    bx = f["source/ball_x"][:]
    by = f["source/ball_y"][:]
    total = frames.shape[0]

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=n, replace=False))

    ram_x, ram_y, pix_x, pix_y = [], [], [], []
    for i in idx:
        cx, cy, npix = ball_centroid(np.asarray(frames[i]))
        # Keep frames with a small, ball-sized blob (reject empty / brick spill).
        if 1 <= npix <= 40:
            ram_x.append(bx[i])
            ram_y.append(by[i])
            pix_x.append(cx)
            pix_y.append(cy)

    ram_x = np.array(ram_x)
    ram_y = np.array(ram_y)
    pix_x = np.array(pix_x)
    pix_y = np.array(pix_y)
    print(f"usable frames: {len(ram_x)} / {n}")

    ax, bx_, r2x = fit(ram_x, pix_x)
    ay, by_, r2y = fit(ram_y, pix_y)
    print(f"ball_x: pix = {ax:.4f} * ram + {bx_:.2f}   R^2 = {r2x:.4f}")
    print(f"ball_y: pix = {ay:.4f} * ram + {by_:.2f}   R^2 = {r2y:.4f}")
    # Residual std in pixels = how tight the label is to the visible ball.
    print(
        f"resid std (px): x={np.std(pix_x - (ax * ram_x + bx_)):.2f}  y={np.std(pix_y - (ay * ram_y + by_)):.2f}"
    )


if __name__ == "__main__":
    main()
