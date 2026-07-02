"""Overlay ball/paddle RAM-state labels on the compiled 128x128 frames.

Diagnostic for the latent-probe investigation: the probe recovers ball_x well
but ball_y poorly across *all* encoders (incl. raw pixels), which points at the
target rather than the encoder. This script draws a box at the (mapped) ball
position and a marker at the paddle on a montage of random frames so we can
eyeball whether the labels actually track the visible ball.

The labels are raw Atari 2600 RAM bytes (ball_x=addr 99, ball_y=addr 101,
paddle_x=addr 72). They are NOT pixel coordinates, so we apply a tunable affine
RAM->pixel map (calibrated by eye against this montage) on top of the known
crop+resize used to build the dataset (crop rows[32:192], cols[0:160], then
resize 160->128).

Run::

    uv run python -m toylib_projects.wm.scripts.viz_ball_labels
"""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — registers LZ4/zstd filters before any h5py read
import numpy as np
from PIL import Image, ImageDraw

_DATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "vision_encoder"
    / "v0"
    / "frames_train.h5"
)
_OUT = Path("/tmp/ball_labels_montage.png")

# Dataset build transform (see datagen/preprocess_frames.py).
CROP_TOP, CROP_LEFT = 32, 0
CROP_H, CROP_W = 160, 160  # rows[32:192], cols[0:160]
TARGET = 128
SCALE = TARGET / CROP_H  # 0.8

# RAM-byte -> 128px affine, calibrated against detected ball centroids by
# scripts/calib_ball_labels.py (R^2 = 1.0000 / 0.9997, residual < 0.4 px).
BALL_X_A, BALL_X_B = 0.7997, -38.87
BALL_Y_A, BALL_Y_B = 0.7950, -16.71
# Paddle uses the same x-scale; its row is roughly fixed near the bottom.
PADDLE_128_Y = 118


def ram_to_128(ram_x: float, ram_y: float) -> tuple[float, float]:
    return BALL_X_A * ram_x + BALL_X_B, BALL_Y_A * ram_y + BALL_Y_B


def main(n: int = 100, cols: int = 10, upscale: int = 3, seed: int = 0) -> None:
    f = h5py.File(_DATA, "r")
    frames = f["frames"]
    bx = f["source/ball_x"][:]
    by = f["source/ball_y"][:]
    px = f["source/paddle_x"][:]
    total = frames.shape[0]

    print(
        f"ball_x : min={bx.min():.0f} max={bx.max():.0f} mean={bx.mean():.1f} std={bx.std():.1f}"
    )
    print(
        f"ball_y : min={by.min():.0f} max={by.max():.0f} mean={by.mean():.1f} std={by.std():.1f}"
    )
    print(
        f"paddle_x: min={px.min():.0f} max={px.max():.0f} mean={px.mean():.1f} std={px.std():.1f}"
    )
    print(
        f"ball_y == min frac: {(by == by.min()).mean():.3f}  (possible off-screen sentinel)"
    )

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=n, replace=False))

    rows = (n + cols - 1) // cols
    cell = TARGET * upscale
    pad = 2
    sheet = Image.new(
        "RGB", (cols * (cell + pad) + pad, rows * (cell + pad) + pad), (30, 30, 30)
    )

    for k, i in enumerate(idx):
        frame = np.asarray(frames[i])
        img = Image.fromarray(frame).resize((cell, cell), Image.NEAREST)
        d = ImageDraw.Draw(img)

        # Ball box (red).
        x, y = ram_to_128(float(bx[i]), float(by[i]))
        x *= upscale
        y *= upscale
        r = 6
        d.rectangle([x - r, y - r, x + r, y + r], outline=(255, 255, 255), width=2)

        # Paddle marker (cyan vertical tick at fixed bottom row).
        pxx = (BALL_X_A * float(px[i]) + BALL_X_B) * upscale
        pyy = PADDLE_128_Y * upscale
        d.line([pxx, pyy - 10, pxx, pyy + 10], fill=(40, 220, 220), width=2)

        d.text((2, 2), f"bx={bx[i]:.0f} by={by[i]:.0f}", fill=(255, 255, 0))

        cx = (k % cols) * (cell + pad) + pad
        cy = (k // cols) * (cell + pad) + pad
        sheet.paste(img, (cx, cy))

    sheet.save(_OUT)
    print(f"wrote {_OUT}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(n=9, cols=3, upscale=4, seed=seed)
