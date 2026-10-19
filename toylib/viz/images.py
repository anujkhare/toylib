"""Grids of images: dataset samples, predictions, conv filters, generated samples."""

import math
import typing

import matplotlib.pyplot as plt
import numpy as np

from toylib.viz import style


def to_display(images: np.ndarray) -> np.ndarray:
    """Convert a batch of images to float32 in [0, 1] with shape (N, H, W) or (N, H, W, 3).

    Accepts uint8 in [0, 255], floats in [0, 1], or floats in [-1, 1] (detected
    by a negative minimum). A trailing channel axis of size 1 is dropped.
    """
    x = np.asarray(images)
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim not in (3, 4):
        raise ValueError(f"Expected (N, H, W) or (N, H, W, C) images, got shape {x.shape}")
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    x = x.astype(np.float32)
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)


def image_grid(
    images: np.ndarray,
    *,
    ncols: int = 8,
    titles: typing.Sequence[str] | None = None,
    normalize: typing.Literal["auto", "each", "symmetric"] = "auto",
    cell_size: int | float = 1.1,
    suptitle: str | None = None,
) -> plt.Figure:
    """Draw ``images`` in a grid, optionally with a short title above each.

    Titles may span lines (``"label 7\npred 1"``); their font shrinks so the
    longest line fits the cell.

    ``normalize`` chooses how pixel values map to color:

    * ``"auto"``: treat the batch as images (see ``to_display``); grayscale
      uses a black-on-white map.
    * ``"each"``: min-max scale every image on its own (activation maps).
    * ``"symmetric"``: signed values such as conv kernels; each image is scaled
      to its own max ``|value|`` and drawn with the diverging map, so zero is
      gray, positive red and negative blue.
    """
    x = np.asarray(images)
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    n = x.shape[0]
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    if normalize == "auto":
        shown = to_display(x)
        kwargs = {"cmap": "gray_r", "vmin": 0.0, "vmax": 1.0}
    elif normalize == "each":
        shown = x.astype(np.float32)
        lo = shown.reshape(n, -1).min(axis=1).reshape((n,) + (1,) * (shown.ndim - 1))
        hi = shown.reshape(n, -1).max(axis=1).reshape((n,) + (1,) * (shown.ndim - 1))
        shown = (shown - lo) / np.maximum(hi - lo, 1e-12)
        kwargs = {"cmap": "toylib_sequential", "vmin": 0.0, "vmax": 1.0}
    elif normalize == "symmetric":
        if x.ndim != 3:
            raise ValueError("normalize='symmetric' needs single-channel images (N, H, W)")
        scale = np.abs(x).reshape(n, -1).max(axis=1).reshape(n, 1, 1)
        shown = x / np.maximum(scale, 1e-12)
        kwargs = {"cmap": "toylib_diverging", "vmin": -1.0, "vmax": 1.0}
    else:
        raise ValueError(f"Unknown normalize mode {normalize!r}")

    # Lay the grid out in inches, so cells, titles and gaps keep their size
    # whatever the number of rows and columns.
    gap = 0.06
    title_size, title_h = 0.0, 0.0
    if titles is not None:
        lines = [line for t in titles[:n] for line in str(t).split("\n")]
        n_lines = max(str(t).count("\n") + 1 for t in titles[:n])
        longest = max((len(line) for line in lines), default=1)
        # ~0.6 em per character; shrink the font until the longest line fits the cell.
        title_size = float(np.clip(cell_size * 72 * 0.95 / (0.6 * max(longest, 1)), 5.0, 8.5))
        title_h = n_lines * title_size * 1.3 / 72 + 0.04
    head_h = 0.4 if suptitle else 0.0
    width = ncols * cell_size + (ncols - 1) * gap
    height = head_h + nrows * (cell_size + title_h) + (nrows - 1) * gap
    fig = plt.figure(figsize=(width, height))

    for i in range(n):
        row, col = divmod(i, ncols)
        left = col * (cell_size + gap)
        top = head_h + row * (cell_size + title_h + gap) + title_h
        ax = fig.add_axes((left / width, 1 - (top + cell_size) / height,
                           cell_size / width, cell_size / height))
        ax.axis("off")
        img = shown[i]
        ax.imshow(img, interpolation="nearest", **({} if img.ndim == 3 else kwargs))
        if titles is not None:
            ax.set_title(titles[i], fontsize=title_size, fontweight="normal",
                         color=style.INK_SECONDARY, loc="center", pad=2)
    if suptitle:
        fig.text(0.0, 1 - 0.06 / height, suptitle, ha="left", va="top",
                 fontsize=11, fontweight="semibold", color=style.INK)
    return fig
