"""Matrix heatmaps (attention patterns, confusion matrices) and per-token heatmaps."""

import typing

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from toylib.viz import style


def _ink_for(rgba: tuple[float, ...]) -> str:
    """Dark text on light cells, light text on dark cells."""
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return style.INK if luminance > 0.55 else style.SURFACE


def heatmap(
    matrix: np.ndarray,
    *,
    row_labels: typing.Sequence[str] | None = None,
    col_labels: typing.Sequence[str] | None = None,
    diverging: bool = False,
    vmin: int | float | None = None,
    vmax: int | float | None = None,
    annotate: bool = False,
    fmt: str = "{:.2f}",
    colorbar_label: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw a 2D array as a heatmap.

    Use the default sequential map for magnitudes (attention weights,
    probabilities, counts) and ``diverging=True`` for signed values, which
    centers the neutral gray on zero. ``annotate`` prints each value in its
    cell; keep it for small matrices such as a 10x10 confusion matrix.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"heatmap needs a 2D array, got shape {m.shape}")
    if ax is None:
        h, w = m.shape
        _, ax = plt.subplots(figsize=(min(1.0 + 0.35 * w, 10), min(0.8 + 0.35 * h, 10)))
    if diverging:
        bound = np.nanmax(np.abs(m)) if vmax is None else vmax
        norm = mcolors.Normalize(-bound, bound)
        cmap = "toylib_diverging"
    else:
        norm = mcolors.Normalize(np.nanmin(m) if vmin is None else vmin,
                                 np.nanmax(m) if vmax is None else vmax)
        cmap = "toylib_sequential"
    im = ax.imshow(m, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)), labels=row_labels)
    if col_labels is not None:
        ax.set_xticks(range(len(col_labels)), labels=col_labels)
    ax.tick_params(length=0)
    if annotate:
        cmap_obj = plt.get_cmap(cmap)
        for (i, j), value in np.ndenumerate(m):
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=8,
                    color=_ink_for(cmap_obj(norm(value))))
    bar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, colors=style.INK_MUTED, labelcolor=style.INK_SECONDARY)
    if colorbar_label:
        bar.set_label(colorbar_label)
    return ax


def token_heatmap(
    tokens: typing.Sequence[str],
    values: np.ndarray,
    *,
    diverging: bool = False,
    vmax: int | float | None = None,
    line_width: int = 80,
    colorbar_label: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Draw text with each token's background colored by a value.

    Made for per-character loss ("where is the model surprised?"), but works
    for any per-token score. Tokens are laid out left to right in a monospace
    grid that wraps at ``line_width`` characters and after newlines (shown as
    a return symbol so their value stays visible).
    """
    values = np.asarray(values, dtype=float)
    if len(tokens) != len(values):
        raise ValueError(f"{len(tokens)} tokens but {len(values)} values")
    bound = np.nanmax(np.abs(values)) if vmax is None else vmax
    if diverging:
        norm, cmap = mcolors.Normalize(-bound, bound), plt.get_cmap("toylib_diverging")
    else:
        norm, cmap = mcolors.Normalize(0, bound), plt.get_cmap("toylib_sequential")

    # Lay out tokens on a character grid.
    placed = []  # (row, col, width, text, value)
    row = col = 0
    for tok, value in zip(tokens, values):
        shown = tok.replace("\n", "↵")
        width = max(len(shown), 1)
        if col + width > line_width:
            row, col = row + 1, 0
        placed.append((row, col, width, shown, value))
        col += width
        if tok.endswith("\n"):
            row, col = row + 1, 0
    nrows = max(r for r, *_ in placed) + 1

    char_w, line_h = 0.105, 0.24  # inches per character cell
    top = 0.35 if title else 0.1  # inches
    bottom = 0.7 if colorbar_label else 0.1
    height = nrows * line_h + top + bottom
    fig = plt.figure(figsize=(line_width * char_w + 0.4, height))
    ax = fig.add_axes((0.02, bottom / height, 0.96, nrows * line_h / height))
    ax.set_xlim(0, line_width)
    ax.set_ylim(nrows, 0)
    ax.axis("off")
    for r, c, w, text, value in placed:
        color = cmap(norm(value))
        ax.add_patch(mpatches.Rectangle((c, r + 0.08), w, 0.84, facecolor=color, linewidth=0))
        ax.text(c + w / 2, r + 0.5, text, ha="center", va="center", family="monospace",
                fontsize=9, color=_ink_for(color))
    if title:
        ax.set_title(title, loc="left")
    if colorbar_label:
        cax = fig.add_axes((0.02, 0.3 / height, 0.3, 0.1 / height))
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                           orientation="horizontal")
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, colors=style.INK_MUTED, labelcolor=style.INK_SECONDARY)
        bar.set_label(colorbar_label)
    return fig
