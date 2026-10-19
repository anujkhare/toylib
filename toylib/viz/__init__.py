"""Plotting tools shared by every post.

    from toylib import viz
    viz.use_style()
    fig = viz.image_grid(x_test[:32], titles=[str(y) for y in y_test[:32]])
    viz.save(fig, "posts_out/samples.png")

Depends only on NumPy, matplotlib and Pillow, never on the harness or models.
"""

from toylib.viz.animation import figure_to_array, save_gif
from toylib.viz.curves import ema, find_log, plot_curves, read_log
from toylib.viz.heatmap import heatmap, token_heatmap
from toylib.viz.images import image_grid, to_display
from toylib.viz.style import (
    CATEGORICAL,
    CRITICAL,
    DIVERGING,
    GOOD,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SEQUENTIAL,
    SURFACE,
    save,
    use_style,
)

__all__ = [
    "CATEGORICAL", "CRITICAL", "DIVERGING", "GOOD", "INK", "INK_MUTED", "INK_SECONDARY",
    "SEQUENTIAL", "SURFACE", "ema", "figure_to_array", "find_log", "heatmap",
    "image_grid", "plot_curves", "read_log", "save", "save_gif", "to_display",
    "token_heatmap", "use_style",
]
