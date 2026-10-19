"""Shared figure style, so every post's figures look like one system.

Colors come from a CVD-validated palette. Assign series colors in the fixed
order of ``CATEGORICAL`` (never cycle past it; fold extra series into "other"
or use small multiples). Sequential maps are one hue light-to-dark; diverging
maps are blue and red around a neutral gray, for signed values such as weights.
"""

import pathlib

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Categorical slots, in order. Slots 1-3 stay distinguishable for every pair
# (use them for scatters); all eight are safe for adjacent series in lines/bars.
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Status colors are reserved for states (correct / wrong), always with a text label.
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
_NEUTRAL = "#f0efec"
# Equal-length arms so the neutral gray sits exactly at zero.
_DIVERGING_BLUE = ["#104281", "#2a78d6", "#86b6ef", "#cde2fb"]  # dark -> light
_DIVERGING_RED = ["#f9d0cf", "#ec8a89", "#e34948", "#9c2c2b"]  # light -> dark

SEQUENTIAL = mcolors.LinearSegmentedColormap.from_list(
    "toylib_sequential", [SURFACE] + _BLUE_RAMP
)
DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "toylib_diverging", _DIVERGING_BLUE + [_NEUTRAL] + _DIVERGING_RED
)


def use_style() -> None:
    """Apply the shared matplotlib style. Call once at the top of a figure script."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 6,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
            "legend.frameon": False,
            "legend.fontsize": 9,
            "image.cmap": "toylib_sequential",
        }
    )


def save(fig: plt.Figure, path: str | pathlib.Path, *, formats=("png",)) -> list[pathlib.Path]:
    """Save ``fig`` to ``path`` (suffix replaced per format), creating parent dirs."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out)
        written.append(out)
    return written


for _cmap in (SEQUENTIAL, DIVERGING):
    if _cmap.name not in mpl.colormaps:
        mpl.colormaps.register(_cmap)
