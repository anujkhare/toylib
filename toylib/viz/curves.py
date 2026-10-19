"""Training curves read from the harness's run logs.

``FileLogger`` writes one JSON object per line with a ``step`` field and the
logged metrics (``train/loss``, ``val/loss``, ...). ``read_log`` turns that file
into per-metric ``(steps, values)`` arrays; ``plot_curves`` compares runs.
"""

import json
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np

from toylib.viz import style

Curves = dict[str, tuple[np.ndarray, np.ndarray]]


def read_log(path: str | pathlib.Path) -> Curves:
    """Parse a ``FileLogger`` JSON-lines file into ``{metric: (steps, values)}``.

    Only numeric scalar metrics are kept; other entries (samples, timestamps)
    are skipped.
    """
    points: dict[str, list[tuple[int, float]]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = record.get("step")
            if step is None:
                continue
            for key, value in record.items():
                if key in ("step", "timestamp") or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    points.setdefault(key, []).append((int(step), float(value)))
    return {
        key: (np.array([s for s, _ in pts]), np.array([v for _, v in pts]))
        for key, pts in points.items()
    }


def find_log(log_dir: str | pathlib.Path, run_id: str | None = None) -> pathlib.Path:
    """Return ``logs_<run_id>.txt`` in ``log_dir``, or the newest log if no id is given."""
    log_dir = pathlib.Path(log_dir)
    if run_id is not None:
        return log_dir / f"logs_{run_id}.txt"
    logs = sorted(log_dir.glob("logs_*.txt"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"No logs_*.txt files in {log_dir}")
    return logs[-1]


def ema(values: np.ndarray, smoothing: int | float) -> np.ndarray:
    """Exponential moving average with debiasing; ``smoothing`` in [0, 1)."""
    if smoothing <= 0:
        return np.asarray(values, dtype=float)
    out = np.empty(len(values))
    avg = 0.0
    for i, v in enumerate(values):
        avg = smoothing * avg + (1 - smoothing) * v
        out[i] = avg / (1 - smoothing ** (i + 1))
    return out


def _label_line_ends(ax: plt.Axes, ends: list, min_gap_pt: float = 11.0) -> None:
    """Label each line at its last point, nudging labels apart so none overlap."""
    ax.autoscale_view()
    to_pt = 72.0 / ax.figure.dpi
    ys = [ax.transData.transform((x, y))[1] * to_pt for _, x, y in ends]
    order = sorted(range(len(ends)), key=lambda i: ys[i])
    placed = list(ys)
    for prev, cur in zip(order, order[1:]):
        placed[cur] = max(placed[cur], placed[prev] + min_gap_pt)
    # Re-center the stack so labels stay as close to their lines as possible.
    shift = (sum(ys) - sum(placed)) / len(ys)
    for i, (name, x, y) in enumerate(ends):
        ax.annotate(name, xy=(x, y), xytext=(6, placed[i] - ys[i] + shift),
                    textcoords="offset points", va="center", fontsize=9,
                    color=style.INK_SECONDARY)


def plot_curves(
    runs: typing.Mapping[str, str | pathlib.Path | Curves],
    metric: str = "train/loss",
    *,
    smoothing: int | float = 0.0,
    log_y: bool = False,
    ax: plt.Axes | None = None,
    ylabel: str | None = None,
) -> plt.Axes:
    """Plot one metric for several runs on one axis.

    ``runs`` maps a display name to a log path (or to curves already read).
    Colors follow the fixed categorical order of the runs as given, so a run
    keeps its color across figures if you keep the order. With two or more
    runs a legend is drawn, and with up to four each line is also labeled at
    its end.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.6))
    if len(runs) > len(style.CATEGORICAL):
        raise ValueError(
            f"{len(runs)} runs is more than the {len(style.CATEGORICAL)} palette slots; "
            "use small multiples instead"
        )
    ends = []  # (name, last step, last smoothed value) for direct labels
    for i, (name, run) in enumerate(runs.items()):
        curves = run if isinstance(run, dict) else read_log(run)
        if metric not in curves:
            raise KeyError(f"Run {name!r} has no metric {metric!r}; has {sorted(curves)}")
        steps, values = curves[metric]
        smoothed = ema(values, smoothing)
        ax.plot(steps, smoothed, color=style.CATEGORICAL[i], label=name)
        ends.append((name, steps[-1], smoothed[-1]))
    if log_y:
        ax.set_yscale("log")
    if 2 <= len(runs) <= 4:
        _label_line_ends(ax, ends)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel or metric)
    if len(runs) >= 2:
        ax.legend(loc="upper right")
    return ax
