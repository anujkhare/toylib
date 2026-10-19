"""Turn a sequence of figures or images into a GIF ("filters over training")."""

import io
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def figure_to_array(fig: plt.Figure, *, dpi: int = 100) -> np.ndarray:
    """Render ``fig`` to a uint8 RGB array of shape (H, W, 3)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def save_gif(
    frames: typing.Sequence[np.ndarray | plt.Figure],
    path: str | pathlib.Path,
    *,
    fps: int | float = 4.0,
    hold_last: int = 4,
) -> pathlib.Path:
    """Write ``frames`` to a looping GIF at ``path``.

    Frames may be uint8 (H, W, 3) arrays or matplotlib figures (rendered and
    then closed). All frames are padded to the largest frame size. The last
    frame is shown ``hold_last`` times longer so the end state is readable.
    """
    if not frames:
        raise ValueError("No frames to write")
    arrays = []
    for frame in frames:
        if isinstance(frame, plt.Figure):
            arrays.append(figure_to_array(frame))
            plt.close(frame)
        else:
            arrays.append(np.asarray(frame, dtype=np.uint8))
    h = max(a.shape[0] for a in arrays)
    w = max(a.shape[1] for a in arrays)
    images = []
    for a in arrays:
        canvas = np.full((h, w, 3), 252, dtype=np.uint8)
        canvas[: a.shape[0], : a.shape[1]] = a[..., :3]
        images.append(Image.fromarray(canvas))

    frame_ms = int(1000 / fps)
    durations = [frame_ms] * (len(images) - 1) + [frame_ms * (1 + hold_last)]
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)
    return path
