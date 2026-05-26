"""Episode visualization for Stage 1 raw HDF5 shards.

Two entry points, same underlying renderer:

  - **Jupyter / Colab**: load an episode and call ``show_episode``:

        from viz import load_episode, show_episode
        ep = load_episode("data/raw/episodes_shard_0000.h5", "episode_000000")
        show_episode(ep)        # returns IPython.display.HTML, renders inline

  - **Standalone HTML files**: use the CLI:

        uv run python -m viz.cli --input data/raw --output viz_out

    or call ``build_episode_page(ep)`` and write the string to a file.
"""

from __future__ import annotations

from pathlib import Path

from .loader import Episode, find_shards, iter_episodes, list_episodes, load_episode
from .render import (
    RenderOptions,
    build_episode_page,
    build_episode_viewer,
    build_shard_index_page,
)

__all__ = [
    "Episode",
    "RenderOptions",
    "build_episode_page",
    "build_episode_viewer",
    "build_shard_index_page",
    "find_shards",
    "iter_episodes",
    "list_episodes",
    "load_episode",
    "show_episode",
    "write_episode_html",
]


def show_episode(ep: Episode, opts: RenderOptions | None = None):
    """Return an IPython.display.HTML for inline rendering in Jupyter/Colab."""
    from IPython.display import HTML  # lazy import; not required for HTML files

    return HTML(build_episode_viewer(ep, opts))


def write_episode_html(
    ep: Episode, out_path: str | Path, opts: RenderOptions | None = None
) -> Path:
    """Write a self-contained .html for one episode and return its path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_episode_page(ep, opts))
    return out_path
