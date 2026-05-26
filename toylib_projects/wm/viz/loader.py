"""Load Stage 1 raw episodes out of HDF5 shards into in-memory dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import h5py
import hdf5plugin  # registers LZ4/zstd filters before any h5py read
import numpy as np

_STATE_KEYS = ("paddle_x", "ball_x", "ball_y", "score", "bricks_remaining", "lives")


@dataclass
class Episode:
    """One fully-loaded episode.

    All arrays have length ``L`` along axis 0. ``shard_path`` and ``name``
    identify where this episode came from so we can label viz output.

    ``mode`` / ``difficulty`` / ``seed`` are read from the HDF5 group attrs
    when present (default 0 for backwards compat with pre-multimode shards).
    """

    shard_path: Path
    name: str  # e.g. "episode_000042"
    length: int
    frames: np.ndarray  # (L, 210, 160, 3) uint8
    actions: np.ndarray  # (L,) int32
    states: dict[str, np.ndarray] = field(default_factory=dict)
    mode: int = 0
    difficulty: int = 0
    seed: int | None = None

    @property
    def stem(self) -> str:
        """Short identifier suitable for filenames: e.g. ``shard_0000__episode_000042``."""
        return f"{self.shard_path.stem}__{self.name}"


def list_episodes(shard_path: Path) -> list[str]:
    """Return the episode group names inside one shard, in storage order."""
    with h5py.File(shard_path, "r") as f:
        return sorted(f.keys())


def load_episode(shard_path: Path, name: str) -> Episode:
    """Load one episode from a shard. All arrays are eagerly decompressed."""
    with h5py.File(shard_path, "r") as f:
        grp = f[name]
        length = int(grp.attrs["length"])
        mode = int(grp.attrs["mode"]) if "mode" in grp.attrs else 0
        difficulty = int(grp.attrs["difficulty"]) if "difficulty" in grp.attrs else 0
        seed = int(grp.attrs["seed"]) if "seed" in grp.attrs else None
        frames = grp["frames"][:]
        actions = grp["actions"][:]
        states = {k: grp[f"states/{k}"][:] for k in _STATE_KEYS}
    return Episode(
        shard_path=Path(shard_path),
        name=name,
        length=length,
        frames=frames,
        actions=actions,
        states=states,
        mode=mode,
        difficulty=difficulty,
        seed=seed,
    )


def iter_episodes(shard_paths: list[Path]) -> Iterator[Episode]:
    """Yield every episode across an ordered list of shard files."""
    for shard in shard_paths:
        for name in list_episodes(shard):
            yield load_episode(shard, name)


def find_shards(root: Path) -> list[Path]:
    """Sorted list of `episodes_shard_*.h5` files under a directory.

    Searches recursively so a multi-mode dataset root (e.g. ``data/raw/``
    containing ``mode_00_diff_0/`` subdirs) Just Works.
    """
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(root.rglob("episodes_shard_*.h5"))
