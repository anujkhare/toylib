"""HDF5 shard writer for Stage 1 raw episodes.

Writes one episode per HDF5 group; rolls to a new shard file after every
`episodes_per_shard` episodes. Frames are LZ4-compressed; small per-step
arrays are stored uncompressed.

Schema per episode group (see dataset.md §2):

    episode_NNNNNN/
      attrs: length (int32), mode (int32), difficulty (int32), seed (int64)
      frames          — (L, 210, 160, 3) uint8, LZ4, chunk=(1, 210, 160, 3)
      actions         — (L,) int32
      states/
        paddle_x          — (L,) float32
        ball_x            — (L,) float32
        ball_y            — (L,) float32
        score             — (L,) int32
        bricks_remaining  — (L,) int32
        lives             — (L,) int32
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import h5py
import hdf5plugin
import numpy as np

from .breakout import State

_STATE_INT_KEYS = ("score", "bricks_remaining", "lives")
_STATE_FLOAT_KEYS = ("paddle_x", "ball_x", "ball_y")


class ShardWriter:
    """Streaming writer that rolls into multiple HDF5 shards."""

    def __init__(self, output_dir: Path, episodes_per_shard: int) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_shard = episodes_per_shard
        self._shard_index = 0
        self._global_ep_index = 0
        self._ep_in_shard = 0
        self._shard: h5py.File | None = None

    def __enter__(self) -> "ShardWriter":
        self._open_new_shard()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._shard is not None:
            self._shard.close()
            self._shard = None

    def _open_new_shard(self) -> None:
        self.close()
        path = self.output_dir / f"episodes_shard_{self._shard_index:04d}.h5"
        self._shard = h5py.File(path, "w")
        self._ep_in_shard = 0

    def write_episode(
        self,
        frames: np.ndarray,
        actions: np.ndarray,
        states: list[State],
        *,
        mode: int = 0,
        difficulty: int = 0,
        seed: int | None = None,
    ) -> None:
        if self._shard is None:
            raise RuntimeError("ShardWriter must be used as a context manager")
        if self._ep_in_shard >= self.episodes_per_shard:
            self._shard_index += 1
            self._open_new_shard()

        length = int(frames.shape[0])
        assert frames.shape == (length, 210, 160, 3) and frames.dtype == np.uint8
        assert actions.shape == (length,) and actions.dtype == np.int32
        assert len(states) == length

        grp = self._shard.create_group(f"episode_{self._global_ep_index:06d}")
        grp.attrs["length"] = np.int32(length)
        grp.attrs["mode"] = np.int32(mode)
        grp.attrs["difficulty"] = np.int32(difficulty)
        if seed is not None:
            grp.attrs["seed"] = np.int64(seed)

        grp.create_dataset(
            "frames",
            data=frames,
            chunks=(1, 210, 160, 3),
            **hdf5plugin.LZ4(),
        )
        grp.create_dataset("actions", data=actions)

        states_grp = grp.create_group("states")
        for key in _STATE_FLOAT_KEYS:
            arr = np.array([s[key] for s in states], dtype=np.float32)
            states_grp.create_dataset(key, data=arr)
        for key in _STATE_INT_KEYS:
            arr = np.array([s[key] for s in states], dtype=np.int32)
            states_grp.create_dataset(key, data=arr)

        self._shard.flush()
        self._ep_in_shard += 1
        self._global_ep_index += 1
