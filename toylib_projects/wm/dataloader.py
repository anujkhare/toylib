"""Dataloaders for the vision encoder (VAE) training pipeline.

Design notes
------------

Two patterns live in this file:

  - ``DatasetState`` — a generic dataclass-based "state" type that any
    dataset can subclass. The intent is that dataset checkpoints round-trip
    cleanly through orbax/Pickle by being plain dataclasses.

  - ``Hdf5FramesDataset`` — concrete loader over the compiled vae_*.h5 files
    produced by ``datagen/generate_vision_enc_data.py``. Uses Grain's random
    access pipeline (shuffle → batch → iter) and exposes Grain's iterator
    state via ``get_state`` / ``restore_state`` so training can checkpoint
    its position in the shuffled stream.

We deliberately do **not** inherit the frames loader from any text-streaming
base class: the tokenizer / token-buffer / seq_len machinery used by the
text loader doesn't apply to images. We only share the ``DatasetState``
checkpointing protocol.
"""

import dataclasses

# import datasets as hf_datasets
import h5py
import hdf5plugin  # registers LZ4/zstd filters before any h5py read
import jax.numpy as jnp
import numpy as np
import typing

import grain.python as grain


@dataclasses.dataclass
class DatasetState:
    """Serializable state for dataset checkpointing."""

    pass


@dataclasses.dataclass
class Hdf5FramesDatasetState(DatasetState):
    """Checkpointable state for ``Hdf5FramesDataset``.

    Currently just the Grain iterator's internal state (the next index in the
    shuffled stream). Kept as a dict — Grain's get/set_state already returns
    a dict — so this trivially round-trips through json/orbax.
    """

    sampler_state: dict = dataclasses.field(default_factory=dict)


class _Hdf5FramesSource:
    """Grain RandomAccessDataSource backed by one compiled vae_*.h5 file.

    Implements the protocol Grain expects of a random-access source:
    ``__len__`` + ``__getitem__``. Each ``__getitem__`` returns a single
    frame as ``(H, W, 3)`` uint8.

    The HDF5 file handle is opened **lazily** on first access. This matters
    because Grain may pickle this source to send to worker processes; an
    open ``h5py.File`` cannot be pickled. The ``__getstate__`` /
    ``__setstate__`` overrides drop the live handle on pickle so each
    worker opens its own descriptor.
    """

    def __init__(self, dataset_path: str) -> None:
        self._path = str(dataset_path)
        self._f: typing.Optional[h5py.File] = None
        # Read the small attrs eagerly so __len__ and frame_shape are cheap
        # and don't require an open file handle on the main process.
        with h5py.File(self._path, "r") as f:
            self._n = int(f.attrs["n_frames"])
            self._height = int(f.attrs["height"])
            self._width = int(f.attrs["width"])

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return (self._height, self._width, 3)

    def _ensure_open(self) -> None:
        if self._f is None:
            self._f = h5py.File(self._path, "r")

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> np.ndarray:
        self._ensure_open()
        # h5py random access on the first axis decompresses exactly one chunk
        # — our compiled file uses chunks=(1, H, W, 3), so this is per-frame.
        return self._f["frames"][idx]  # type: ignore[index]

    def __getstate__(self) -> dict:
        # Drop the live file handle before pickling; recipient re-opens lazily.
        return {
            "_path": self._path,
            "_f": None,
            "_n": self._n,
            "_height": self._height,
            "_width": self._width,
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


@dataclasses.dataclass
class Hdf5FramesDataset:
    """Stream batches of frames from a compiled vae_*.h5 file.

    Yields ``jnp.ndarray`` batches of shape ``(batch_size, H, W, 3)`` with
    dtype ``uint8``. Normalization (uint8 → float32 / [-1, 1]) is left to
    the training loop so this loader stays pure and easy to inspect in tests.

    Checkpointing follows the same ``DatasetState`` / ``get_state`` /
    ``restore_state`` protocol used elsewhere in the project. The state
    captures the Grain iterator's position so resuming after a crash gives
    the same per-epoch shuffle order from the next index forward.

    Args
    ----
    dataset_path :
        Path to ``data/compiled/vae_train.h5`` (or val/test).
    batch_size :
        Number of frames per batch.
    seed :
        Shuffle seed. Together with the iterator state, fully determines the
        shuffled order for reproducible resumes.
    shuffle :
        If False, frames are read in storage order. The compiled file is
        already pre-shuffled at write time, so a sequential read is also a
        valid (and faster) "random-but-fixed" order.
    drop_remainder :
        Drop the last partial batch (so every batch has exactly
        ``batch_size`` frames).
    """

    dataset_path: str  # e.g. data/compiled/vae_train.h5
    batch_size: int = 64
    seed: int = 0
    shuffle: bool = True
    drop_remainder: bool = True
    repeat: bool = False

    def __post_init__(self) -> None:
        self._state = Hdf5FramesDatasetState()
        self._source = _Hdf5FramesSource(self.dataset_path)
        self._grain_iterator: typing.Optional[typing.Any] = None
        self.dataset_iter = self._make_iterator()

    # ---- introspection -----------------------------------------------------

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """``(H, W, 3)`` per-frame shape, read from the file attrs."""
        return self._source.frame_shape

    @property
    def num_frames(self) -> int:
        """Total frames in the underlying file (before batching)."""
        return len(self._source)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self._source) // self.batch_size
        if not self.drop_remainder and len(self._source) % self.batch_size:
            n += 1
        return n

    # ---- iterator construction --------------------------------------------

    def _make_iterator(self) -> typing.Iterator[jnp.ndarray]:
        """Build the Grain pipeline and yield (batch_size, H, W, 3) batches.

        When ``repeat=True`` the pipeline loops indefinitely; each epoch uses
        ``seed + epoch`` so the shuffle order differs across epochs.
        """
        epoch = 0
        while True:
            dataset = grain.MapDataset.source(self._source)
            if self.shuffle:
                dataset = dataset.shuffle(seed=self.seed + epoch)
            dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
            iterator = iter(dataset)

            # Restore prior position only on the first epoch (epoch 0).
            if epoch == 0 and self._state.sampler_state:
                iterator.set_state(self._state.sampler_state)

            self._grain_iterator = iterator

            for batch in iterator:
                # Grain returns numpy uint8 of shape (batch_size, H, W, 3).
                yield jnp.asarray(batch)

            if not self.repeat:
                break
            epoch += 1

    # ---- iterator protocol -------------------------------------------------

    def __iter__(self) -> "Hdf5FramesDataset":
        return self

    def __next__(self) -> jnp.ndarray:
        return next(self.dataset_iter)

    # ---- checkpoint protocol ----------------------------------------------

    def get_state(self) -> dict[str, typing.Any]:
        """Capture the current shuffled-iterator position for checkpointing."""
        if self._grain_iterator is None:
            raise RuntimeError("Iterator not initialized; call __post_init__ first.")
        self._state.sampler_state = self._grain_iterator.get_state()
        return dataclasses.asdict(self._state)

    def restore_state(self, state: dict[str, typing.Any]) -> None:
        """Restore from a previously captured ``get_state()`` dict.

        Rebuilds the iterator and fast-forwards to the saved position. After
        this call, ``__next__`` will return the same batch sequence as if the
        original run hadn't been interrupted (assuming same ``seed``,
        ``batch_size``, and underlying file).
        """
        self._state = Hdf5FramesDatasetState(**state)
        self.dataset_iter = self._make_iterator()
