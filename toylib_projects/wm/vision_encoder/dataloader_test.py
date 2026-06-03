"""Tests for Hdf5FramesDataset.

Each test writes a tiny synthetic ``vae_*.h5`` matching the schema produced
by ``datagen/generate_vision_enc_data.py`` and exercises the loader against
it.
"""

import dataclasses
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers LZ4 filter)
import jax.numpy as jnp
import numpy as np
import pytest

from .dataloader import (
    DatasetState,
    Hdf5FramesDataset,
    Hdf5FramesDatasetState,
)


def _write_fake_vae_h5(path: Path, n: int = 32, h: int = 16, w: int = 16) -> Path:
    """Tiny synthetic compiled VAE file. Each frame's pixel value encodes its
    index, so we can verify which frame the loader handed us."""
    rng = np.random.default_rng(0)
    frames = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        # Fill red channel with index so we can recover it from any frame.
        frames[i, :, :, 0] = i
        frames[i, :, :, 1] = rng.integers(0, 256, size=(h, w))
        frames[i, :, :, 2] = rng.integers(0, 256, size=(h, w))
    with h5py.File(path, "w") as f:
        f.attrs["n_frames"] = np.int32(n)
        f.attrs["height"] = np.int32(h)
        f.attrs["width"] = np.int32(w)
        f.create_dataset("frames", data=frames, chunks=(1, h, w, 3), **hdf5plugin.LZ4())
    return path


def _indices_from_batches(batches: list[jnp.ndarray]) -> list[int]:
    """Recover per-frame indices from the red-channel encoding."""
    out: list[int] = []
    for b in batches:
        for fr in np.asarray(b):
            out.append(int(fr[0, 0, 0]))
    return out


# ────────────────────────────────────────────────────────────────────────────
# Basic shape / dtype / iteration
# ────────────────────────────────────────────────────────────────────────────


def test_state_subclass_is_dataclass() -> None:
    """Hdf5FramesDatasetState must subclass DatasetState and stay a dataclass."""
    assert issubclass(Hdf5FramesDatasetState, DatasetState)
    s = Hdf5FramesDatasetState(sampler_state={"next_index": 5})
    assert dataclasses.asdict(s) == {"sampler_state": {"next_index": 5}}


def test_basic_iteration_shape_and_dtype(tmp_path: Path) -> None:
    _write_fake_vae_h5(tmp_path / "vae_train.h5", n=32, h=16, w=16)
    ds = Hdf5FramesDataset(
        dataset_path=str(tmp_path / "vae_train.h5"),
        batch_size=8,
        seed=0,
    )
    assert ds.num_frames == 32
    assert ds.frame_shape == (16, 16, 3)
    assert len(ds) == 4  # 32 / 8

    batch = next(iter(ds))
    assert batch.shape == (8, 16, 16, 3)
    assert batch.dtype == jnp.uint8


def test_drop_remainder_false_yields_partial(tmp_path: Path) -> None:
    _write_fake_vae_h5(tmp_path / "v.h5", n=10)
    ds = Hdf5FramesDataset(
        dataset_path=str(tmp_path / "v.h5"),
        batch_size=4,
        drop_remainder=False,
        shuffle=False,
    )
    assert len(ds) == 3  # 2 full + 1 partial
    batches = list(ds)
    assert [b.shape[0] for b in batches] == [4, 4, 2]


def test_no_shuffle_returns_storage_order(tmp_path: Path) -> None:
    _write_fake_vae_h5(tmp_path / "v.h5", n=12)
    ds = Hdf5FramesDataset(
        dataset_path=str(tmp_path / "v.h5"),
        batch_size=4,
        shuffle=False,
    )
    seen = _indices_from_batches(list(ds))
    assert seen == list(range(12))


def test_shuffle_is_deterministic(tmp_path: Path) -> None:
    """Same seed → same shuffled order across fresh dataset instances."""
    _write_fake_vae_h5(tmp_path / "v.h5", n=24)
    seen_a = _indices_from_batches(
        list(
            Hdf5FramesDataset(
                dataset_path=str(tmp_path / "v.h5"), batch_size=4, seed=42
            )
        )
    )
    seen_b = _indices_from_batches(
        list(
            Hdf5FramesDataset(
                dataset_path=str(tmp_path / "v.h5"), batch_size=4, seed=42
            )
        )
    )
    assert seen_a == seen_b
    # Different seed must change order (with 24 items the chance of equality is negligible).
    seen_c = _indices_from_batches(
        list(
            Hdf5FramesDataset(dataset_path=str(tmp_path / "v.h5"), batch_size=4, seed=7)
        )
    )
    assert seen_c != seen_a


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint round-trip
# ────────────────────────────────────────────────────────────────────────────


def test_get_state_restore_state_resumes_exactly(tmp_path: Path) -> None:
    """Checkpointing after k batches and restoring yields the same remaining sequence."""
    _write_fake_vae_h5(tmp_path / "v.h5", n=64)
    cfg = dict(
        dataset_path=str(tmp_path / "v.h5"),
        batch_size=8,
        seed=123,
        shuffle=True,
        drop_remainder=True,
    )

    # Reference: pull all 8 batches with one continuous dataset.
    ref = Hdf5FramesDataset(**cfg)
    reference = _indices_from_batches(list(ref))

    # Pause after 3 batches, capture state, build a fresh dataset, restore,
    # and finish — the indices should line up exactly with the reference.
    run1 = Hdf5FramesDataset(**cfg)
    first_three = [next(run1) for _ in range(3)]
    state = run1.get_state()

    run2 = Hdf5FramesDataset(**cfg)
    run2.restore_state(state)
    remainder = list(run2)

    resumed = _indices_from_batches(first_three) + _indices_from_batches(remainder)
    assert resumed == reference


def test_state_is_json_serializable(tmp_path: Path) -> None:
    """State must be plain Python so it can land in an orbax / json checkpoint."""
    import json

    _write_fake_vae_h5(tmp_path / "v.h5", n=16)
    ds = Hdf5FramesDataset(
        dataset_path=str(tmp_path / "v.h5"),
        batch_size=4,
        seed=0,
    )
    _ = next(iter(ds))  # advance once so the state isn't empty
    s = ds.get_state()
    # json round-trip must work and survive restore.
    s_round = json.loads(json.dumps(s))
    ds2 = Hdf5FramesDataset(
        dataset_path=str(tmp_path / "v.h5"),
        batch_size=4,
        seed=0,
    )
    ds2.restore_state(s_round)
    assert isinstance(next(ds2), jnp.ndarray)


# ────────────────────────────────────────────────────────────────────────────
# Source pickle-safety (multi-worker prep)
# ────────────────────────────────────────────────────────────────────────────


def test_source_survives_pickle(tmp_path: Path) -> None:
    """_Hdf5FramesSource must pickle cleanly even after a read (drops live handle)."""
    import pickle

    from .dataloader import _Hdf5FramesSource

    _write_fake_vae_h5(tmp_path / "v.h5", n=8)
    src = _Hdf5FramesSource(str(tmp_path / "v.h5"))
    _ = src[0]  # opens the file
    pickled = pickle.dumps(src)  # must not include the live handle
    restored = pickle.loads(pickled)
    # Restored copy has no open handle; first __getitem__ re-opens.
    assert restored[3].shape == (16, 16, 3)
    assert int(restored[3][0, 0, 0]) == 3  # index encoding survived
