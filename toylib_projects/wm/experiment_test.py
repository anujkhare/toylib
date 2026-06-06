"""Smoke tests for the forked experiment harness.

Exercises construction, config validation, logger plumbing, and a minimal
end-to-end loop using a *trivial* "model" (a single scalar parameter) and a
forward_fn that returns a constant loss. The point is to confirm the
infrastructure (sharding, microbatching, optimizer chain, checkpoint
manager, logger sink, dataset iteration) wires together — not to test any
model behavior.

A real VAE / model lives outside this file per the project's learning rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers LZ4 filter)
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from toylib_projects.wm.dataloader import Hdf5FramesDataset
from toylib_projects.wm.experiment import (
    CheckpointConfig,
    EvalConfig,
    Experiment,
    LoggerConfig,
    Task,
    TrainingConfig,
    _serialize_dataclass_config,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _write_tiny_vae_h5(path: Path, n: int = 16, h: int = 8, w: int = 8) -> Path:
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)
    with h5py.File(path, "w") as f:
        f.attrs["n_frames"] = np.int32(n)
        f.attrs["height"] = np.int32(h)
        f.attrs["width"] = np.int32(w)
        f.create_dataset(
            "frames", data=frames, chunks=(1, h, w, 3), **hdf5plugin.LZ4()
        )
    return path


def _trivial_model_factory(config, key):
    """A tiny dict-of-arrays — orbax needs structured pytrees, not a 0-d scalar."""
    del config, key
    return {"w": jnp.zeros((4,))}


def _trivial_forward_fn(model, batch):
    """Loss is just ``(w**2).mean()`` so grads are nonzero and observable."""
    del batch
    return (model["w"] ** 2).mean(), {"dummy": jnp.array(0.0)}


@pytest.fixture
def train_ds(tmp_path: Path) -> Hdf5FramesDataset:
    _write_tiny_vae_h5(tmp_path / "train.h5", n=16)
    return Hdf5FramesDataset(
        dataset_path=str(tmp_path / "train.h5"),
        batch_size=jax.local_device_count(),  # smallest divisible-by-devices size
        seed=0,
        shuffle=False,
    )


@pytest.fixture
def val_ds(tmp_path: Path) -> Hdf5FramesDataset:
    _write_tiny_vae_h5(tmp_path / "val.h5", n=8)
    return Hdf5FramesDataset(
        dataset_path=str(tmp_path / "val.h5"),
        batch_size=jax.local_device_count(),
        seed=0,
        shuffle=False,
    )


# Logger and metric sinks have their own dedicated test files
# (logger_test.py, metrics_test.py).


# ──────────────────────────────────────────────────────────────────────────
# Config serialization
# ──────────────────────────────────────────────────────────────────────────


def test_serialize_dataclass_config_handles_callables_and_dataclasses() -> None:
    """Mixed dataclasses + callables + plain values must serialize without crashing."""
    cfg = TrainingConfig(num_microbatches=2, max_grad_norm=1.0)
    out = _serialize_dataclass_config(cfg)
    assert isinstance(out, dict)
    assert out["num_microbatches"] == 2
    assert out["max_grad_norm"] == 1.0

    # A callable must survive serialization (stringified, not exploded).
    serialized = _serialize_dataclass_config({"fn": (lambda x: x)})
    assert isinstance(serialized["fn"], str)


# ──────────────────────────────────────────────────────────────────────────
# Experiment construction + validation
# ──────────────────────────────────────────────────────────────────────────


def test_experiment_construction(tmp_path: Path, train_ds, val_ds) -> None:
    exp = Experiment(
        train_task=Task("train", train_ds),
        eval_task=Task("val", val_ds),
        forward_fn=_trivial_forward_fn,
        model_factory=_trivial_model_factory,
        training_config=TrainingConfig(max_steps=2, num_microbatches=1),
        eval_config=EvalConfig(eval_interval_steps=1, num_eval_steps=1),
        checkpoint_config=CheckpointConfig(
            checkpoint_dir=str(tmp_path / "ckpts"),
            save_interval_steps=10_000,
        ),
        logger_config=LoggerConfig(log_dir=str(tmp_path), run_id="construction"),
        jit_computations=False,  # trivial test, no need to JIT
    )
    assert exp.num_devices >= 1
    assert exp.model is None  # init_state not yet called
    exp.cleanup()


def test_microbatch_divisibility_rejected(tmp_path: Path) -> None:
    """num_microbatches must evenly divide the per-device batch size.

    Picking the device-divisible / microbatch-indivisible case so the test
    runs on single-device CPU (where every batch size is trivially device-
    divisible).
    """
    _write_tiny_vae_h5(tmp_path / "train.h5", n=16)
    n_devices = jax.local_device_count()
    batch_size = 2 * n_devices  # per-device batch = 2
    ds = Hdf5FramesDataset(
        dataset_path=str(tmp_path / "train.h5"),
        batch_size=batch_size,
        seed=0,
    )
    with pytest.raises(ValueError, match="does not evenly divide per-device batch size"):
        Experiment(
            train_task=Task("train", ds),
            forward_fn=_trivial_forward_fn,
            model_factory=_trivial_model_factory,
            training_config=TrainingConfig(num_microbatches=3),  # 2 % 3 != 0
            logger_config=LoggerConfig(log_dir=str(tmp_path)),
            jit_computations=False,
        )


# ──────────────────────────────────────────────────────────────────────────
# End-to-end trivial loop
# ──────────────────────────────────────────────────────────────────────────


def test_trivial_loop_completes_and_logs(tmp_path: Path, train_ds, val_ds) -> None:
    """One full mini-loop: 4 train steps + validation + log file written.

    Tests the full pipeline (sharding, microbatching, optimizer step, eval,
    logging) end-to-end without touching anything model-specific.
    """
    exp = Experiment(
        train_task=Task("train", train_ds),
        eval_task=Task("val", val_ds),
        forward_fn=_trivial_forward_fn,
        model_factory=_trivial_model_factory,
        training_config=TrainingConfig(max_steps=4, num_microbatches=1),
        eval_config=EvalConfig(eval_interval_steps=2, num_eval_steps=1),
        checkpoint_config=CheckpointConfig(
            checkpoint_dir=str(tmp_path / "ckpts"),
            save_interval_steps=10_000,  # skip in this short run
        ),
        logger_config=LoggerConfig(log_dir=str(tmp_path), run_id="loop"),
        jit_computations=False,
    )
    exp.init_state()
    assert exp.model is not None
    assert exp.opt_state is not None

    exp.outer_loop()
    exp.cleanup()

    # 4 steps run with log_interval=1 means we should see at least 4 train rows.
    log_path = tmp_path / "logs_loop.txt"
    assert log_path.exists()
    rows = [
        json.loads(line) for line in log_path.read_text().strip().split("\n") if line.strip()
    ]
    train_rows = [r for r in rows if "train/loss" in r]
    val_rows = [r for r in rows if "val/loss" in r]
    assert len(train_rows) >= 4, f"expected ≥4 train rows, got {len(train_rows)}"
    assert len(val_rows) >= 1, f"expected ≥1 val row, got {len(val_rows)}"


class _EmptyDataset:
    """Iterable dataset stub that yields no batches (e.g. an empty eval split)."""

    def __init__(self) -> None:
        self.batch_size = jax.local_device_count()

    def __iter__(self):
        return iter(())


def test_run_validation_empty_eval_dataset_returns_empty(tmp_path: Path, train_ds) -> None:
    """An eval dataset that yields zero batches must not crash run_validation.

    Regression: an empty eval iterator left ``accumulated_scalars`` as None, so
    ``jax.tree.map(..., None)`` produced None and the subsequent ``.items()``
    raised ``AttributeError: 'NoneType' object has no attribute 'items'``.
    """
    exp = Experiment(
        train_task=Task("train", train_ds),
        eval_task=Task("val", _EmptyDataset()),
        forward_fn=_trivial_forward_fn,
        model_factory=_trivial_model_factory,
        training_config=TrainingConfig(max_steps=2, num_microbatches=1),
        eval_config=EvalConfig(eval_interval_steps=1, num_eval_steps=1),
        checkpoint_config=CheckpointConfig(
            checkpoint_dir=str(tmp_path / "ckpts"),
            save_interval_steps=10_000,
        ),
        logger_config=LoggerConfig(log_dir=str(tmp_path), run_id="empty_eval"),
        jit_computations=False,
    )
    exp.init_state()

    assert exp.run_validation() == {}
    # eval() chains run_validation + sampling_evaluation and must also survive.
    exp.eval()
    exp.cleanup()


def test_checkpoint_roundtrip(tmp_path: Path, train_ds) -> None:
    """save_checkpoint then restore_checkpoint must round-trip model + opt state."""
    exp = Experiment(
        train_task=Task("train", train_ds),
        forward_fn=_trivial_forward_fn,
        model_factory=_trivial_model_factory,
        training_config=TrainingConfig(max_steps=2),
        checkpoint_config=CheckpointConfig(
            checkpoint_dir=str(tmp_path / "ckpts"),
            save_interval_steps=10_000,
        ),
        logger_config=LoggerConfig(log_dir=str(tmp_path), run_id="ckpt"),
        jit_computations=False,
    )
    exp.init_state()
    # Mutate the model so we can detect a successful restore.
    exp.model = {"w": jnp.array([1.0, 2.0, 3.0, 4.0])}
    exp.step = 1
    exp.save_checkpoint()

    # Reset to a sentinel value, then restore.
    exp.model = {"w": jnp.zeros((4,))}
    exp.restore_checkpoint(step=1)
    np.testing.assert_array_equal(np.asarray(exp.model["w"]), [1.0, 2.0, 3.0, 4.0])
    exp.cleanup()
