"""Smoke tests for the experiment harness.

Exercises construction, config validation, logger plumbing, and a minimal
end-to-end loop using a *trivial* "model" (a small parameter vector) and a
forward_fn whose loss depends only on it. The point is to confirm the
infrastructure (sharding, microbatching, optimizer chain, checkpoint
manager, logger sink, dataset iteration) wires together, not to test any
model behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from toylib.harness.data import ArrayDataset
from toylib.harness.experiment import (
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


def _frames(n: int, h: int = 8, w: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)
    return {"frames": frames}


def _trivial_model_factory(config, key):
    """A tiny dict-of-arrays — orbax needs structured pytrees, not a 0-d scalar."""
    del config, key
    return {"w": jnp.zeros((4,))}


def _trivial_forward_fn(model, batch):
    """Loss is just ``(w**2).mean()`` so grads are nonzero and observable."""
    del batch
    return (model["w"] ** 2).mean(), {"dummy": jnp.array(0.0)}


@pytest.fixture
def train_ds() -> ArrayDataset:
    return ArrayDataset(
        _frames(16),
        batch_size=jax.local_device_count(),  # smallest divisible-by-devices size
        shuffle=False,
    )


@pytest.fixture
def val_ds() -> ArrayDataset:
    return ArrayDataset(
        _frames(8),
        batch_size=jax.local_device_count(),
        shuffle=False,
        restart_each_iter=True,
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
    n_devices = jax.local_device_count()
    batch_size = 2 * n_devices  # per-device batch = 2
    ds = ArrayDataset(_frames(16), batch_size=batch_size)
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


def _linear_forward_fn(model, batch):
    """Mean of the frames times w: depends on the batch, so data order matters."""
    x = batch["frames"].astype(jnp.float32).mean(axis=(1, 2, 3)) / 255.0
    loss = jnp.mean((x[:, None] * model["w"] - 1.0) ** 2)
    return loss, {}


def test_jitted_loop_reduces_loss(tmp_path: Path) -> None:
    """The jitted path trains: loss after 50 steps is below the first step's."""
    ds = ArrayDataset(_frames(32), batch_size=4 * jax.local_device_count(), seed=1)
    exp = Experiment(
        train_task=Task("train", ds),
        forward_fn=_linear_forward_fn,
        model_factory=_trivial_model_factory,
        training_config=TrainingConfig(max_steps=50, num_microbatches=2),
        checkpoint_config=CheckpointConfig(
            checkpoint_dir=str(tmp_path / "ckpts"), save_interval_steps=10_000
        ),
        logger_config=LoggerConfig(log_dir=str(tmp_path), run_id="jit"),
    )
    exp.init_state()
    exp.outer_loop()
    exp.cleanup()
    rows = [json.loads(line) for line in (tmp_path / "logs_jit.txt").read_text().split("\n") if line.strip()]
    losses = [r["train/loss"] for r in rows if "train/loss" in r]
    assert len(losses) == 50
    assert losses[-1] < losses[0]


def test_dataset_iterator_checkpoint_resumes_mid_epoch(tmp_path: Path) -> None:
    """With checkpoint_dataset_iterator, a restored run continues the same data order."""

    def make(run_id: str) -> Experiment:
        return Experiment(
            train_task=Task("train", ArrayDataset(_frames(16), batch_size=jax.local_device_count(), seed=3)),
            forward_fn=_linear_forward_fn,
            model_factory=_trivial_model_factory,
            training_config=TrainingConfig(max_steps=6),
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=str(tmp_path / f"ckpts_{run_id}"),
                save_interval_steps=3,
                checkpoint_dataset_iterator=True,
            ),
            logger_config=LoggerConfig(log_dir=str(tmp_path), run_id=run_id),
            jit_computations=False,
        )

    exp = make("a")
    exp.init_state()
    exp.outer_loop()  # saves at steps 0 and 3
    exp.restore_checkpoint(step=3)
    expected_state = exp.train_task.dataset.get_state()
    exp.cleanup()

    resumed = make("a")
    resumed.init_state()
    resumed.restore_checkpoint(step=3)
    assert resumed.train_task.dataset.get_state() == expected_state
    assert resumed.train_task.dataset.get_state()["position"] == 4  # batch after step 3
    resumed.cleanup()


# ──────────────────────────────────────────────────────────────────────────
# Regressions
# ──────────────────────────────────────────────────────────────────────────

_MULTI_DEVICE_SCRIPT = """
import os, sys
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + sys.argv[1]
import jax, jax.numpy as jnp, numpy as np, optax
from toylib.harness.data import ArrayDataset
from toylib.harness.experiment import (CheckpointConfig, Experiment, LoggerConfig,
    MultiOptimizerConfig, OptimizerConfig, Task, TrainingConfig)

def forward(model, batch):
    pred = batch["x"] @ model["w"]
    return jnp.mean((pred - batch["y"]) ** 2), {}

rng = np.random.default_rng(0)
data = {"x": rng.normal(size=(8, 3)).astype(np.float32), "y": rng.normal(size=(8,)).astype(np.float32)}
sgd = MultiOptimizerConfig([OptimizerConfig("sgd", optax.sgd(1.0))], lambda path: "sgd")
exp = Experiment(
    train_task=Task("train", ArrayDataset(data, batch_size=8, shuffle=False)),
    forward_fn=forward,
    model_factory=lambda cfg, key: {"w": jnp.ones((3,))},
    training_config=TrainingConfig(optimizer_config=sgd, num_microbatches=2),
    checkpoint_config=CheckpointConfig(checkpoint_dir=sys.argv[2] + "/ckpt"),
    logger_config=LoggerConfig(log_dir=sys.argv[2], run_id="multi"),
)
exp.init_state()
batch = next(iter(exp.train_task.dataset))
before = jax.tree.map(np.asarray, exp.model)
true_grad = jax.grad(lambda m: forward(m, batch)[0])(before)
after, _, _ = exp.train_step_fn(exp.model, exp.opt_state, batch)
applied = before["w"] - np.asarray(after["w"])  # SGD with lr=1 applies exactly the gradient
print(jax.local_device_count(), float(np.linalg.norm(applied) / np.linalg.norm(true_grad["w"])))
"""


@pytest.mark.parametrize("num_devices", [1, 2, 4])
def test_gradient_is_not_scaled_by_device_count(tmp_path: Path, num_devices: int) -> None:
    """Under jit with sharded inputs the gradient is already global; the update
    must equal the true full-batch gradient whatever the device count."""
    import subprocess
    import sys

    script = tmp_path / "multi_device.py"
    script.write_text(_MULTI_DEVICE_SCRIPT)
    result = subprocess.run(
        [sys.executable, str(script), str(num_devices), str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    devices, ratio = result.stdout.split()[-2:]
    assert int(devices) == num_devices
    assert float(ratio) == pytest.approx(1.0, rel=1e-4)


def test_validation_runs_exactly_num_eval_steps_batches(tmp_path: Path, train_ds) -> None:
    calls = []

    def counting_forward(model, batch):
        calls.append(1)
        return _trivial_forward_fn(model, batch)

    exp = Experiment(
        train_task=Task("train", train_ds),
        eval_task=Task("val", ArrayDataset(_frames(8), batch_size=1, restart_each_iter=True)),
        forward_fn=_trivial_forward_fn,
        eval_forward_fn=counting_forward,
        model_factory=_trivial_model_factory,
        eval_config=EvalConfig(num_eval_steps=3),
        checkpoint_config=CheckpointConfig(checkpoint_dir=str(tmp_path / "ckpts")),
        logger_config=LoggerConfig(log_dir=str(tmp_path), run_id="eval_steps"),
        jit_computations=False,
    )
    exp.init_state()
    exp.run_validation()
    exp.cleanup()
    assert len(calls) == 3


def test_restore_checkpoint_defaults_to_latest(tmp_path: Path, train_ds) -> None:
    exp = Experiment(
        train_task=Task("train", train_ds),
        forward_fn=_trivial_forward_fn,
        model_factory=_trivial_model_factory,
        checkpoint_config=CheckpointConfig(checkpoint_dir=str(tmp_path / "ckpts")),
        logger_config=LoggerConfig(log_dir=str(tmp_path), run_id="latest"),
        jit_computations=False,
    )
    exp.init_state()
    for step in (2, 7):
        exp.model, exp.step = {"w": jnp.full((4,), float(step))}, step
        exp.save_checkpoint()
    exp.restore_checkpoint()
    assert exp.step == 7
    np.testing.assert_array_equal(np.asarray(exp.model["w"]), np.full((4,), 7.0))
    exp.cleanup()
