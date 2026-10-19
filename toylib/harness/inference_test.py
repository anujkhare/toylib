"""Tests for toylib.harness.inference."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from toylib.harness import inference
from toylib.harness.data import ArrayDataset
from toylib.harness.experiment import CheckpointConfig, Experiment, LoggerConfig, Task


def _factory(config, key):
    del config, key
    return {"w": jnp.zeros((3, 2)), "b": jnp.zeros((2,))}


def _forward(model, batch):
    return jnp.mean(model["w"] ** 2), {}


@pytest.fixture
def ckpt_dir(tmp_path: Path) -> Path:
    """A checkpoint directory with saves at steps 1 and 5, holding different weights."""
    exp = Experiment(
        train_task=Task("train", ArrayDataset({"x": np.zeros((4, 3))}, batch_size=1)),
        forward_fn=_forward,
        model_factory=_factory,
        checkpoint_config=CheckpointConfig(checkpoint_dir=str(tmp_path / "ckpt")),
        logger_config=LoggerConfig(log_dir=str(tmp_path), run_id="inference"),
        jit_computations=False,
    )
    exp.init_state()
    for step, value in [(1, 1.0), (5, 5.0)]:
        exp.model = {"w": jnp.full((3, 2), value), "b": jnp.full((2,), -value)}
        exp.step = step
        exp.save_checkpoint()
    exp.cleanup()
    return tmp_path / "ckpt"


def test_checkpoint_steps_and_latest(ckpt_dir: Path):
    assert inference.checkpoint_steps(ckpt_dir) == [1, 5]
    assert inference.latest_step(ckpt_dir) == 5


def test_latest_step_raises_when_empty(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No checkpoints"):
        inference.latest_step(tmp_path / "empty")


def test_load_model_restores_each_step(ckpt_dir: Path):
    template = _factory(None, jax.random.key(0))
    latest = inference.load_model(template, ckpt_dir)
    first = inference.load_model(template, ckpt_dir, step=1)
    np.testing.assert_array_equal(latest["w"], np.full((3, 2), 5.0))
    np.testing.assert_array_equal(first["b"], np.full((2,), -1.0))
    assert isinstance(latest["w"], jax.Array)


def test_batched_apply_pads_last_batch_and_compiles_once():
    traces = []

    def fn(model, batch):
        traces.append(batch["x"].shape)
        return {"y": batch["x"] @ model["w"], "norm": jnp.linalg.norm(batch["x"], axis=-1)}

    model = {"w": jnp.arange(6.0).reshape(3, 2)}
    x = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    out = inference.batched_apply(fn, model, {"x": x}, batch_size=4)

    assert out["y"].shape == (10, 2) and out["norm"].shape == (10,)
    np.testing.assert_allclose(out["y"], x @ np.asarray(model["w"]), rtol=1e-5)
    np.testing.assert_allclose(out["norm"], np.linalg.norm(x, axis=-1), rtol=1e-5)
    assert traces == [(4, 3)]  # one shape, one compile, despite the short last batch
