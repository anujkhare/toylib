"""The data -> model -> harness wiring, on synthetic data so no download is needed.

These keep passing as the model and loss are filled in: they check shapes and
plumbing, not what the model learns.
"""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from toylib.harness import inference
from toylib.harness.data import ArrayDataset
from toylib.harness.experiment import (
    CheckpointConfig,
    EvalConfig,
    Experiment,
    LoggerConfig,
    Task,
    TrainingConfig,
)
from toylib.harness.metrics import Loss
from toylib_projects.p01_cnn_mnist import data, metrics, model


def _arrays(n: int, image_shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "image": rng.normal(size=(n, *image_shape)).astype(np.float32),
        "label": rng.integers(0, 10, n).astype(np.int32),
    }


@pytest.mark.parametrize("name", ["mnist", "cifar10"])
def test_forward_shapes(name):
    image_shape = data.DATASETS[name].image_shape
    net = model.build_model(model.ModelConfig(image_shape=image_shape), jax.random.key(0))
    batch = _arrays(4, image_shape)

    logits, acts = net(batch["image"], return_intermediates=True)
    assert logits.shape == (4, 10)
    assert all(a.shape[0] == 4 for a in acts.values())

    loss, aux = model.forward(net, batch)
    assert loss.shape == ()
    assert aux["logits"].shape == (4, 10)
    grads = jax.grad(lambda m: model.forward(m, batch)[0])(net)
    assert jax.tree.structure(grads) == jax.tree.structure(net)


def test_accuracy_metric():
    logits = np.array([[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
    out = metrics.Accuracy()(loss=0.0, aux={"logits": logits}, batch={"label": np.array([1, 0, 0])})
    assert float(out["accuracy"]) == pytest.approx(2 / 3)


def test_preprocess_roundtrip():
    info = data.DATASETS["cifar10"]
    images = np.random.default_rng(0).integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)
    x = data.preprocess(images, info)
    assert x.dtype == np.float32
    np.testing.assert_allclose(data.unpreprocess(x, info), images / 255.0, atol=1e-6)


@pytest.mark.parametrize("name", ["mnist", "cifar10"])
def test_trains_through_harness(tmp_path: Path, name):
    """A few jitted steps with eval and checkpoints, then restore with inference."""
    image_shape = data.DATASETS[name].image_shape
    n_dev = jax.local_device_count()
    config = model.ModelConfig(image_shape=image_shape)
    task_metrics = [Loss(), metrics.Accuracy()]
    exp = Experiment(
        train_task=Task("train", ArrayDataset(_arrays(32, image_shape), batch_size=8 * n_dev), task_metrics),
        eval_task=Task(
            "test",
            ArrayDataset(_arrays(16, image_shape), batch_size=8 * n_dev, shuffle=False, restart_each_iter=True),
            task_metrics,
        ),
        model_factory=model.build_model,
        forward_fn=model.forward,
        model_config=config,
        training_config=TrainingConfig(max_steps=6),
        eval_config=EvalConfig(eval_interval_steps=3, num_eval_steps=2),
        checkpoint_config=CheckpointConfig(save_interval_steps=3, checkpoint_dir=str(tmp_path / "ckpt")),
        logger_config=LoggerConfig(log_dir=str(tmp_path), run_id="p01"),
    )
    exp.init_state()
    exp.outer_loop()
    exp.cleanup()

    records = [json.loads(line) for line in (tmp_path / "logs_p01.txt").read_text().splitlines() if line]
    assert {"train/loss", "train/accuracy"} <= records[0].keys()
    assert any("val/accuracy" in r for r in records)

    assert inference.checkpoint_steps(tmp_path / "ckpt") == [0, 3]
    template = model.build_model(config, jax.random.key(1))
    restored = inference.load_model(template, tmp_path / "ckpt")
    logits = inference.batched_apply(lambda m, b: m(b["image"]), restored, _arrays(5, image_shape), 4)
    assert logits.shape == (5, 10)
