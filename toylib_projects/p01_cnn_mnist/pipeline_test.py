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


def test_cutout_blanks_one_square_per_image():
    images = np.ones((200, 32, 32, 3), np.float32)
    out = data.cutout({"image": images, "label": np.zeros(200)}, np.random.default_rng(0))
    hidden = out["image"][..., 0] == 0

    assert out["image"].dtype == np.float32
    assert (out["image"] == out["image"][..., :1]).all()  # All channels together.
    assert hidden.sum(axis=(1, 2)).max() == 16 * 16
    assert hidden.sum(axis=(1, 2)).min() > 0
    for mask in hidden:  # Hidden pixels fill their bounding box: one rectangle.
        rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
        assert mask.sum() == len(rows) * len(cols)
        assert np.ptp(rows) == len(rows) - 1 and np.ptp(cols) == len(cols) - 1
    # Centres vary, and a square can hang off the edge.
    assert len({tuple(m.ravel()) for m in hidden}) > 100
    assert (hidden[:, 0, :].any(axis=1)).any() and (hidden[:, -1, :].any(axis=1)).any()


def test_cutout_size_needs_augment():
    with pytest.raises(ValueError, match="augment"):
        data.make_datasets("cifar10", batch_size=8, cutout_size=16)


def test_random_flip_and_crop():
    images = np.random.default_rng(0).normal(size=(64, 32, 32, 3)).astype(np.float32)
    batch = {"image": images, "label": np.arange(64)}

    out = data.random_flip_and_crop(batch, np.random.default_rng(1), pad=4)
    assert out["image"].shape == images.shape and out["image"].dtype == images.dtype
    np.testing.assert_array_equal(out["label"], batch["label"])

    # With no padding the only change is the flip: each image is itself or its mirror.
    flipped = data.random_flip_and_crop(batch, np.random.default_rng(1), pad=0)["image"]
    same = (flipped == images).all(axis=(1, 2, 3))
    mirrored = (flipped == images[:, :, ::-1]).all(axis=(1, 2, 3))
    assert (same | mirrored).all() and same.any() and mirrored.any()

    # Pixels encode their position, so each crop must be one solid window of the
    # image, shifted by at most ``pad``, with zeros only in the padding.
    rows, cols = np.mgrid[:32, :32]
    coded = np.tile((1 + 100 * rows + cols).astype(np.float32)[None, :, :, None], (64, 1, 1, 1))
    crops = data.random_flip_and_crop({"image": coded}, np.random.default_rng(2), pad=4)["image"]
    for crop in crops[..., 0]:
        r, c = np.nonzero(crop)
        window = crop[r.min() : r.max() + 1, c.min() : c.max() + 1]
        assert min(window.shape) >= 28 and (window != 0).all()
        assert (np.diff(window, axis=0) == 100).all()
        assert (np.abs(np.diff(window, axis=1)) == 1).all()
    assert len({tuple(np.argwhere(crop[..., 0])[0]) for crop in crops}) > 1


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
