"""The training wiring, end to end, with the stand-in model and synthetic data.

These pass before the ViT exists: they check the schedule, the optimizer,
checkpoints and resuming, not what the model learns.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from toylib.harness import inference
from toylib_projects.p03_vit import data
from toylib_projects.p03_vit import model
from toylib_projects.p03_vit import testing
from toylib_projects.p03_vit import train

TINY = dataclasses.replace(
    train.RUNS["smoke"], epochs=2, batch_size=8, eval_batch_size=8, limit=None
)
LOAD = testing.SyntheticImagenette(n_train=32, n_test=20)


@pytest.fixture(autouse=True)
def stand_in(monkeypatch):
    testing.use_stand_in(monkeypatch)


def test_runs_are_consistent():
    for name, run in train.RUNS.items():
        assert run.epochs % run.checkpoint_every_epochs == 0, name
        assert run.warmup_epochs < run.epochs, name
        assert run.model_config.image_shape[0] % run.model_config.patch_size == 0, name
        assert run.model_config.dim % run.model_config.num_heads == 0, name


def test_learning_rate_warms_up_then_decays_to_zero():
    run = dataclasses.replace(TINY, epochs=10, warmup_epochs=2, peak_lr=1e-3)
    schedule = train.learning_rate(run, steps_per_epoch=5)
    assert float(schedule(0)) == 0.0
    assert float(schedule(5)) == pytest.approx(5e-4)
    assert float(schedule(10)) == pytest.approx(1e-3)
    assert float(schedule(30)) < 1e-3
    assert float(schedule(50)) == pytest.approx(0.0, abs=1e-12)


def test_weight_decay_skips_vectors():
    params = {"matrix": jnp.ones((2, 2)), "bias": jnp.ones(2)}
    zeros = jax.tree.map(jnp.zeros_like, params)
    run = dataclasses.replace(TINY, weight_decay=0.5, warmup_epochs=0, peak_lr=0.1)
    optimizer = train.optimizer_config(run, steps_per_epoch=10).optimizer_configs[0].optimizer

    updates, _ = optimizer.update(zeros, optimizer.init(params), params)
    assert float(updates["matrix"][0, 0]) == pytest.approx(-0.1 * 0.5)
    assert float(updates["bias"][0]) == 0.0


def test_train_saves_logs_and_evaluates(tmp_path):
    results = train.train("smoke", run=TINY, load=LOAD, output_dir=tmp_path)

    assert np.isfinite(results["val/loss"])
    assert 0.0 <= results["full_test_accuracy"] <= 1.0
    run_dir = tmp_path / train.run_id("smoke", TINY)
    steps = inference.checkpoint_steps(run_dir / "checkpoints")
    # 32 images in batches of 8 is 4 steps an epoch: a checkpoint every epoch.
    assert steps == [0, 4, 8]

    log = (run_dir / "logs_smoke_e2.txt").read_text().splitlines()
    lines = [json.loads(line) for line in log if line]
    assert any("train/accuracy" in line for line in lines)
    assert any("val/accuracy" in line for line in lines)


def test_full_test_accuracy_counts_every_image(tmp_path):
    """The harness drops a partial batch, but the final accuracy doesn't."""
    exp = train.make_experiment("smoke", run=TINY, load=LOAD, output_dir=tmp_path)
    exp.init_state()
    try:
        net = exp.model
        arrays = data.evaluation_arrays(load=LOAD)
        expected = float((np.asarray(net(arrays["image"])).argmax(-1) == arrays["label"]).mean())
        assert exp.eval_task.dataset.batches_per_epoch * 8 < 20
        assert train.full_test_accuracy(exp, LOAD) == pytest.approx(expected)
    finally:
        exp.cleanup()


def test_a_finished_run_resumes_and_only_evaluates(tmp_path):
    first = train.train("smoke", run=TINY, load=LOAD, output_dir=tmp_path)

    exp = train.make_experiment("smoke", run=TINY, load=LOAD, output_dir=tmp_path)
    exp.init_state()
    try:
        assert exp.resume()
        assert exp.step == exp.training_config.max_steps
        assert exp.run_validation()["val/loss"] == pytest.approx(first["val/loss"])
    finally:
        exp.cleanup()


def test_resume_continues_a_stopped_run(tmp_path):
    longer = dataclasses.replace(TINY, epochs=4, checkpoint_every_epochs=2)
    exp = train.make_experiment("smoke", run=longer, load=LOAD, output_dir=tmp_path)
    exp.init_state()
    exp.training_config.max_steps = 9  # Stop after the checkpoint at step 8.
    exp.outer_loop()
    exp.cleanup()

    exp = train.make_experiment("smoke", run=longer, load=LOAD, output_dir=tmp_path)
    exp.init_state()
    try:
        assert exp.resume()
        assert exp.step == 9
        exp.outer_loop()
        assert exp.step == exp.training_config.max_steps == 17
    finally:
        exp.cleanup()
    assert inference.checkpoint_steps(tmp_path / "smoke_e4" / "checkpoints") == [0, 8, 16]


def test_time_run_estimates_hours():
    timing = train.time_run("smoke", steps=6, run=TINY, load=LOAD)  # Past an epoch.
    assert timing["params"] == 3 * 10 + 10  # The stand-in's one linear layer.
    assert timing["steps_per_second"] > 0
    assert timing["hours"] > 0


def test_the_vit_is_what_trains_by_default(monkeypatch, tmp_path):
    """Outside these tests, the experiment builds ``model.build_model``."""
    monkeypatch.undo()
    exp = train.make_experiment("smoke", run=TINY, load=LOAD, output_dir=tmp_path)
    try:
        assert exp.model_factory is model.build_model
        assert exp.forward_fn is model.forward
    finally:
        exp.cleanup()
