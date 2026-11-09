"""Train the ViT on Imagenette through the harness.

    uv run python -m toylib_projects.p03_vit.train --run smoke

``RUNS`` holds each configuration. ``notebooks/01_imagenette_vit_colab.ipynb`` trains
the real ones on Colab, with checkpoints in GCS and metrics on W&B.

The recipe is post 1's, so the ViT compares directly with its MLP and ResNet:
200 epochs at batch 128, AdamW at 1e-3 with cosine decay and weight decay 0.1, and a
random resized crop, a flip and a 40x40 cutout. Two changes suit a transformer: a
linear warmup, and gradient clipping at a global norm of 1. Weight decay also skips
parameters with fewer than two dimensions, such as biases and norm scales, as
nanoGPT does.
"""

import dataclasses
import pathlib
import tempfile
import time

import jax
import optax

from toylib.harness import analyze
from toylib.harness import inference
from toylib.harness.experiment import (
    CheckpointConfig,
    EvalConfig,
    Experiment,
    LoggerConfig,
    MultiOptimizerConfig,
    OptimizerConfig,
    Task,
    TrainingConfig,
    WandBLoggerConfig,
)
from toylib.harness.metrics import Loss
from toylib_projects.p03_vit import data
from toylib_projects.p03_vit import metrics
from toylib_projects.p03_vit import model

OUTPUT_DIR = pathlib.Path(__file__).parent / "outputs"


@dataclasses.dataclass
class Run:
    model_config: model.ModelConfig = dataclasses.field(
        default_factory=model.ModelConfig
    )
    epochs: int = 200
    batch_size: int = 128
    eval_batch_size: int = 400
    peak_lr: float = 1e-3
    warmup_epochs: int = 5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    crop_scale: tuple[float, float] = (0.35, 1.0)
    cutout_size: int = 40
    checkpoint_every_epochs: int = 10  # The most work a lost runtime can cost.
    limit: int | None = None  # Keep only the first N images of each split.


RUNS = {
    # The whole pipeline on 64 images, in seconds on a laptop.
    "smoke": Run(
        model_config=model.ModelConfig(dim=32, depth=2, num_heads=2, mlp_dim=64),
        epochs=2,
        batch_size=16,
        eval_batch_size=16,
        warmup_epochs=1,
        cutout_size=16,
        checkpoint_every_epochs=1,
        limit=64,
    ),
    # ViT-Ti/16's shape on post 1's recipe: 100 patches of 16x16.
    "vit_ti16": Run(),
}


def run_id(name: str, run: Run) -> str:
    """The name of the run's folder and W&B run. Reusing it resumes the run."""
    return f"{name}_e{run.epochs}"


def learning_rate(run: Run, steps_per_epoch: int) -> optax.Schedule:
    """A linear warmup to ``peak_lr``, then cosine decay to zero at the last step."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=run.peak_lr,
        warmup_steps=run.warmup_epochs * steps_per_epoch,
        decay_steps=run.epochs * steps_per_epoch,  # Includes the warmup.
        end_value=0.0,
    )


def decay_mask(params):
    """Decay matrices and kernels, and leave biases, norm scales and the like alone."""
    return jax.tree.map(lambda p: p.ndim >= 2, params)


def optimizer_config(run: Run, steps_per_epoch: int) -> MultiOptimizerConfig:
    adamw = optax.adamw(
        learning_rate(run, steps_per_epoch),
        weight_decay=run.weight_decay,
        mask=decay_mask,
    )
    return MultiOptimizerConfig(
        [OptimizerConfig("adamw", adamw)], optimizer_for_param=lambda path: "adamw"
    )


def make_experiment(
    name: str,
    *,
    run: Run | None = None,
    checkpoint_dir: str | None = None,
    logger_config: LoggerConfig | None = None,
    output_dir: pathlib.Path | None = None,
    load: data.Loader | None = None,
) -> Experiment:
    """The experiment for ``RUNS[name]``, not yet initialized.

    Logs and checkpoints go under ``output_dir / run_id`` unless ``checkpoint_dir``,
    which can be a gs:// path, or ``logger_config`` say otherwise. ``run`` replaces
    the configuration, and ``load`` the data, for tests.
    """
    run = run or RUNS[name]
    train_ds, test_ds = data.make_datasets(
        batch_size=run.batch_size,
        eval_batch_size=run.eval_batch_size,
        crop_scale=run.crop_scale,
        cutout_size=run.cutout_size,
        limit=run.limit,
        load=load,
    )
    steps_per_epoch = train_ds.batches_per_epoch
    total_steps = run.epochs * steps_per_epoch
    rid = run_id(name, run)
    run_dir = (pathlib.Path(output_dir or OUTPUT_DIR) / rid).resolve()

    task_metrics = [Loss(), metrics.Accuracy()]
    return Experiment(
        train_task=Task("train", train_ds, metrics=task_metrics),
        eval_task=Task("test", test_ds, metrics=task_metrics),
        model_factory=model.build_model,
        forward_fn=model.forward,
        model_config=run.model_config,
        # +1: the harness evaluates and saves a step before counting it, so this
        # makes the last epoch's evaluation and checkpoint happen.
        training_config=TrainingConfig(
            optimizer_config=optimizer_config(run, steps_per_epoch),
            max_steps=total_steps + 1,
            max_grad_norm=run.max_grad_norm,
        ),
        eval_config=EvalConfig(
            eval_interval_steps=steps_per_epoch,
            num_eval_steps=test_ds.batches_per_epoch,
        ),
        checkpoint_config=CheckpointConfig(
            save_interval_steps=run.checkpoint_every_epochs * steps_per_epoch,
            max_to_keep=None,  # Small, and useful for figures over training.
            checkpoint_dir=checkpoint_dir or str(run_dir / "checkpoints"),
            checkpoint_dataset_iterator=True,  # Resume mid-epoch in the same order.
        ),
        logger_config=logger_config
        or LoggerConfig(log_dir=str(run_dir), run_id=rid, train_log_interval_steps=10),
    )


def wandb_logger(
    name: str, project: str, entity: str = "", output_dir: pathlib.Path | None = None
) -> WandBLoggerConfig:
    """Log to W&B and to a local file. A repeat of the run resumes its W&B run."""
    rid = run_id(name, RUNS[name])
    return WandBLoggerConfig(
        project_name=project,
        user_name=entity,
        log_dir=str((pathlib.Path(output_dir or OUTPUT_DIR) / rid).resolve()),
        run_id=rid,
        train_log_interval_steps=50,
        also_log_to_file=True,
    )


def full_test_accuracy(exp: Experiment, load: data.Loader | None = None) -> float:
    """Accuracy over every test image. The harness's evaluation drops a partial batch."""
    arrays = data.evaluation_arrays(exp.eval_task.dataset.num_examples, load)
    logits = inference.batched_apply(
        lambda m, b: m(b["image"]),
        exp.model,
        {"image": arrays["image"]},
        exp.eval_task.dataset.batch_size,
    )
    return float((logits.argmax(-1) == arrays["label"]).mean())


def time_run(name: str, steps: int = 50, **kwargs) -> dict[str, float]:
    """Time ``steps`` training steps and one evaluation, saving nothing.

    The steps are the run's real size, so the estimate covers the whole run with an
    evaluation each epoch. It also proves the model, sharding and evaluation work on
    this machine before hours go into them. ``kwargs`` go to ``make_experiment``.
    """
    run = kwargs.get("run") or RUNS[name]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        exp = make_experiment(
            name,
            logger_config=LoggerConfig(log_dir=str(tmp), run_id="timing"),
            checkpoint_dir=str(tmp / "checkpoints"),
            **kwargs,
        )
        def forever():
            while True:  # One pass over the dataset is one epoch.
                yield from exp.train_task.dataset

        try:
            exp.init_state()
            batches = forever()
            exp.inner_loop(next(batches))  # Compiles the training step.
            jax.block_until_ready(exp.model)

            start = time.monotonic()
            for _ in range(steps):
                exp.inner_loop(next(batches))
            jax.block_until_ready(exp.model)
            seconds_per_step = (time.monotonic() - start) / steps

            exp.run_validation()  # Compiles the evaluation step.
            start = time.monotonic()
            exp.run_validation()
            seconds_per_eval = time.monotonic() - start

            params = analyze.print_param_sizes(exp.model, depth=1)[1]
            steps_total = exp.training_config.max_steps
        finally:
            exp.cleanup()
    hours = (steps_total * seconds_per_step + run.epochs * seconds_per_eval) / 3600
    return {
        "params": params,
        "steps_per_second": 1 / seconds_per_step,
        "seconds_per_eval": seconds_per_eval,
        "hours": hours,
    }


def train(name: str, load: data.Loader | None = None, **kwargs) -> dict[str, float]:
    """Train ``RUNS[name]`` to the end, from its newest checkpoint if it has one.

    A finished run only evaluates. Returns the final evaluation, plus the accuracy
    over every test image. ``kwargs`` go to ``make_experiment``.
    """
    exp = make_experiment(name, load=load, **kwargs)
    try:
        exp.init_state()
        resumed = exp.resume()
        print(
            f"{name}: "
            + (f"resuming at step {exp.step}" if resumed else "starting fresh")
            + f" of {exp.training_config.max_steps}"
        )
        exp.outer_loop()
        results = exp.run_validation()
        results["full_test_accuracy"] = full_test_accuracy(exp, load)
    finally:
        exp.cleanup()  # Waits for W&B to finish syncing.
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", choices=sorted(RUNS), default="smoke")
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", help="Can be gs://. Default: under outputs/")
    parser.add_argument("--wandb-project", help="Default: log to a file only")
    parser.add_argument("--wandb-entity", default="")
    args = parser.parse_args()

    logger_config = None
    if args.wandb_project:
        logger_config = wandb_logger(
            args.run, args.wandb_project, args.wandb_entity, args.output_dir
        )
    results = train(
        args.run,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        logger_config=logger_config,
    )
    print(results)
