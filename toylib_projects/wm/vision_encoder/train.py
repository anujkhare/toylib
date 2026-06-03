"""Training entry point for the Track A1 KL-VAE vision codec.

Mirrors the shape of ``toylib_projects/tinystories/train.py``: a
``create_experiment`` factory that wires together the dataset, model,
optimizer, logger, and checkpoint manager into a ready-to-run
``Experiment``, plus a ``main`` that exposes a small CLI.

Usage (interactive / Colab)::

    from vision_encoder import train

    exp = train.create_experiment(
        train_path="data/compiled/vae_train.h5",
        val_path="data/compiled/vae_val.h5",
        max_steps=1000,
        batch_size_per_device=16,
    )
    exp.outer_loop()
    exp.cleanup()

Usage (CLI)::

    uv run python -m vision_encoder.train \\
        --train-path data/compiled/vae_train.h5 \\
        --val-path   data/compiled/vae_val.h5 \\
        --max-steps  100 \\
        --batch-size-per-device 8


Design notes
------------

**Normalization lives in ``forward_fn``.** The compiled VAE dataset stores
uint8 frames in ``[0, 255]``. The VAE consumes float32 in ``[-1, 1]``. We
convert inside the forward function (a tiny one-liner) so the loader stays
pure and the same conversion applies in eval/inference. The cost is a
single mul/sub op per batch.

**Reparameterization RNG.** The base ``Experiment._train_step`` doesn't
thread an RNG into ``forward_fn`` — the only inputs are ``(model,
opt_state, batch)``. For this v1 we therefore call ``vae_loss`` with a
**fixed** ``rng_key`` per step. Practical implications:

  - Gradients still flow through the reparameterization trick (it's
    differentiable in μ and log σ²).
  - The noise sample ε is the same on every step. The encoder is being
    asked to fit a deterministic posterior — closer to a plain AE than a
    true VAE.
  - For smoke-testing the harness end-to-end this is fine; for a real
    training run, thread a per-step key through ``Experiment`` (TODO
    below) before drawing conclusions about VAE behavior.

**β warmup is not yet plumbed through training.** ``vae_loss`` accepts a
``beta`` argument and ``beta_warmup`` computes the right schedule, but
``Experiment._train_step`` doesn't know about the current step
inside the JIT'd compute. For v1 we pass a constant ``beta``. Hooking up
the schedule cleanly requires either passing ``step`` into ``forward_fn``
or building it into a stateful model — TODO.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import typing
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from . import experiment as exp_lib
from . import metrics as metrics_lib
from .dataloader import Hdf5FramesDataset
from .model import ModelConfig, VAE, vae_loss


# ──────────────────────────────────────────────────────────────────────────
# Glue: model factory, forward_fn, metric
# ──────────────────────────────────────────────────────────────────────────


def make_model_factory() -> typing.Callable:
    """Return a factory that builds an initialized VAE from (config, key)."""

    def factory(config: ModelConfig, key) -> VAE:
        vae = VAE(config=config, key=key)
        vae.init()
        return vae

    return factory


def make_forward_fn(beta: float, rng_seed: int = 0) -> typing.Callable:
    """Build a ``(model, batch) -> (loss, aux)`` closure for ``Experiment``.

    Normalizes uint8 frames to ``[-1, 1]`` float32 before calling
    ``vae_loss``. ``beta`` is the constant KL weight applied at every step
    (see the module docstring for why we don't yet do warmup here).
    """
    fixed_key = jax.random.key(rng_seed)

    def forward_fn(model: VAE, batch):
        # batch shape: (B, H, W, 3) uint8. Normalize to (B, H, W, 3) float32 in [-1, 1].
        frames = batch.astype(jnp.float32) / 127.5 - 1.0
        loss, aux = vae_loss(model, frames, rng_key=fixed_key, beta=beta)
        # Only return the scalar pieces the metric/log layer cares about —
        # passing the full recon/mu/log_sigma_sq tensors through the metric
        # accumulator would explode batch * shape memory needlessly.
        return loss, {"l_rec": aux["l_rec"], "l_kl": aux["l_kl"]}

    return forward_fn


@dataclasses.dataclass
class VaeAuxMetric:
    """Surface the per-step scalar VAE losses (``l_rec``, ``l_kl``) as named metrics."""

    keys: tuple[str, ...] = ("l_rec", "l_kl")

    def __call__(self, loss, aux, batch):
        del loss, batch
        return {k: aux[k] for k in self.keys if k in aux}


# ──────────────────────────────────────────────────────────────────────────
# Experiment construction
# ──────────────────────────────────────────────────────────────────────────


def create_experiment(
    train_path: str | Path,
    val_path: str | Path | None = None,
    *,
    # Model
    base_ch: int = 64,
    latent_channels: int = 4,
    # Training
    batch_size_per_device: int = 16,
    num_microbatches: int = 1,
    max_steps: int = 1000,
    learning_rate: float = 1e-4,
    max_grad_norm: float = 1.0,
    beta: float = 1e-6,
    # Eval / checkpoint / logging
    eval_interval_steps: int = 100,
    num_eval_steps: int = 4,
    save_interval_steps: int = 1000,
    checkpoint_dir: str = "/tmp/wm_vae_ckpt",
    log_dir: str = "/tmp/wm_vae_logs",
    run_id: str | None = None,
    # Logger selection: if ``wandb_project`` is set, log to W&B; otherwise
    # fall back to the local JSONL FileLogger. Smoke tests and CI typically
    # leave ``wandb_project`` unset.
    wandb_project: str | None = None,
    wandb_user: str | None = None,
    seed: int = 0,
    jit_computations: bool = True,
) -> exp_lib.Experiment:
    """Wire datasets, model, optimizer, logger and checkpointer into an Experiment."""
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    n_devices = jax.local_device_count()
    batch_size = batch_size_per_device * n_devices * num_microbatches

    # ── Datasets ────────────────────────────────────────────────────────
    train_ds = Hdf5FramesDataset(
        dataset_path=str(train_path), batch_size=batch_size,
        seed=seed, shuffle=True, drop_remainder=True,
    )
    train_task = exp_lib.Task(
        name="train",
        dataset=train_ds,
        metrics=[metrics_lib.Loss(), VaeAuxMetric()],
    )
    eval_task = None
    if val_path is not None:
        val_ds = Hdf5FramesDataset(
            dataset_path=str(val_path), batch_size=batch_size,
            seed=seed, shuffle=False, drop_remainder=True,
        )
        eval_task = exp_lib.Task(
            name="val",
            dataset=val_ds,
            metrics=[metrics_lib.Loss(), VaeAuxMetric()],
        )

    # ── Optimizer ───────────────────────────────────────────────────────
    # Single Adam at fixed LR per the walkthrough, with global-norm clipping.
    optimizer_config = exp_lib.MultiOptimizerConfig(
        optimizer_configs=[
            exp_lib.OptimizerConfig(
                name="adam_all",
                optimizer=optax.adam(learning_rate=learning_rate),
            ),
        ],
        optimizer_for_param=lambda path: "adam_all",
    )

    # ── Logger ──────────────────────────────────────────────────────────
    if wandb_project is not None:
        if wandb_user is None:
            raise ValueError("--wandb-user is required when --wandb-project is set")
        logger_config = exp_lib.WandBLoggerConfig(
            project_name=wandb_project,
            user_name=wandb_user,
            log_dir=log_dir,
            run_id=run_id,
        )
    else:
        logger_config = exp_lib.LoggerConfig(log_dir=log_dir, run_id=run_id)

    # ── Experiment ──────────────────────────────────────────────────────
    return exp_lib.Experiment(
        train_task=train_task,
        eval_task=eval_task,
        forward_fn=make_forward_fn(beta=beta),
        model_factory=make_model_factory(),
        model_config=ModelConfig(
            base_ch=base_ch, latent_channels=latent_channels,
        ),
        training_config=exp_lib.TrainingConfig(
            max_steps=max_steps,
            num_microbatches=num_microbatches,
            max_grad_norm=max_grad_norm,
            optimizer_config=optimizer_config,
        ),
        eval_config=exp_lib.EvalConfig(
            eval_interval_steps=eval_interval_steps,
            num_eval_steps=num_eval_steps,
        ),
        checkpoint_config=exp_lib.CheckpointConfig(
            save_interval_steps=save_interval_steps,
            checkpoint_dir=f"{checkpoint_dir}/{run_id}",
        ),
        logger_config=logger_config,
        seed=seed,
        jit_computations=jit_computations,
    )


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def main() -> exp_lib.Experiment:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, required=True,
                        help="Path to compiled vae_train.h5 file.")
    parser.add_argument("--val-path", type=Path, default=None,
                        help="Path to compiled vae_val.h5 file (optional).")
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--batch-size-per-device", type=int, default=16)
    parser.add_argument("--num-microbatches", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1e-6)
    parser.add_argument("--eval-interval-steps", type=int, default=100)
    parser.add_argument("--num-eval-steps", type=int, default=4)
    parser.add_argument("--save-interval-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", default="/tmp/wm_vae_ckpt")
    parser.add_argument("--log-dir", default="/tmp/wm_vae_logs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--wandb-project", default=None,
        help="W&B project name. If set, metrics go to wandb instead of a local file.",
    )
    parser.add_argument(
        "--wandb-user", default=None,
        help="W&B entity/username. Required when --wandb-project is set.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-jit", action="store_true",
                        help="Disable JIT (useful for debugging with pdb).")
    args = parser.parse_args()

    exp = create_experiment(
        train_path=args.train_path,
        val_path=args.val_path,
        base_ch=args.base_ch,
        latent_channels=args.latent_channels,
        batch_size_per_device=args.batch_size_per_device,
        num_microbatches=args.num_microbatches,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        beta=args.beta,
        eval_interval_steps=args.eval_interval_steps,
        num_eval_steps=args.num_eval_steps,
        save_interval_steps=args.save_interval_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        run_id=args.run_id,
        wandb_project=args.wandb_project,
        wandb_user=args.wandb_user,
        seed=args.seed,
        jit_computations=not args.no_jit,
    )
    exp.init_state()
    try:
        exp.outer_loop()
    finally:
        exp.cleanup()
    return exp


if __name__ == "__main__":
    main()
