"""Training entry point for the latent state probe.

Trains a small MLP (``probe/model.py``) on top of a **frozen** VAE encoder to
regress per-frame state targets (ball_x, ball_y, paddle_x) from the encoder's
latent. It's a diagnostic: if the probe predicts ball position well, the
encoder is storing it and any reconstruction failure is the decoder's fault; if
the probe fails, the encoder never captured the ball in the first place.

Reuses the shared ``Experiment`` harness. Two probe-specific wrinkles:

  - **Encoder weights come from a VAE checkpoint.** ``--vae-checkpoint-dir`` /
    ``--vae-checkpoint-step`` point at a checkpoint written by
    ``vision_encoder.train``. ``load_vae`` restores it; the probe factory grafts
    its encoder into a fresh ``MLPProbe`` (random MLP head).

  - **The encoder is frozen at the optimizer level.** The optimizer is a
    multi-transform that maps the whole ``encoder`` sub-tree to
    ``optax.set_to_zero`` and the MLP head (``fc1`` / ``fc2``) to Adam. The
    encoder still receives gradients in the backward pass, but the optimizer
    drops its updates, so its weights never change. This keeps "what is
    trainable" a single, inspectable decision rather than a ``stop_gradient``
    buried in the model.

Usage (CLI)::

    uv run python -m probe.train \\
        --train-path data/compiled/vae_train.h5 \\
        --val-path   data/compiled/vae_val.h5 \\
        --vae-checkpoint-dir /tmp/wm_vae_ckpt/<run_id> \\
        --vae-checkpoint-step 1000 \\
        --max-steps 2000 --batch-size-per-device 32
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import typing
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

from toylib.nn import module as module_lib
from toylib_projects.wm import dataloader as dataloader_lib
from toylib_projects.wm import experiment as exp_lib
from toylib_projects.wm import metrics as metrics_lib
from toylib_projects.wm.probe import model as probe_model
from toylib_projects.wm.vision_encoder import inference as ve_inference
from toylib_projects.wm.vision_encoder import model as vae_model

# Order matters: it defines the column order of the dataset's ``targets`` array
# and therefore which MLP output predicts which quantity.
TARGET_KEYS: tuple[str, ...] = ("ball_x", "ball_y", "paddle_x")

# RAM state values are single bytes (0–255). Dividing by this maps targets into
# roughly [0, 1] so MSE is on a sane scale; multiply errors back by it to report
# them in original RAM units.
TARGET_SCALE: float = 255.0


# Floor on each target's normalized variance, used as the R² denominator. Guards
# against divide-by-zero for a (degenerate) constant target.
_VAR_EPS: float = 1e-8


# ──────────────────────────────────────────────────────────────────────────
# Target statistics (for R²)
# ──────────────────────────────────────────────────────────────────────────


def target_variance(path: str | Path, keys: tuple[str, ...]) -> np.ndarray:
    """Per-target variance of the labels, on the **normalized** scale.

    R² compares the probe's error to the error of always predicting the target
    mean (whose MSE *is* the variance). We read the variance once from the
    training labels so R² can be computed cheaply inside the jitted forward pass.
    The values are divided by ``TARGET_SCALE**2`` to match the normalized targets
    used in the loss.
    """
    with h5py.File(path, "r") as f:
        src = f["source"]
        cols = [np.asarray(src[k][:], dtype=np.float64) for k in keys]
    raw = np.stack(cols, axis=1)  # (N, K)
    var = raw.var(axis=0) / (TARGET_SCALE**2)
    return np.maximum(var, _VAR_EPS).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────


def make_probe_factory(
    vae_config: vae_model.ModelConfig,
    vae_checkpoint_dir: str | None,
    vae_checkpoint_step: int | None,
    encoder_type: probe_model.EncoderType = probe_model.EncoderType.VAE,
) -> typing.Callable:
    """Return a ``(ProbeConfig, key) -> MLPProbe`` factory.

    The encoder feeding the probe head depends on ``encoder_type``:

      - ``VAE`` with a checkpoint: load the pretrained VAE encoder.
      - ``VAE`` with ``vae_checkpoint_dir=None``: a **randomly initialized**
        encoder — a smoke test / weak baseline (random features should probe
        *worse* than trained ones).
      - ``PASSTHROUGH``: an ``IdentityEncoder`` that feeds raw pixels to the
        head — the upper-bound baseline (ignores the VAE entirely).

    In every case the MLP head is freshly (randomly) initialized.
    """

    def factory(config: probe_model.ProbeConfig, key) -> probe_model.MLPProbe:
        vae_key, mlp_key = jax.random.split(key, 2)

        if encoder_type is probe_model.EncoderType.PASSTHROUGH:
            encoder: module_lib.Module = probe_model.IdentityEncoder()
            encoder.init()
        elif vae_checkpoint_dir is None:
            vae = vae_model.VAE(config=vae_config, key=vae_key)
            vae.init()
            encoder = jax.tree.map(jnp.asarray, vae.encoder)
        else:
            if vae_checkpoint_step is None:
                raise ValueError(
                    "vae_checkpoint_step is required when vae_checkpoint_dir is set"
                )
            vae = ve_inference.load_vae(
                vae_checkpoint_dir, vae_checkpoint_step, config=vae_config, key=vae_key
            )
            # Bring the restored (host) arrays onto the device/mesh in effect.
            encoder = jax.tree.map(jnp.asarray, vae.encoder)

        probe = probe_model.MLPProbe(config=config, encoder=encoder, key=mlp_key)
        probe.init()
        return probe

    return factory


# ──────────────────────────────────────────────────────────────────────────
# Glue: forward_fn + metric
# ──────────────────────────────────────────────────────────────────────────


def make_forward_fn(target_var: np.ndarray | None = None) -> typing.Callable:
    """Build a ``(model, batch) -> (loss, aux)`` closure.

    ``batch`` is ``{"frames": (B, H, W, 3) uint8, "targets": (B, K) float32}``
    in RAM coordinates. Frames are mapped to ``[-1, 1]`` and targets to ``~[0,1]``
    before an MSE regression loss. ``aux`` carries, per target:

      - ``mse_{name}``    — MSE on the normalized scale.
      - ``mae_raw_{name}``— MAE back in RAM units (multiply by ``TARGET_SCALE``).
      - ``r2_{name}``     — R² = 1 − MSE / Var(target), if ``target_var`` is given.

    R² is the interpretable signal: ≈0 means the latent carries no recoverable
    info about that target (no better than predicting the mean); →1 means it is
    fully recoverable. ``target_var`` is the per-target variance on the
    normalized scale (see ``target_variance``); it is closed over as a constant
    so R² stays cheap inside the jitted step.
    """
    var = None if target_var is None else jnp.asarray(target_var)

    def forward_fn(model: probe_model.MLPProbe, batch):
        frames = batch["frames"].astype(jnp.float32) / 127.5 - 1.0
        targets = batch["targets"].astype(jnp.float32) / TARGET_SCALE

        pred = model(frames)  # (B, K)
        err = pred - targets
        se = err**2

        loss = jnp.mean(se)
        aux = {}
        for i, name in enumerate(TARGET_KEYS):
            mse_i = jnp.mean(se[:, i])
            aux[f"mse_{name}"] = mse_i
            aux[f"mae_raw_{name}"] = jnp.mean(jnp.abs(err[:, i])) * TARGET_SCALE
            if var is not None:
                aux[f"r2_{name}"] = 1.0 - mse_i / var[i]
        return loss, aux

    return forward_fn


@dataclasses.dataclass
class ProbeAuxMetric:
    """Surface the per-target probe errors (MSE + RAM-unit MAE) as named metrics."""

    keys: tuple[str, ...] = tuple(
        [f"mse_{k}" for k in TARGET_KEYS]
        + [f"mae_raw_{k}" for k in TARGET_KEYS]
        + [f"r2_{k}" for k in TARGET_KEYS]
    )

    def __call__(self, loss, aux, batch):
        del loss, batch
        return {k: aux[k] for k in self.keys if k in aux}


# ──────────────────────────────────────────────────────────────────────────
# Experiment construction
# ──────────────────────────────────────────────────────────────────────────


def create_experiment(
    train_path: str | Path,
    val_path: str | Path | None,
    *,
    encoder_type: probe_model.EncoderType = probe_model.EncoderType.VAE,
    vae_checkpoint_dir: str | None = None,
    vae_checkpoint_step: int | None = None,
    # VAE encoder shape (must match the checkpoint). Ignored for PASSTHROUGH,
    # where the latent dims are taken from the raw frame shape instead.
    base_ch: int = 64,
    latent_channels: int = 4,
    latent_spatial: int = 16,
    # Probe head.
    hidden_dim: int = 256,
    pooling: probe_model.Pooling = probe_model.Pooling.FLATTEN,
    # Training.
    batch_size_per_device: int = 32,
    num_microbatches: int = 1,
    max_steps: int = 2000,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    # Eval / checkpoint / logging.
    eval_interval_steps: int = 100,
    num_eval_steps: int = 4,
    save_interval_steps: int = 1000,
    checkpoint_dir: str = "/tmp/wm_probe_ckpt",
    log_dir: str = "/tmp/wm_probe_logs",
    run_id: str | None = None,
    wandb_project: str | None = None,
    wandb_user: str | None = None,
    seed: int = 0,
    jit_computations: bool = True,
) -> exp_lib.Experiment:
    """Wire labelled datasets, the frozen-encoder probe, and the optimizer."""
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    n_devices = jax.local_device_count()
    batch_size = batch_size_per_device * n_devices * num_microbatches

    # ── Datasets (labelled) ──────────────────────────────────────────────
    train_ds = dataloader_lib.Hdf5FramesDataset(
        dataset_path=str(train_path),
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        drop_remainder=True,
        repeat=True,
        label_keys=TARGET_KEYS,
    )
    train_task = exp_lib.Task(
        name="train",
        dataset=train_ds,
        metrics=[metrics_lib.Loss(), ProbeAuxMetric()],
    )

    # For the pass-through baseline the "latent" *is* the image, so its feature
    # dims come from the frame shape rather than the (unused) VAE config.
    if encoder_type is probe_model.EncoderType.PASSTHROUGH:
        frame_h, frame_w, frame_c = train_ds.frame_shape
        if frame_h != frame_w:
            raise ValueError(
                f"pass-through probe assumes square frames, got {frame_h}x{frame_w}"
            )
        latent_spatial, latent_channels = frame_h, frame_c

    # Per-target variance → the R² denominator (computed from the train labels).
    target_var = target_variance(train_path, TARGET_KEYS)
    eval_task = None
    if val_path is not None:
        val_ds = dataloader_lib.Hdf5FramesDataset(
            dataset_path=str(val_path),
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
            drop_remainder=False,
            repeat=True,
            label_keys=TARGET_KEYS,
        )
        eval_task = exp_lib.Task(
            name="val",
            dataset=val_ds,
            metrics=[metrics_lib.Loss(), ProbeAuxMetric()],
        )

    # ── Optimizer: freeze the encoder, train the MLP head ────────────────
    # The probe pytree's top-level fields are ``encoder``, ``fc1``, ``fc2``;
    # routing the ``encoder`` sub-tree to ``set_to_zero`` drops its updates.
    optimizer_config = exp_lib.MultiOptimizerConfig(
        optimizer_configs=[
            exp_lib.OptimizerConfig(name="frozen", optimizer=optax.set_to_zero()),
            exp_lib.OptimizerConfig(
                name="trainable", optimizer=optax.adam(learning_rate=learning_rate)
            ),
        ],
        optimizer_for_param=lambda path: (
            "frozen" if path[0].name == "encoder" else "trainable"
        ),
    )

    # ── Logger ───────────────────────────────────────────────────────────
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

    vae_config = vae_model.ModelConfig(base_ch=base_ch, latent_channels=latent_channels)
    probe_config = probe_model.ProbeConfig(
        latent_channels=latent_channels,
        latent_spatial=latent_spatial,
        hidden_dim=hidden_dim,
        num_targets=len(TARGET_KEYS),
        pooling=pooling,
    )

    return exp_lib.Experiment(
        train_task=train_task,
        eval_task=eval_task,
        forward_fn=make_forward_fn(target_var=target_var),
        model_factory=make_probe_factory(
            vae_config=vae_config,
            vae_checkpoint_dir=vae_checkpoint_dir,
            vae_checkpoint_step=vae_checkpoint_step,
            encoder_type=encoder_type,
        ),
        model_config=probe_config,
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
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, default=None)
    parser.add_argument(
        "--encoder-type",
        type=probe_model.EncoderType,
        default=probe_model.EncoderType.VAE,
        choices=list(probe_model.EncoderType),
        help=(
            "vae: probe the (frozen) VAE latent. "
            "passthrough: probe raw pixels directly (upper-bound baseline; "
            "ignores --vae-checkpoint-dir)."
        ),
    )
    parser.add_argument(
        "--vae-checkpoint-dir",
        default=None,
        help=(
            "Local path or gs://bucket/... URI of the VAE checkpoint. "
            "Omit to train against a randomly-initialized encoder (smoke test)."
        ),
    )
    parser.add_argument("--vae-checkpoint-step", type=int, default=None)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-spatial", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--pooling",
        type=probe_model.Pooling,
        default=probe_model.Pooling.FLATTEN,
        choices=list(probe_model.Pooling),
    )
    parser.add_argument("--batch-size-per-device", type=int, default=32)
    parser.add_argument("--num-microbatches", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-interval-steps", type=int, default=100)
    parser.add_argument("--num-eval-steps", type=int, default=4)
    parser.add_argument("--save-interval-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", default="/tmp/wm_probe_ckpt")
    parser.add_argument("--log-dir", default="/tmp/wm_probe_logs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-user", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-jit", action="store_true")
    args = parser.parse_args()

    exp = create_experiment(
        train_path=args.train_path,
        val_path=args.val_path,
        encoder_type=args.encoder_type,
        vae_checkpoint_dir=args.vae_checkpoint_dir,
        vae_checkpoint_step=args.vae_checkpoint_step,
        base_ch=args.base_ch,
        latent_channels=args.latent_channels,
        latent_spatial=args.latent_spatial,
        hidden_dim=args.hidden_dim,
        pooling=args.pooling,
        batch_size_per_device=args.batch_size_per_device,
        num_microbatches=args.num_microbatches,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
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
