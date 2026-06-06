"""Experiment harness for vision-encoder training.

Forked and adapted from `toylib_projects/tinystories/experiment.py`. The bulk
of the orchestration (sharded `Mesh`, microbatching scan, optimizer chain,
checkpoint manager, logger plumbing) is unchanged — only the text-specific
pieces have been ripped out and a couple of generic seams added so this
harness works with the vision-encoder dataset and any user-supplied model.


What changed vs. the tinystories template
-----------------------------------------

  - **No model imports.** Per the project's learning rules, this file knows
    nothing about VAEs / DiTs / attention. The user supplies the model via
    ``model_factory(model_config, key) -> model`` and the loss via
    ``forward_fn(model, batch) -> (loss, aux_pytree)``.
  - **No tokenizer.** The text-only ``sampling_evaluation`` and the
    ``DEFAULT_PROMPTS`` constant are gone. ``eval()`` runs validation only.
    Subclasses can override ``sampling_evaluation`` if they want to log VAE
    reconstructions or similar — it's a no-op by default.
  - **Generic batch shape.** ``_train_step`` no longer assumes
    ``batch["inputs"]``; it pulls the batch size from
    ``jax.tree.leaves(batch)[0]`` so dicts *and* raw tensors both work.
  - **Generic forward-fn contract.** ``forward_fn(model, batch)`` always
    returns ``(loss, aux)``. The old ``return_aux=True`` kwarg has been
    dropped — one consistent signature for train and eval.
  - **Dataset type.** ``Task.dataset`` is typed as ``Hdf5FramesDataset``
    (forked HDF5-backed Grain loader with checkpointable iterator state).
    Any object with the same protocol (``__iter__``, ``batch_size``,
    ``get_state``, ``restore_state``) works at runtime.
  - **Required model_factory + forward_fn.** No tinystories defaults.


Intended usage (sketch)
-----------------------

::

    from toylib_projects.wm.dataloader import Hdf5FramesDataset
    from toylib_projects.wm.experiment import (
        Experiment, Task, TrainingConfig, EvalConfig,
        CheckpointConfig, LoggerConfig,
    )

    train_ds = Hdf5FramesDataset("data/compiled/vae_train.h5", batch_size=64, seed=0)
    val_ds   = Hdf5FramesDataset("data/compiled/vae_val.h5",   batch_size=64,
                                 seed=0, shuffle=False)

    # User code (the bits this harness deliberately doesn't touch):
    #   class MyVAEConfig: ...
    #   class MyVAE: ...
    #   def vae_factory(cfg, key): return MyVAE(cfg, key=key)
    #   def vae_train_step(model, batch): return loss, {"recon_mse": ...}

    exp = Experiment(
        train_task=Task("train", train_ds),
        eval_task=Task("val", val_ds),
        model_config=MyVAEConfig(...),
        model_factory=vae_factory,
        forward_fn=vae_train_step,
    )
    exp.init_state()
    exp.outer_loop()
    exp.cleanup()
"""

import dataclasses
import time
import typing

import jax
import jax.numpy as jnp
import jaxtyping as jt
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax
import orbax.checkpoint as ocp

from toylib_projects.wm import dataloader as dataloader_lib
from toylib_projects.wm import logger
from toylib_projects.wm import metrics as metrics_module


# ──────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class CheckpointConfig:
    save_interval_steps: int = 5_000
    max_to_keep: typing.Optional[int] = 10
    checkpoint_dir: str = "/tmp/checkpoints"
    checkpoint_dataset_iterator: bool = False


@dataclasses.dataclass
class OptimizerConfig:
    """Configuration for a single optimizer."""

    name: str
    optimizer: optax.GradientTransformation


@dataclasses.dataclass
class MultiOptimizerConfig:
    """Configuration for multi-optimizer training (e.g. different lr for VAE encoder vs decoder)."""

    optimizer_configs: list[OptimizerConfig]
    optimizer_for_param: typing.Callable[[tuple], str]

    def build_optimizer_map(self) -> dict[str, optax.GradientTransformation]:
        if not self.optimizer_configs:
            raise ValueError("multi_optimizer_config.optimizer_configs cannot be empty")
        optimizer_map = {
            config.name: config.optimizer for config in self.optimizer_configs
        }
        assert len(optimizer_map) == len(self.optimizer_configs)
        return optimizer_map


@dataclasses.dataclass
class TrainingConfig:
    # Configuration for applying different optimizers to different model parts.
    # If None, uses a single Adam optimizer for all parameters.
    optimizer_config: MultiOptimizerConfig | None = None

    max_steps: int = 100_000

    # The effective batch is `dataset.batch_size * num_microbatches`.
    # The dataset's batch is sharded across local devices.
    num_microbatches: int = 1

    # A value > 0.0 enables global-norm gradient clipping.
    max_grad_norm: float = 0.0


@dataclasses.dataclass
class EvalConfig:
    eval_interval_steps: int = 500
    num_eval_steps: int = 1


@dataclasses.dataclass
class Task:
    name: str
    dataset: dataloader_lib.Hdf5FramesDataset
    metrics: list[metrics_module.Metric] = dataclasses.field(
        default_factory=lambda: [metrics_module.Loss()]
    )
    visualization_metrics: list[metrics_module.VisualizationMetric] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass(kw_only=True)
class LoggerConfig:
    logger_cls: type[logger.Logger] = logger.FileLogger
    log_dir: str = "/tmp/"
    run_id: str | None = None

    train_log_interval_steps: int = 1

    def build_logger(self, config_dict: dict) -> logger.Logger:
        return self.logger_cls(
            config_dict=config_dict, output_path=self.log_dir, run_id=self.run_id
        )


@dataclasses.dataclass
class WandBLoggerConfig(LoggerConfig):
    logger_cls: type[logger.Logger] = logger.WandBLogger
    project_name: str = ""
    user_name: str = ""

    def build_logger(self, config_dict: dict) -> logger.Logger:
        return self.logger_cls(
            config_dict=config_dict,
            output_path=self.log_dir,
            project_name=self.project_name,
            user_name=self.user_name,
            run_id=self.run_id,
        )


def _serialize_dataclass_config(config) -> dict:
    """Recursively convert dataclasses to dicts, leaving non-dataclass values alone.

    Used to capture the experiment configuration for the logger. Non-dataclass
    values (callables, user model configs, etc.) are stringified so json/wandb
    can serialize them without crashing.
    """
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        result = {}
        for f in dataclasses.fields(config):
            v = getattr(config, f.name)
            try:
                result[f.name] = _serialize_dataclass_config(v)
            except (TypeError, ValueError):
                result[f.name] = repr(v)
        return result
    if isinstance(config, (list, tuple)):
        return [_serialize_dataclass_config(x) for x in config]
    if isinstance(config, dict):
        return {k: _serialize_dataclass_config(v) for k, v in config.items()}
    if isinstance(config, (str, int, float, bool, type(None))):
        return config
    return repr(config)


# ──────────────────────────────────────────────────────────────────────────
# Experiment
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Experiment:
    """Generic training harness for the vision-encoder pipeline.

    The user supplies the model + loss; this class owns sharding, microbatching,
    optimizer chain, checkpointing, validation loop, and metric logging.

    Required fields:
        train_task:     ``Task`` with a ``Hdf5FramesDataset``.
        forward_fn:     ``(model, batch) -> (loss, aux_pytree)``.
        model_factory:  ``(model_config, jax.random.key) -> model`` callable
                        invoked at ``init_state()`` time.

    Optional fields default to sensible scaffolding values.
    """

    # Required user-supplied
    train_task: Task
    forward_fn: typing.Callable[..., tuple[jt.Array, jt.PyTree]]
    model_factory: typing.Callable[..., typing.Any]

    # Optional
    eval_task: Task | None = None
    # If set, used instead of forward_fn for eval steps. Lets the eval pass
    # return extra tensors (e.g. reconstructions for visualization) without
    # bloating the training scan with large aux arrays.
    eval_forward_fn: typing.Callable[..., tuple[jt.Array, jt.PyTree]] | None = None
    model_config: typing.Any = None

    training_config: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    eval_config: EvalConfig = dataclasses.field(default_factory=EvalConfig)
    checkpoint_config: CheckpointConfig = dataclasses.field(
        default_factory=CheckpointConfig
    )
    logger_config: LoggerConfig = dataclasses.field(default_factory=LoggerConfig)

    seed: int = 0

    # JIT-compile the training step. Disable for stepping through with a debugger.
    jit_computations: bool = True

    # ---- setup --------------------------------------------------------------

    def _validate_configs(self) -> None:
        train_batch_size = self.train_task.dataset.batch_size
        eval_batch_size = (
            self.eval_task.dataset.batch_size if self.eval_task is not None else 0
        )
        if train_batch_size % self.num_devices != 0:
            raise ValueError(
                f"Train batch size {train_batch_size} not divisible by number of "
                f"devices {self.num_devices}"
            )
        if eval_batch_size % self.num_devices != 0 and eval_batch_size != 0:
            raise ValueError(
                f"Eval batch size {eval_batch_size} not divisible by number of "
                f"devices {self.num_devices}"
            )
        if (
            train_batch_size // self.num_devices
        ) % self.training_config.num_microbatches != 0:
            raise ValueError(
                f"Number of microbatches {self.training_config.num_microbatches} "
                f"does not evenly divide per-device batch size "
                f"{train_batch_size // self.num_devices}"
            )

    def _setup_sharding(self) -> None:
        self.num_devices = jax.local_device_count()
        devices = np.array(jax.local_devices())
        self.mesh = Mesh(devices, axis_names=("data",))

        # Model + optimizer state: replicated.
        self.replicated_sharding = NamedSharding(self.mesh, P())
        # Data: sharded along the batch (leading) axis.
        self.data_sharding = NamedSharding(self.mesh, P("data"))

        print(
            f"Initialized mesh {self.mesh} with {self.num_devices} devices: {devices}"
        )

    def _compute_metrics(
        self, task: Task, loss: float, aux: jt.PyTree, batch: jt.PyTree
    ) -> dict[str, jt.Array]:
        all_metrics: dict[str, jt.Array] = {}
        for metric in task.metrics:
            metric_results = metric(loss=loss, aux=aux, batch=batch)
            all_metrics.update(metric_results)
        return all_metrics

    def _create_optimizer(self) -> optax.GradientTransformation:
        """Build the optimizer chain: optional grad-clip then a single or multi-optimizer."""
        optimizer_chain: list[optax.GradientTransformation] = []
        if self.training_config.max_grad_norm > 0.0:
            optimizer_chain.append(
                optax.clip_by_global_norm(self.training_config.max_grad_norm)
            )

        if self.training_config.optimizer_config is None:
            print("Using default optimizer: Adam, lr=1e-3")
            optimizer_chain.append(optax.adam(learning_rate=1e-3))
        else:
            optimizer_map = self.training_config.optimizer_config.build_optimizer_map()
            optimizer_for_param = (
                self.training_config.optimizer_config.optimizer_for_param
            )

            def label_fn(params):
                return jax.tree_util.tree_map_with_path(
                    lambda path, _: optimizer_for_param(path), params
                )

            optimizer_chain.append(
                optax.multi_transform(
                    transforms=optimizer_map,
                    param_labels=label_fn,
                )
            )

        if len(optimizer_chain) == 1:
            return optimizer_chain[0]
        return optax.chain(*optimizer_chain)

    # ---- train / eval steps ------------------------------------------------

    def _train_step(self, model, opt_state, batch):
        """One sharded training step with microbatching."""
        # Generic batch-size lookup — works for raw tensor batches AND dict batches.
        sharded_batch_size = jax.tree.leaves(batch)[0].shape[0]

        num_microbatches = self.training_config.num_microbatches
        microbatch_size = sharded_batch_size // num_microbatches

        # Reshape leaves from [sharded_batch, ...] to
        # [num_microbatches, microbatch_size, ...] so scan can slice along axis 0.
        microbatches = jax.tree.map(
            lambda x: x.reshape(num_microbatches, microbatch_size, *x.shape[1:]),
            batch,
        )

        def scan_fn(carry_grads, microbatch):
            (loss_val, aux), grads = jax.value_and_grad(self.forward_fn, has_aux=True)(
                model, microbatch
            )
            new_carry_grads = jax.tree.map(lambda c, g: c + g, carry_grads, grads)
            microbatch_metrics = self._compute_metrics(
                task=self.train_task, loss=loss_val, aux=aux, batch=microbatch
            )
            return new_carry_grads, microbatch_metrics

        init_carry_grads = jax.tree.map(jnp.zeros_like, model)
        with jax.profiler.TraceAnnotation("microbatch_loop"):
            total_grads, all_metrics = jax.lax.scan(
                scan_fn, init_carry_grads, microbatches
            )

        # Average grads over devices × microbatches (jax sums by default across the mesh).
        total_grads = jax.tree.map(
            lambda g: g / self.num_devices / num_microbatches, total_grads
        )
        averaged_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), all_metrics)

        with jax.profiler.TraceAnnotation("optimizer_update"):
            updates, opt_state = self.optimizer.update(total_grads, opt_state, model)
            model = optax.apply_updates(model, updates)

        return model, opt_state, averaged_metrics

    def _eval_step(self, model, batch):
        """One sharded eval step. Uses eval_forward_fn if set, else forward_fn."""
        fwd = self.eval_forward_fn if self.eval_forward_fn is not None else self.forward_fn
        with jax.profiler.TraceAnnotation("eval_forward"):
            loss_val, aux = fwd(model, batch)
        return self._compute_metrics(
            task=self.eval_task, loss=loss_val, aux=aux, batch=batch
        )

    # ---- lifecycle ---------------------------------------------------------

    def __post_init__(self):
        self._setup_sharding()
        self._validate_configs()

        self.logger_obj = self.logger_config.build_logger(
            config_dict=_serialize_dataclass_config(self),
        )

        # Lazy state — initialized in init_state().
        self.optimizer: optax.GradientTransformation | None = None
        self.opt_state = None
        self.model = None

        self.ckpt_manager = ocp.CheckpointManager(
            self.checkpoint_config.checkpoint_dir,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=self.checkpoint_config.max_to_keep
            ),
        )

        if self.jit_computations:
            self.train_step_fn = jax.jit(
                self._train_step,
                in_shardings=(
                    self.replicated_sharding,  # model
                    self.replicated_sharding,  # opt_state
                    self.data_sharding,  # batch
                ),
                out_shardings=(
                    self.replicated_sharding,  # model
                    self.replicated_sharding,  # opt_state
                    self.replicated_sharding,  # metrics
                ),
            )
            self.eval_step_fn = jax.jit(
                self._eval_step,
                in_shardings=(
                    self.replicated_sharding,  # model
                    self.data_sharding,  # batch
                ),
                out_shardings=self.replicated_sharding,
            )
        else:
            self.train_step_fn = self._train_step
            self.eval_step_fn = self._eval_step

    def init_state(self):
        """Construct the model + optimizer state. Must be called before ``outer_loop``."""
        with jax.set_mesh(self.mesh):
            self.model = self.model_factory(
                self.model_config, jax.random.key(self.seed)
            )

        self.optimizer = self._create_optimizer()

        with jax.set_mesh(self.mesh):
            self.opt_state = self.optimizer.init(self.model)

        self.step = 0
        self._train_start_time = time.monotonic()

        print(f"Model initialized and replicated across {self.num_devices} devices")

    def _assert_initialized(self) -> None:
        assert self.model is not None and self.opt_state is not None, (
            "Experiment state not initialized. Call init_state() first."
        )

    # ---- checkpointing -----------------------------------------------------

    def _unreplicate_for_checkpoint(self, pytree):
        """Pull a single host-side copy of replicated state for serialization."""
        return jax.tree.map(lambda x: np.asarray(x), pytree)

    def save_checkpoint(self):
        self._assert_initialized()

        model_to_save = self._unreplicate_for_checkpoint(self.model)
        opt_state_to_save = self._unreplicate_for_checkpoint(self.opt_state)

        args = {
            "model": ocp.args.StandardSave(model_to_save),
            "opt_state": ocp.args.StandardSave(opt_state_to_save),
        }
        if self.checkpoint_config.checkpoint_dataset_iterator:
            args["dataset_iterator"] = ocp.args.StandardSave(
                self.train_task.dataset.get_state()
            )
        self.ckpt_manager.save(self.step, args=ocp.args.Composite(**args))
        self.ckpt_manager.wait_until_finished()

    def _resolve_latest_saved_checkpoint_step(self) -> int:
        raise NotImplementedError("provide a step explicitly!")

    def restore_checkpoint(self, step: int | None = None):
        self._assert_initialized()
        if step is None:
            step = self._resolve_latest_saved_checkpoint_step()

        model_template = self._unreplicate_for_checkpoint(self.model)
        opt_state_template = self._unreplicate_for_checkpoint(self.opt_state)

        args = {
            "model": ocp.args.StandardRestore(model_template),
            "opt_state": ocp.args.StandardRestore(opt_state_template),
        }
        if self.checkpoint_config.checkpoint_dataset_iterator:
            args["dataset_iterator"] = ocp.args.StandardRestore(
                self.train_task.dataset.get_state()
            )

        restored = self.ckpt_manager.restore(step, args=ocp.args.Composite(**args))

        self.model = jax.device_put(restored["model"], self.replicated_sharding)
        self.opt_state = jax.device_put(restored["opt_state"], self.replicated_sharding)

        if self.checkpoint_config.checkpoint_dataset_iterator:
            self.train_task.dataset.restore_state(restored["dataset_iterator"])
        self.step = step

    # ---- eval --------------------------------------------------------------

    def run_validation(self) -> dict[str, float]:
        """Run validation and return averaged scalar metrics (with ``val/`` prefix).

        Metric values with ndim == 0 are accumulated and averaged across batches.
        Values with ndim > 0 are treated as image batches: those from the first
        eval step are logged once via ``log_images`` and not averaged.
        """
        self._assert_initialized()
        if self.eval_task is None:
            print("No eval task defined, skipping validation.")
            return {}

        accumulated_scalars: dict | None = None
        first_batch_images: dict | None = None
        num_batches = 0

        for ix, batch in enumerate(self.eval_task.dataset):
            batch_metrics = self.eval_step_fn(self.model, batch)

            scalars = {k: v for k, v in batch_metrics.items() if v.ndim == 0}
            images = {k: v for k, v in batch_metrics.items() if v.ndim > 0}

            if accumulated_scalars is None:
                accumulated_scalars = scalars
            else:
                accumulated_scalars = jax.tree.map(
                    lambda x, y: x + y, accumulated_scalars, scalars
                )
            if first_batch_images is None and images:
                first_batch_images = images

            num_batches += 1
            if ix >= self.eval_config.num_eval_steps:
                break

        if num_batches == 0:
            print("Eval dataset yielded no batches, skipping validation.")
            return {}

        avg_metrics = jax.tree.map(
            lambda x: float(x) / num_batches, accumulated_scalars
        )
        avg_metrics = {f"val/{k}": v for k, v in avg_metrics.items()}
        self.logger_obj.log(self.step, metrics=avg_metrics)

        if first_batch_images:
            for k, imgs in first_batch_images.items():
                self.logger_obj.log_images(self.step, f"val/{k}", np.asarray(imgs))

        return avg_metrics

    def sampling_evaluation(self) -> None:
        """Run generative VisualizationMetrics (outside JIT) and log resulting images.

        These metrics don't consume an input batch — they generate images from the
        model directly (e.g. decoding random prior latents). Reconstruction metrics
        that read from ``aux`` belong in the regular ``metrics`` list on the task and
        run inside the JIT-compiled eval step instead.
        """
        if self.eval_task is None or not self.eval_task.visualization_metrics:
            return

        for metric in self.eval_task.visualization_metrics:
            images = metric(self.model)
            for key, imgs in images.items():
                self.logger_obj.log_images(self.step, f"val/{key}", np.asarray(imgs))

    def eval(self):
        self._assert_initialized()
        self.run_validation()
        self.sampling_evaluation()

    # ---- main loop ---------------------------------------------------------

    def inner_loop(self, batch: jt.PyTree):
        self._assert_initialized()

        self.model, self.opt_state, train_metrics = self.train_step_fn(
            self.model, self.opt_state, batch
        )

        if self.step % self.logger_config.train_log_interval_steps == 0:
            train_metrics_with_prefix = {
                f"train/{k}": float(v) for k, v in train_metrics.items()
            }
            elapsed = time.monotonic() - self._train_start_time
            if elapsed > 0 and self.step > 0:
                train_metrics_with_prefix["train/steps_per_sec"] = self.step / elapsed
            self.logger_obj.log(self.step, metrics=train_metrics_with_prefix)

    def outer_loop(self):
        finished = self.step >= self.training_config.max_steps

        while True:
            epoch_start_step = self.step
            for batch in self.train_task.dataset:
                with jax.profiler.StepTraceAnnotation("inner_loop", step_num=self.step):
                    self.inner_loop(batch)

                if self.step % self.checkpoint_config.save_interval_steps == 0:
                    self.save_checkpoint()
                if self.step % self.eval_config.eval_interval_steps == 0:
                    self.eval()

                self.step += 1
                if self.step >= self.training_config.max_steps:
                    finished = True
                    break

            if finished:
                break
            if self.step == epoch_start_step:
                raise ValueError(f"Dataset for task {self.train_task.name} is empty.")

    def cleanup(self):
        """Release logger + checkpoint manager. Call once after training."""
        self.logger_obj.close()
        self.ckpt_manager.close()
