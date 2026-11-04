"""The training harness every model in this repo trains through.

The caller supplies the model and the loss; ``Experiment`` owns everything
else: the device mesh and data sharding, gradient accumulation over
microbatches, the optimizer chain (with per-parameter routing), checkpointing,
the validation loop, and metric logging.

* ``model_factory(model_config, key) -> model`` builds the model at
  ``init_state()`` time.
* ``forward_fn(model, batch) -> (loss, aux)`` computes the training loss;
  ``aux`` is a pytree of extra arrays that metrics can read.
* ``eval_forward_fn`` (optional) is used for evaluation instead, e.g. to also
  return reconstructions or per-token losses.
* Datasets are anything with ``batch_size`` and ``__iter__`` (see ``Dataset``);
  ``toylib.harness.data.ArrayDataset`` serves in-memory NumPy arrays.

Usage::

    from toylib.harness.data import ArrayDataset
    from toylib.harness.experiment import Experiment, Task, TrainingConfig

    exp = Experiment(
        train_task=Task("train", ArrayDataset({"image": x, "label": y}, batch_size=128)),
        model_factory=build_model,      # (config, key) -> model
        forward_fn=loss_fn,             # (model, batch) -> (loss, aux)
        model_config=config,
        training_config=TrainingConfig(max_steps=2_000),
    )
    exp.init_state()
    exp.outer_loop()
    exp.cleanup()

Projects add domain-specific behaviour by subclassing; for example
``toylib_projects.p02_gpt_web.experiment.Experiment`` samples text during eval.
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

from toylib.harness import inference
from toylib.harness import logger
from toylib.harness import metrics as metrics_module
from toylib.nn import module

# The optimizer group for frozen parameters, which get no update.
FROZEN = "frozen"


# ──────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class CheckpointConfig:
    save_interval_steps: int = 5_000
    max_to_keep: typing.Optional[int] = 10
    checkpoint_dir: str = "/tmp/checkpoints"  # A local path or gs://bucket/path.
    checkpoint_dataset_iterator: bool = False


@dataclasses.dataclass
class OptimizerConfig:
    """Configuration for a single optimizer."""

    name: str
    optimizer: optax.GradientTransformation


@dataclasses.dataclass
class MultiOptimizerConfig:
    """Route each parameter to one of several optimizers by its pytree path.

    ``optimizer_for_param(path) -> name`` picks the optimizer for a parameter,
    e.g. Muon for transformer blocks and AdamW for embeddings, or
    ``optax.set_to_zero()`` for a group that shouldn't train. Parameters frozen with
    ``Module.freeze`` never reach it: they get their own group.
    """

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

    # The dataset's batch is sharded across local devices, then each device's
    # share is split into this many microbatches whose gradients are averaged.
    # The effective batch stays `dataset.batch_size`; this only lowers peak memory.
    num_microbatches: int = 1

    # A value > 0.0 enables global-norm gradient clipping.
    max_grad_norm: float = 0.0


@dataclasses.dataclass
class EvalConfig:
    eval_interval_steps: int = 500
    num_eval_steps: int = 1


@typing.runtime_checkable
class Dataset(typing.Protocol):
    """What the harness needs from a dataset.

    Iterating yields batches: pytrees of arrays whose leading axis is
    ``batch_size``. Datasets used with ``checkpoint_dataset_iterator=True``
    also need ``get_state() -> dict`` and ``restore_state(dict)``.
    """

    batch_size: int

    def __iter__(self) -> typing.Iterator[typing.Any]: ...


@dataclasses.dataclass
class Task:
    name: str
    dataset: Dataset
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
    user_name: str = ""  # The wandb entity; blank for the account's default.
    # Also write the JSON-lines log that ``FileLogger`` writes.
    also_log_to_file: bool = False

    def build_logger(self, config_dict: dict) -> logger.Logger:
        wandb_logger = self.logger_cls(
            config_dict=config_dict,
            output_path=self.log_dir,
            project_name=self.project_name,
            user_name=self.user_name,
            run_id=self.run_id,
        )
        if not self.also_log_to_file:
            return wandb_logger
        file_logger = logger.FileLogger(
            config_dict=config_dict, output_path=self.log_dir, run_id=self.run_id
        )
        return logger.MultiLogger([file_logger, wandb_logger])


def _serialize_dataclass_config(config) -> typing.Any:
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
    """Train and evaluate a user-supplied model.

    The user supplies the model + loss; this class owns sharding, microbatching,
    optimizer chain, checkpointing, validation loop, and metric logging.

    Required fields:
        train_task:     ``Task`` whose dataset yields training batches.
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
        self, task: Task, loss: jt.ArrayLike, aux: jt.PyTree, batch: jt.PyTree
    ) -> dict[str, jt.Array]:
        all_metrics: dict[str, jt.Array] = {}
        for metric in task.metrics:
            metric_results = metric(loss=loss, aux=aux, batch=batch)
            all_metrics.update(metric_results)
        return all_metrics

    def _create_optimizer(self, has_frozen: bool = False) -> optax.GradientTransformation:
        """Build the optimizer chain: optional grad-clip then a single or multi-optimizer.

        With frozen parameters, they get their own group, `FROZEN`, whose update is
        zero and which keeps no state, so they don't move, not even by weight decay.
        """
        optimizer_chain: list[optax.GradientTransformation] = []
        if self.training_config.max_grad_norm > 0.0:
            optimizer_chain.append(
                optax.clip_by_global_norm(self.training_config.max_grad_norm)
            )

        optimizer_config = self.training_config.optimizer_config
        if optimizer_config is None:
            print("Using default optimizer: Adam, lr=1e-3")
        if optimizer_config is None and not has_frozen:
            optimizer_chain.append(optax.adam(learning_rate=1e-3))
        else:
            if optimizer_config is None:
                optimizer_map = {"adam": optax.adam(learning_rate=1e-3)}

                def optimizer_for_param(path):
                    return "adam"
            else:
                optimizer_map = optimizer_config.build_optimizer_map()
                optimizer_for_param = optimizer_config.optimizer_for_param

            if has_frozen:
                if FROZEN in optimizer_map:
                    raise ValueError(f"The optimizer group name {FROZEN!r} is reserved")
                optimizer_map = {**optimizer_map, FROZEN: optax.set_to_zero()}

            def label_fn(params):
                return jax.tree_util.tree_map_with_path(
                    lambda path, _, trainable: (
                        optimizer_for_param(path) if trainable else FROZEN
                    ),
                    params,
                    module.trainable_mask(params),
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

        # Differentiate only the trainable part. Frozen leaves get zero gradients,
        # so the rest of the step sees one tree shaped like the model.
        mask = module.trainable_mask(model)
        trainable, frozen = module.partition(model, mask)

        def loss_of_trainable(trainable, microbatch):
            return self.forward_fn(module.combine(trainable, frozen), microbatch)

        def compute_microbatch(microbatch):
            (loss_val, aux), grads = jax.value_and_grad(loss_of_trainable, has_aux=True)(
                trainable, microbatch
            )
            grads = module.combine(grads, jax.tree.map(jnp.zeros_like, frozen))
            microbatch_metrics = self._compute_metrics(
                task=self.train_task, loss=loss_val, aux=aux, batch=microbatch
            )
            return grads, microbatch_metrics

        if num_microbatches == 1:
            # Skip the scan: on XLA CPU a one-iteration scan made training 8x
            # slower (8 vs 64 steps/s on the post 1 CNN).
            total_grads, averaged_metrics = compute_microbatch(batch)
        else:
            # Reshape leaves from [sharded_batch, ...] to
            # [num_microbatches, microbatch_size, ...] so scan can slice along axis 0.
            microbatches = jax.tree.map(
                lambda x: x.reshape(num_microbatches, microbatch_size, *x.shape[1:]),
                batch,
            )

            def scan_fn(carry_grads, microbatch):
                grads, microbatch_metrics = compute_microbatch(microbatch)
                new_carry_grads = jax.tree.map(lambda c, g: c + g, carry_grads, grads)
                return new_carry_grads, microbatch_metrics

            init_carry_grads = jax.tree.map(jnp.zeros_like, model)
            with jax.profiler.TraceAnnotation("microbatch_loop"):
                total_grads, all_metrics = jax.lax.scan(
                    scan_fn, init_carry_grads, microbatches
                )

            # Average over microbatches. No division by the device count: under jit
            # the batch is one global array (sharded behind the scenes), so each
            # microbatch's loss is already the mean over every device's rows and
            # its gradient is already global. Dividing by num_devices, as one would
            # with pmap, shrank every update by 1/N on N devices.
            total_grads = jax.tree.map(lambda g: g / num_microbatches, total_grads)
            averaged_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), all_metrics)

        with jax.profiler.TraceAnnotation("optimizer_update"):
            updates, opt_state = self.optimizer.update(total_grads, opt_state, model)
            # Before any clipping, and relative to the weights, so both stay readable
            # at any learning rate or model size. Frozen weights don't count: RoPE's
            # tables alone would outweigh a small model's.
            averaged_metrics = {
                **averaged_metrics,
                "grad_norm": optax.global_norm(total_grads),
                "update_to_weight": optax.global_norm(updates)
                / optax.global_norm(trainable),
            }
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

        has_frozen = not all(jax.tree.leaves(module.trainable_mask(self.model)))
        self.optimizer = self._create_optimizer(has_frozen)

        with jax.set_mesh(self.mesh):
            self.opt_state = self.optimizer.init(self.model)

        self.step = 0
        self._restart_speed_clock()

        print(f"Model initialized and replicated across {self.num_devices} devices")

    def _restart_speed_clock(self) -> None:
        self._train_start_step = self.step
        self._train_start_time = time.monotonic()

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
        step = self.ckpt_manager.latest_step()
        if step is None:
            raise ValueError(
                f"No checkpoints in {self.checkpoint_config.checkpoint_dir!r} to restore"
            )
        return int(step)

    def restore_checkpoint(self, step: int | None = None):
        self._assert_initialized()
        if step is None:
            step = self._resolve_latest_saved_checkpoint_step()

        model_template = self._unreplicate_for_checkpoint(self.model)
        opt_state_template = self._unreplicate_for_checkpoint(self.opt_state)

        # Frozen parameters added since the checkpoint keep their fresh values.
        frozen = jax.tree.map(lambda m: not m, module.trainable_mask(model_template))
        model = inference.restore_tree(
            self.ckpt_manager, step, "model", model_template, frozen
        )
        opt_state = inference.restore_tree(
            self.ckpt_manager, step, "opt_state", opt_state_template
        )
        self.model = jax.device_put(model, self.replicated_sharding)
        self.opt_state = jax.device_put(opt_state, self.replicated_sharding)

        if self.checkpoint_config.checkpoint_dataset_iterator:
            restored = self.ckpt_manager.restore(
                step,
                args=ocp.args.Composite(
                    dataset_iterator=ocp.args.StandardRestore(
                        self.train_task.dataset.get_state()
                    )
                ),
            )
            self.train_task.dataset.restore_state(restored["dataset_iterator"])
        self.step = step

    def resume(self) -> bool:
        """Continue from the newest checkpoint, if there is one.

        With ``checkpoint_dir`` in a ``gs://`` bucket, this continues a run on a new
        machine. Returns whether a checkpoint was restored; if not, training starts
        from the beginning.
        """
        self._assert_initialized()
        if self.ckpt_manager.latest_step() is None:
            return False
        self.restore_checkpoint()
        self.step += 1  # A checkpoint is saved after its step has trained.
        self._restart_speed_clock()
        return True

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

        for batch in self.eval_task.dataset:
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
            if num_batches >= self.eval_config.num_eval_steps:
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

    def extra_train_logs(self, steps_per_sec: float) -> dict[str, float]:
        """Values to log with the training metrics. Subclasses can add to them."""
        return {}

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
            steps_done = self.step - self._train_start_step
            steps_per_sec = 0.0
            if elapsed > 0 and steps_done > 0:
                steps_per_sec = steps_done / elapsed
                train_metrics_with_prefix["train/steps_per_sec"] = steps_per_sec
            train_metrics_with_prefix.update(self.extra_train_logs(steps_per_sec))
            self.logger_obj.log(self.step, metrics=train_metrics_with_prefix)

    def outer_loop(self):
        finished = self.step >= self.training_config.max_steps

        while not finished:
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

            if self.step == epoch_start_step:
                raise ValueError(f"Dataset for task {self.train_task.name} is empty.")

        # Also save the last step, which the interval usually misses.
        if self.step == self.training_config.max_steps:
            self.step -= 1  # Label the checkpoint with the step it finished.
            if self.step not in self.ckpt_manager.all_steps():
                self.save_checkpoint()
            self.step += 1

    def cleanup(self):
        """Release logger + checkpoint manager. Call once after training."""
        self.logger_obj.close()
        self.ckpt_manager.close()
