"""Basic types for the training loop and configurations."""

import dataclasses
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np
import optax
import orbax.checkpoint as ocp
import typing

from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import data
from toylib_projects.tinystories import logger
from toylib_projects.tinystories import metrics as metrics_module


DEFAULT_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]


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
    """Configuration for multi-optimizer training."""

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

    # The total batch size is effectively `batch_size * num_microbatches`.
    # This leads to an effective total number of tokens of
    # `batch_size * seq_len * num_microbatches`. The batch is sharded
    # between the available devices.
    # Only applied to the training config.
    num_microbatches: int = 1

    # A value > 0.0 enables gradient clipping
    max_grad_norm: float = 0.0


@dataclasses.dataclass
class EvalConfig:
    # Evaluation interval in steps
    eval_interval_steps: int = 500
    # Number of batches to use for evaluation
    num_eval_steps: int = 1


@dataclasses.dataclass
class Task:
    name: str
    dataset: data.BatchedTokenizedDataset
    metrics: list[metrics_module.Metric] = dataclasses.field(
        default_factory=lambda: [metrics_module.Loss()]
    )


@dataclasses.dataclass(kw_only=True)
class LoggerConfig:
    logger_cls: logger.Logger = logger.FileLogger
    log_dir: str = "/tmp/"

    train_log_interval_steps: int = 1


def _serialize_dataclass_config(config: dataclasses.dataclass) -> dict:
    result = dataclasses.asdict(config)
    for k, v in result.items():
        if dataclasses.is_dataclass(v):
            result[k] = _serialize_dataclass_config(v)
    return result


@dataclasses.dataclass
class Experiment:
    """Base Experiment class."""

    # Tasks to train and evaluate
    train_task: Task
    eval_task: Task | None = None

    # Model config
    model_config: decoder_only_model.ModelConfig = dataclasses.field(
        default_factory=decoder_only_model.ModelConfig
    )
    # Training Hyperparameters
    training_config: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    # Eval config
    eval_config: EvalConfig = dataclasses.field(default_factory=EvalConfig)

    # Checkpointing config
    checkpoint_config: CheckpointConfig = dataclasses.field(
        default_factory=CheckpointConfig
    )

    # Logger config
    logger_config: LoggerConfig = dataclasses.field(default_factory=LoggerConfig)

    # Forward function: returns loss value and aux data dict
    forward_fn: ... = dataclasses.field(
        default_factory=lambda: decoder_only_model.train_step
    )

    # Whether to JIT compile the training step - disable only for debugging
    jit_computations: bool = True

    def _validate_configs(self) -> None:
        # Validate train and eval batch sizes
        train_batch_size = self.train_task.dataset.batch_size
        eval_batch_size = (
            self.eval_task.dataset.batch_size if self.eval_task is not None else 0
        )
        if train_batch_size % self.num_devices != 0:
            raise ValueError(
                f"Batch size {self.batch_size} not divisible by number of devices "
                f"{self.num_devices}"
            )
        if eval_batch_size % self.num_devices != 0 and eval_batch_size != 0:
            raise ValueError(
                f"Eval batch size {eval_batch_size} not divisible by number of "
                f"devices {self.num_devices}"
            )
        if (
            train_batch_size / self.num_devices
        ) % self.training_config.num_microbatches != 0:
            raise ValueError(
                f"Number of microbatches {self.training_config.num_microbatches} "
                f"does not evenly divide per-device batch size "
                f"{train_batch_size // self.num_devices}"
            )

    def _setup_sharding(self) -> None:
        # Set up device mesh for data parallelism
        self.num_devices = jax.local_device_count()
        devices = np.array(jax.local_devices())
        self.mesh = Mesh(devices, axis_names=("data",))

        # Model and optimizer state: replicated across all devices
        self.replicated_sharding = NamedSharding(self.mesh, P())
        # Data: sharded along batch dimension across devices
        self.data_sharding = NamedSharding(self.mesh, P("data"))

        print(
            f"Initialized mesh {self.mesh} with {self.num_devices} devices: {devices}"
        )

    def _compute_metrics(
        self, task: Task, loss: float, aux: jt.PyTree, batch: jt.PyTree
    ) -> dict[str, jt.Array]:
        """Compute all metrics for a task.

        Args:
            task: The task containing the metrics to compute
            loss: The loss value
            aux: The auxiliary PyTree from forward_fn
            batch: The input batch

        Returns:
            Dictionary mapping metric names to values
        """
        all_metrics = {}
        for metric in task.metrics:
            metric_results = metric(loss=loss, aux=aux, batch=batch)
            all_metrics.update(metric_results)
        return all_metrics

    def _create_optimizer(self, model: jt.PyTree) -> optax.GradientTransformation:
        """Create the optimizer with optional per-parameter optimization.

        This is called after model initialization to create an optimizer that
        can apply different optimizers to different parts of the model. The
        gradient clipping is applied globally before the multi-optimizer.

        Args:
            model: The initialized model PyTree

        Returns:
            Optax optimizer chain with gradient clipping and multi-optimizer
        """

        # Build optimizer chain with gradient clipping
        optimizer_chain = []
        if self.training_config.max_grad_norm > 0.0:
            optimizer_chain.append(
                optax.clip_by_global_norm(self.training_config.max_grad_norm)
            )

        # Use multi-optimizer if multi_optimizer_config is provided
        if self.training_config.optimizer_config is None:
            print("Using default optimizer: Adam, 1e-3")
            optimizer_chain.append(optax.adam(learning_rate=1e-3))

        else:
            optimizer_map = self.training_config.optimizer_config.build_optimizer_map()

            # Create a function that maps the params PyTree to a PyTree of labels
            optimizer_for_param = (
                self.training_config.optimizer_config.optimizer_for_param
            )

            def label_fn(params):
                """Map params PyTree to labels PyTree using optimizer_for_param."""
                return jax.tree_util.tree_map_with_path(
                    lambda path, _: optimizer_for_param(path), params
                )

            optimizer_chain.append(
                optax.multi_transform(
                    transforms=optimizer_map,
                    param_labels=label_fn,
                )
            )

        # If there's only one transform, return it directly to avoid wrapping
        if len(optimizer_chain) == 1:
            return optimizer_chain[0]
        return optax.chain(*optimizer_chain)

    def _train_step(self, model, opt_state, batch):
        """Perform a single training step with microbatching."""

        # train_step is run on each device with a shard of the batch
        sharded_batch_size = batch["inputs"].shape[0]

        # Microbatching related config
        num_microbatches = self.training_config.num_microbatches
        microbatch_size = sharded_batch_size // num_microbatches

        # Reshape batch leaves from [sharded_batch, ...] to [num_microbatches, microbatch_size, ...]
        # so scan can slice one microbatch per iteration along the leading axis.
        microbatches = jax.tree.map(
            lambda x: x.reshape(num_microbatches, microbatch_size, *x.shape[1:]), batch
        )

        def scan_fn(carry_grads, microbatch):
            # Run forward and backward pass on the microbatch.
            # model is closed over — it does not change during gradient accumulation.
            (loss_val, aux), grads = jax.value_and_grad(
                self.forward_fn, has_aux=True
            )(model, microbatch)
            # Accumulate gradients into the carry.
            new_carry_grads = jax.tree.map(lambda c, g: c + g, carry_grads, grads)
            # Metrics are emitted as outputs and stacked by scan across iterations.
            microbatch_metrics = self._compute_metrics(
                task=self.train_task, loss=loss_val, aux=aux, batch=microbatch
            )
            return new_carry_grads, microbatch_metrics

        init_carry_grads = jax.tree.map(jnp.zeros_like, model)
        with jax.profiler.TraceAnnotation("microbatch_loop"):
            total_grads, all_metrics = jax.lax.scan(
                scan_fn, init_carry_grads, microbatches
            )

        # The default reduce operation in jax is a sum across devices.
        # Average the grads and metrics by the number of devices and microbatches.
        # all_metrics leaves have shape [num_microbatches, ...] — mean over the leading axis.
        total_grads = jax.tree.map(
            lambda g: g / self.num_devices / num_microbatches, total_grads
        )
        averaged_metrics = jax.tree.map(
            lambda x: jnp.mean(x, axis=0), all_metrics
        )

        with jax.profiler.TraceAnnotation("optimizer_update"):
            updates, opt_state = self.optimizer.update(total_grads, opt_state, model)

            # Update the model and optimizer state
            model = optax.apply_updates(model, updates)

        return model, opt_state, averaged_metrics

    def _eval_step(self, model, batch):
        """Perform a single evaluation step and compute metrics."""
        with jax.profiler.TraceAnnotation("eval_forward"):
            loss_val, aux = self.forward_fn(model, batch)

        # Compute metrics for this batch
        eval_metrics = self._compute_metrics(
            task=self.eval_task, loss=loss_val, aux=aux, batch=batch
        )

        return eval_metrics

    def __post_init__(self):
        self._setup_sharding()
        self._validate_configs()

        # Logger
        self.logger_obj = self.logger_config.logger_cls(
            config_dict=_serialize_dataclass_config(self),
            output_path=self.logger_config.log_dir,
        )

        # Optimizer will be created in init_state() after model is initialized
        self.optimizer = None
        self.opt_state = None
        self.model = None

        # Checkpoint manager
        self.ckpt_manager = ocp.CheckpointManager(
            self.checkpoint_config.checkpoint_dir,
            checkpointers={
                "model": ocp.StandardCheckpointer(),
                "opt_state": ocp.StandardCheckpointer(),
            },
            options=ocp.CheckpointManagerOptions(
                max_to_keep=self.checkpoint_config.max_to_keep
            ),
        )

        if self.jit_computations:
            # JIT with explicit sharding specifications
            self.train_step_fn = jax.jit(
                self._train_step,
                in_shardings=(
                    self.replicated_sharding,  # model
                    self.replicated_sharding,  # opt_state
                    self.data_sharding,  # batch (sharded)
                ),
                out_shardings=(
                    self.replicated_sharding,  # model
                    self.replicated_sharding,  # opt_state
                    self.replicated_sharding,  # metrics (replicated)
                ),
            )
            self.eval_step_fn = jax.jit(
                self._eval_step,
                in_shardings=(
                    self.replicated_sharding,
                    self.data_sharding,
                ),
                out_shardings=self.replicated_sharding,  # metrics
            )
        else:
            self.train_step_fn = self._train_step
            self.eval_step_fn = self._eval_step

    def init_state(self):
        # Initialize model on CPU first
        self.model = decoder_only_model.DecoderOnlyTransformer(
            config=self.model_config, key=jax.random.key(0)
        )
        # Replicate model across all devices
        self.model = jax.device_put(self.model, self.replicated_sharding)

        # Create the optimizer based on the model structure
        # This allows different optimizers for different parts of the model
        self.optimizer = self._create_optimizer(self.model)

        # Initialize and replicate optimizer state
        self.opt_state = self.optimizer.init(self.model)
        self.opt_state = jax.device_put(self.opt_state, self.replicated_sharding)

        self.step = 0

        print(f"Model initialized and replicated across {self.num_devices} devices")

    def _assert_initialized(self) -> bool:
        initialized = self.model is not None and self.opt_state is not None
        assert initialized, "Experiment state not initialized. Call init_state() first."

    def _unreplicate_for_checkpoint(self, pytree):
        """Get a single copy of replicated state for checkpointing."""
        # For replicated sharding, all devices have the same data,
        # so we just take from the first device
        return jax.tree.map(lambda x: np.asarray(x), pytree)

    def save_checkpoint(self):
        self._assert_initialized()

        # Convert to numpy arrays for checkpointing
        model_to_save = self._unreplicate_for_checkpoint(self.model)
        opt_state_to_save = self._unreplicate_for_checkpoint(self.opt_state)

        args = {
            "model": ocp.args.StandardSave(model_to_save),
            "opt_state": ocp.args.StandardSave(opt_state_to_save),
        }
        # Only the train dataset iterator is checkpointed
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

        # Restore to numpy first
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

        # Re-replicate the restored state
        self.model = jax.device_put(restored["model"], self.replicated_sharding)
        self.opt_state = jax.device_put(restored["opt_state"], self.replicated_sharding)

        if self.checkpoint_config.checkpoint_dataset_iterator:
            self.train_task.dataset.restore_state(restored["dataset_iterator"])
        self.step = step

    def run_validation(self) -> dict[str, float]:
        """Run validation and compute all metrics for the eval task.

        Returns:
            Dictionary of averaged metrics
        """
        self._assert_initialized()
        if self.eval_task is None:
            print("No eval task defined, skipping validation.")
            return {}

        # Accumulate metrics across batches
        accumulated_metrics = None
        num_batches = 0

        for ix, batch in enumerate(self.eval_task.dataset):
            batch_metrics = self.eval_step_fn(self.model, batch)

            # Accumulate metrics using tree_map
            if accumulated_metrics is None:
                accumulated_metrics = batch_metrics
            else:
                accumulated_metrics = jax.tree.map(
                    lambda x, y: x + y, accumulated_metrics, batch_metrics
                )

            num_batches += 1
            if ix >= self.eval_config.num_eval_steps:
                break

        # Average all metrics and add val/ prefix
        avg_metrics = jax.tree.map(
            lambda x: float(x) / num_batches, accumulated_metrics
        )
        avg_metrics = {f"val/{key}": value for key, value in avg_metrics.items()}

        self.logger_obj.log(self.step, metrics=avg_metrics)
        return avg_metrics

    def sampling_evaluation(
        self, prompts: list[str] | None = None, max_tokens: int = 10
    ) -> None:
        """Run sampling evaluation (runs on single device for simplicity).

        Args:
            prompts: List of string prompts to evaluate.
        """
        self._assert_initialized()
        if self.train_task.dataset.tokenizer is None:
            return
        if prompts is None:
            prompts = DEFAULT_PROMPTS

        tokenized_prompts = self.train_task.dataset.tokenizer(
            prompts,
            return_tensors=None,
            padding=False,
            truncation=False,
            max_length=None,
        )["input_ids"]

        seq_len = self.model_config.seq_len
        results = []
        for ix, prompt_tokens in enumerate(tokenized_prompts):
            padded = jnp.zeros(seq_len, dtype=jnp.uint16)
            padded = padded.at[: len(prompt_tokens)].set(
                jnp.array(prompt_tokens, dtype=jnp.uint16)
            )
            generated = decoder_only_model.sample(
                model=self.model,
                input_tokens=padded,
                prompt_len=len(prompt_tokens),
                key=jax.random.key(0),
                max_output_tokens=max_tokens,
                temperature=1.0,
                top_k=5,
            )
            results.append(
                {
                    "prompt": prompts[ix],
                    "output": self.train_task.dataset.tokenizer.decode(
                        generated.tolist()
                    ),
                }
            )
        self.logger_obj.log(self.step, metrics={"step": self.step, "samples": results})

    def eval(self):
        self._assert_initialized()
        self.run_validation()
        self.sampling_evaluation()

    def inner_loop(self, batch: dict):
        self._assert_initialized()

        # Compute loss and gradients
        self.model, self.opt_state, train_metrics = self.train_step_fn(
            self.model, self.opt_state, batch
        )

        # Log metrics
        if self.step % self.logger_config.train_log_interval_steps == 0:
            # Add train/ prefix and convert to float
            train_metrics_with_prefix = {
                f"train/{key}": float(value) for key, value in train_metrics.items()
            }
            self.logger_obj.log(self.step, metrics=train_metrics_with_prefix)

    def outer_loop(self):
        finished = self.step >= self.training_config.max_steps

        while True:
            epoch_start_step = self.step
            for batch in self.train_task.dataset:
                # Perform inner loop step
                with jax.profiler.StepTraceAnnotation("inner_loop", step_num=self.step):
                    self.inner_loop(batch)

                if self.step % self.checkpoint_config.save_interval_steps == 0:
                    self.save_checkpoint()

                if self.step % self.eval_config.eval_interval_steps == 0:
                    self.eval()

                # Increment step
                self.step += 1

                if self.step >= self.training_config.max_steps:
                    finished = True
                    break

            if finished:
                break

            if self.step == epoch_start_step:
                # Dataset is empty!
                raise ValueError(f"Dataset for task {self.train_task.name} is empty.")

    # TODO: how to ensure this is always called?
    def cleanup(self):
        self.logger_obj.close()
        self.ckpt_manager.close()
