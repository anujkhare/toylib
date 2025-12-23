"""Basic types for the training loop and configurations."""

import dataclasses
import jax
import optax
import orbax.checkpoint as ocp
import typing

from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import data
from toylib_projects.tinystories import logger


@dataclasses.dataclass
class CheckpointConfig:
    save_interval_steps: int = 5_000
    max_to_keep: typing.Optional[int] = 10
    checkpoint_dir: str = "/tmp/checkpoints"


@dataclasses.dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    max_steps: int = 100_000


@dataclasses.dataclass
class Task:
    name: str
    dataset: data.BatchedTokenizedDataset


@dataclasses.dataclass(kw_only=True)
class LoggerConfig:
    logger_cls: logger.Logger = logger.FileLogger
    log_dir: str = "/tmp/train_logs.txt"


def _serlialize_dataclass_config(config: dataclasses.dataclass) -> dict:
    result = dataclasses.asdict(config)
    for k, v in result.items():
        if dataclasses.is_dataclass(v):
            result[k] = _serlialize_dataclass_config(v)
    return result


@dataclasses.dataclass
class Experiment:
    """Base Experiment class."""

    # Tasks to train and evaluate
    train_task: Task
    eval_tasks: list[Task] = dataclasses.field(default_factory=list)

    # Model config
    model_config: decoder_only_model.ModelConfig = dataclasses.field(
        default_factory=decoder_only_model.ModelConfig
    )
    # Training Hyperparameters
    training_config: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
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
    jit_train_fn: bool = True

    def __post_init__(self):
        # Logger
        self.logger_obj = self.logger_config.logger_cls(
            config_dict=_serlialize_dataclass_config(self),
            output_path=self.logger_config.log_dir,
        )

        # Optimizer
        self.optimizer = optax.adam(learning_rate=self.training_config.learning_rate)
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

        # Train step
        def train_step(model, opt_state, batch):
            inputs, targets = batch["inputs"], batch["targets"]
            mask = jax.numpy.ones_like(inputs)

            # Compute loss and gradients
            (loss_val, _), grads = jax.value_and_grad(self.forward_fn, has_aux=True)(
                model, inputs, mask, targets
            )

            # Apply gradients
            updates, opt_state = self.optimizer.update(grads, opt_state)

            # Update the model and optimizer state
            model = optax.apply_updates(model, updates)

            return model, opt_state, loss_val

        if self.jit_train_fn:
            self.train_step_fn = jax.jit(train_step)
        else:
            self.train_step_fn = train_step

    def init_state(self):
        # Model
        self.model = decoder_only_model.DecoderOnlyTransformer(
            config=self.model_config, key=jax.random.PRNGKey(0)
        )

        # Optimizer
        self.opt_state = self.optimizer.init(self.model)

        # Global step
        self.step = 0

    def _assert_initialized(self) -> bool:
        initialized = self.model is not None and self.opt_state is not None
        assert initialized, "Experiment state not initialized. Call init_state() first."

    def save_checkpoint(self):
        self._assert_initialized()
        self.ckpt_manager.save(
            self.step,
            args=ocp.args.Composite(
                model=ocp.args.StandardSave(self.model),
                opt_state=ocp.args.StandardSave(self.opt_state),
            ),
        )
        self.ckpt_manager.wait_until_finished()

    def restore_checkpoint(self, step: int):
        self._assert_initialized()
        restored = self.ckpt_manager.restore(
            step,
            args=ocp.args.Composite(
                model=ocp.args.StandardRestore(self.model),
                opt_state=ocp.args.StandardRestore(self.opt_state),
            ),
        )
        # Update the local state
        self.model = restored["model"]
        self.opt_state = restored["opt_state"]
        self.step = step

    def log_metrics(self, step: int, loss_val: float):
        metrics = {
            "train/loss": float(loss_val),
            "train/learning_rate": self.training_config.learning_rate,
        }
        self.logger_obj.log(step=step, metrics=metrics)

    def inner_loop(self, batch: dict):
        self._assert_initialized()

        # Compute loss and gradients
        self.model, self.opt_state, loss_val = self.train_step_fn(
            self.model, self.opt_state, batch
        )

        # Log metrics
        self.log_metrics(self.step, loss_val)

    def outer_loop(self):
        finished = self.step >= self.training_config.max_steps

        while True:
            epoch_start_step = self.step
            for batch in self.train_task.dataset:
                # Perform inner loop step
                self.inner_loop(batch)

                # Increment step
                self.step += 1

                if self.step % self.checkpoint_config.save_interval_steps == 0:
                    self.save_checkpoint()

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
