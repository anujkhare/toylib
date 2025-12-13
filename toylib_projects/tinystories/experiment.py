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
    batch_size: int = 128
    learning_rate: float = 1e-3
    max_steps: int = 100_000


@dataclasses.dataclass
class Task:
    name: str
    dataset: data.BatchedTokenizedDataset


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

    log_dir: str = "/tmp/tensorboard_logs/"

    def __post_init__(self):
        # Logger
        self.logger_obj = logger.TensorBoardLogger(
            config_dict=_serlialize_dataclass_config(self),
            output_path=self.log_dir,
        )

        # Optimizer
        self.optimizer = optax.adam(learning_rate=self.training_config.learning_rate)
        self.opt_state = None
        self.model = None

        # Value and gradient function
        self.loss_and_grad_fn = jax.jit(
            jax.value_and_grad(decoder_only_model.train_step)
        )

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
                model=ocp.args.StandardRestore(),
                opt_state=ocp.args.StandardRestore(),
            ),
        )
        # Update the local state
        self.model = restored["model"]
        self.opt_state = restored["opt_state"]
        self.step = step

    def log_metrics(self, step: int, loss_val: float, updates):
        leaves, _ = jax.tree_util.tree_flatten(updates)
        metrics = {
            "train/loss": float(loss_val),
            "train/learning_rate": self.training_config.learning_rate,
            "gradients/0/mean": leaves[0].mean(),
            "gradients/1/mean": leaves[1].mean(),
            "gradients/2/mean": leaves[2].mean(),
        }
        self.logger_obj.log(step=step, metrics=metrics)

    def inner_loop(self, batch: dict):
        self._assert_initialized()
        inputs, targets = batch["inputs"], batch["targets"]
        mask = jax.numpy.ones_like(inputs)

        # Compute loss and gradients
        loss_val, grads = self.loss_and_grad_fn(self.model, inputs, mask, targets)

        # Apply gradients
        updates, opt_state = self.optimizer.update(grads, self.opt_state)

        # Update the model and optimizer state
        self.model = optax.apply_updates(self.model, updates)
        self.opt_state = opt_state

        # Log metrics
        self.log_metrics(self.step, loss_val, updates)

    def outer_loop(self):
        # Training loop
        while self.step < self.training_config.max_steps:
            for batch in self.train_task.dataset:
                # Perform inner loop step
                self.inner_loop(batch)

                # Increment step
                self.step += 1

                if self.step % self.checkpoint_config.save_interval_steps == 0:
                    self.save_checkpoint()

    # TODO: how to ensure this is always called?
    def cleanup(self):
        self.logger_obj.close()
        self.ckpt_manager.close()
