"""Basic types for the training loop and configurations."""

import dataclasses
import jax
import optax

from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import data
from toylib_projects.tinystories import logger


@dataclasses.dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 1


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

    model_config: decoder_only_model.ModelConfig = dataclasses.field(
        default_factory=decoder_only_model.ModelConfig
    )
    training_config: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    def __post_init__(self):
        # Logger
        self.logger_obj = logger.TensorBoardLogger(
            self,
            config_dict=_serlialize_dataclass_config(self),
            output_path="/tmp/tensorboard_logs/",
        )

        # Optimizer
        self.optimizer = optax.adam(learning_rate=self.training_config.learning_rate)
        self.opt_state = None
        self.model = None

    def init_state(self):
        # Model
        self.model = decoder_only_model.DecoderOnlyTransformer(
            config=self.model_config, key=jax.random.PRNGKey(0)
        )

        # Optimizer
        self.opt_state = self.optimizer.init(self.model)

        # Global step
        self.step = 0

        # Value and gradient function
        self.loss_and_grad_fn = jax.jit(
            jax.value_and_grad(decoder_only_model.train_step)
        )

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

    def _is_initialized(self) -> bool:
        return self.model is not None and self.opt_state is not None

    def inner_loop(self, batch: dict):
        assert self._is_initialized(), (
            "Experiment state not initialized. Call init_state() first."
        )

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
        for _ in range(self.training_config.num_epochs):
            for batch in self.train_task.dataset:
                # Perform inner loop step
                self.inner_loop(batch)

                # Increment step
                self.step += 1
