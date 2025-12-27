"""Basic types for the training loop and configurations."""

import dataclasses
import jax
import optax
import orbax.checkpoint as ocp
import typing

from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import data
from toylib_projects.tinystories import logger


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
class TrainingConfig:
    learning_rate: float = 1e-3
    max_steps: int = 100_000


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


@dataclasses.dataclass(kw_only=True)
class LoggerConfig:
    logger_cls: logger.Logger = logger.FileLogger
    log_dir: str = "/tmp/train_logs.txt"

    train_log_interval_steps: int = 1


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

            with jax.profiler.TraceAnnotation("value_and_grad"):
                # Compute loss and gradients
                (loss_val, _), grads = jax.value_and_grad(
                    self.forward_fn, has_aux=True
                )(model, inputs, mask, targets)

            with jax.profiler.TraceAnnotation("optimizer_update"):
                # Apply gradients
                updates, opt_state = self.optimizer.update(grads, opt_state)

                # Update the model and optimizer state
                model = optax.apply_updates(model, updates)

            return model, opt_state, loss_val

        def eval_step(model, batch):
            inputs, targets = batch["inputs"], batch["targets"]
            mask = jax.numpy.ones_like(inputs)

            with jax.profiler.TraceAnnotation("eval_forward"):
                loss_val, _ = self.forward_fn(model, inputs, mask, targets)

            return loss_val

        if self.jit_computations:
            self.train_step_fn = jax.jit(train_step)
            self.eval_step_fn = jax.jit(eval_step)
        else:
            self.train_step_fn = train_step
            self.eval_step_fn = eval_step

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
        args = {
            "model": ocp.args.StandardSave(self.model),
            "opt_state": ocp.args.StandardSave(self.opt_state),
        }
        # Only the train dataset iterator is checkpointed
        if self.checkpoint_config.checkpoint_dataset_iterator:
            args["dataset_iterator"] = ocp.args.StandardSave(
                self.train_task.dataset.get_state()
            )
        self.ckpt_manager.save(
            self.step,
            args=ocp.args.Composite(**args),
        )
        self.ckpt_manager.wait_until_finished()

    def restore_checkpoint(self, step: int):
        self._assert_initialized()
        args = {
            "model": ocp.args.StandardRestore(self.model),
            "opt_state": ocp.args.StandardRestore(self.opt_state),
        }
        if self.checkpoint_config.checkpoint_dataset_iterator:
            args["dataset_iterator"] = ocp.args.StandardRestore(
                self.train_task.dataset.get_state()
            )

        restored = self.ckpt_manager.restore(step, args=ocp.args.Composite(**args))
        # Update the local state
        self.model = restored["model"]
        self.opt_state = restored["opt_state"]
        if self.checkpoint_config.checkpoint_dataset_iterator:
            self.train_task.dataset.restore_state(restored["dataset_iterator"])
        self.step = step

    def run_validation(self) -> float:
        self._assert_initialized()
        if self.eval_task is None:
            print("No eval task defined, skipping validation loss.")
            return

        total_val_loss = 0.0
        for ix, batch in enumerate(self.eval_task.dataset):
            val_loss = self.eval_step_fn(self.model, batch)
            total_val_loss += val_loss
            if ix >= self.eval_config.num_eval_steps:
                break

        avg_val_loss = total_val_loss / (ix + 1)
        self.logger_obj.log(self.step, metrics={"val/loss": float(avg_val_loss)})

    def sampling_evaluation(
        self, prompts: list[str] | None = None, max_tokens: int = 10
    ) -> None:
        """Run sampling evaluation on the model given a list of prompts.

        Args:
            prompts: List of string prompts to evaluate.
        """
        self._assert_initialized()
        if prompts is None:
            prompts = DEFAULT_PROMPTS

        results = []
        tokenized_prompts = self.train_task.dataset.tokenizer(
            prompts,
            return_tensors=None,  # return a list of lists
            padding=False,
            truncation=False,
            max_length=None,
        )["input_ids"]
        for ix in range(len(tokenized_prompts)):
            generated = list(
                decoder_only_model.sample(
                    model=self.model,
                    input_tokens=tokenized_prompts[ix],
                    key=jax.random.PRNGKey(0),
                    max_output_tokens=max_tokens,
                    temperature=1.0,
                    top_k=5,
                )
            )
            results.append(
                {
                    "prompt": prompts[ix],
                    "output": self.train_task.dataset.tokenizer.decode(generated),
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
        self.model, self.opt_state, loss_val = self.train_step_fn(
            self.model, self.opt_state, batch
        )

        # Log metrics
        if self.step % self.logger_config.train_log_interval_steps == 0:
            self.logger_obj.log(self.step, metrics={"train/loss": float(loss_val)})

        # Increment step
        self.step += 1

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
