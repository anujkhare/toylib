"""Basic types for the training loop and configurations."""

import abc
import dataclasses

from toylib_projects.tinystories import decoder_only_model


@dataclasses.dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 1


@dataclasses.dataclass
class Config:
    """Configuration for the experiment."""

    model_config: decoder_only_model.ModelConfig = dataclasses.field(
        default_factory=decoder_only_model.ModelConfig
    )
    training_config: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)


def _serlialize_dataclass_config(config: Config) -> dict:
    result = dataclasses.asdict(config)
    for k, v in result.items():
        if dataclasses.is_dataclass(v):
            result[k] = _serlialize_dataclass_config(v)
    return result


class Logger(abc.ABC):
    """Interface for logging training metrics."""

    def __init__(self, config: Config, *args, **kwargs) -> None:
        self.config = config

    @abc.abstractmethod
    def log(self, step: int, metrics: dict) -> None:
        """Log the given metrics at the specified step."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Close any resources held by the logger."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class WandBLogger(Logger):
    """Logger implementation using Weights and Biases (wandb)."""

    def __init__(
        self, config: Config, project_name: str, user_name: str, *args, **kwargs
    ) -> None:
        import wandb

        self.config = config
        self.run = wandb.init(
            entity=user_name,
            project=project_name,
            config=_serlialize_dataclass_config(self.config),
        )
        self.run.define_metric("*", step_metric="global_step")

    def log(self, step: int, metrics: dict) -> None:
        metrics["global_step"] = step
        self.run.log(metrics)

    def close(self) -> None:
        self.run.finish()


class TensorBoardLogger(Logger):
    """Logger implementation that logs metrics to tensorboard locally."""

    def __init__(self, config: Config, output_path: str, *args, **kwargs) -> None:
        import os
        from tensorboardX import SummaryWriter
        import time

        self.config = config
        self.writer = SummaryWriter(
            logdir=os.path.join(output_path, time.strftime("%Y%m%d-%H%M%S"))
        )

    def log(self, step: int, metrics: dict, tag: str = "train") -> None:
        self.writer.add_scalars(tag, metrics, step)

    def close(self) -> None:
        self.writer.close()


class FileLogger(Logger):
    """Logger implementation that logs metrics to a local file."""

    def __init__(self, config: Config, output_path: str, *args, **kwargs) -> None:
        import json

        self.config = config
        self.file_ptr = open(output_path, "w")
        json.dump(_serlialize_dataclass_config(self.config), self.file_ptr, indent=4)

    def log(self, step: int, metrics: dict) -> None:
        self.temp_file.write(f"Step {step}: {metrics}\n")

    def close(self) -> None:
        self.file_ptr.close()
