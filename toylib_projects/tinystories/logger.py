import abc
import datetime
import json
import os


class Logger(abc.ABC):
    """Interface for logging training metrics."""

    def __init__(self, config_dict: dict, *args, **kwargs) -> None:
        self.config_dict = config_dict

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


# class WandBLogger(Logger):
#     """Logger implementation using Weights and Biases (wandb)."""

#     def __init__(
#         self, config_dict: dict, project_name: str, user_name: str, *args, **kwargs
#     ) -> None:
#         import wandb

#         self.config_dict = config_dict
#         self.run = wandb.init(
#             entity=user_name,
#             project=project_name,
#             config=self.config_dict,
#         )
#         self.run.define_metric("*", step_metric="global_step")

#     def log(self, step: int, metrics: dict) -> None:
#         metrics["global_step"] = step
#         self.run.log(metrics

#     def close(self) -> None:
#         self.run.finish()


# class TensorBoardLogger(Logger):
#     """Logger implementation that logs metrics to tensorboard locally."""

#     def __init__(self, config_dict: dict, output_path: str, *args, **kwargs) -> None:
#         import os
#         from tensorboardX import SummaryWriter
#         import time

#         self.config_dict = config_dict
#         self.writer = SummaryWriter(
#             logdir=os.path.join(output_path, time.strftime("%Y%m%d-%H%M%S"))
#         )

#     def log(self, step: int, metrics: dict, tag: str = "train") -> None:
#         self.writer.add_scalars(tag, metrics, step)

#     def close(self) -> None:
#         self.writer.close()


class FileLogger(Logger):
    """Logger implementation that logs metrics to a local file."""

    def __init__(self, config_dict: dict, output_path: str, *args, **kwargs) -> None:
        self.config_dict = config_dict
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file_ptr = open(os.path.join(output_path, f"logs_{timestamp}.txt"), "w")
        self.file_ptr.write("\n")

    def log(self, step: int, metrics: dict) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["timestamp"] = timestamp
        metrics["step"] = step
        self.file_ptr.write(json.dumps(metrics) + "\n")
        self.file_ptr.flush()

    def close(self) -> None:
        self.file_ptr.close()


class StdoutLogger(Logger):
    """Logger implementation that logs metrics to standard output."""

    def __init__(self, config_dict: dict, *args, **kwargs) -> None:
        self.config_dict = config_dict

    def log(self, step: int, metrics: dict) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Step {step}: {metrics}")

    def close(self) -> None:
        pass
