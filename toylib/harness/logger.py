"""Metric loggers for the training harness.

Three sinks are provided:

  - ``StdoutLogger`` — prints to stdout (great for local development)
  - ``FileLogger``   — appends JSON-lines to a file under ``output_path``
  - ``WandBLogger``  — lazily imports wandb; for cloud runs

All three share the ``Logger`` ABC so the experiment can swap them via
config without touching the training loop. ``MultiLogger`` sends everything to
several sinks at once, e.g. a local file and wandb.
"""

import abc
import datetime
import json
import os

import numpy as np


class Logger(abc.ABC):
    """Interface for logging training metrics."""

    def __init__(self, config_dict: dict, *args, **kwargs) -> None:
        self.config_dict = config_dict

    @abc.abstractmethod
    def log(self, step: int, metrics: dict) -> None:
        """Log the given metrics at the specified step."""
        pass

    def log_images(self, step: int, key: str, images: np.ndarray) -> None:
        """Log a uint8 (N, H, W, 3) image batch. No-op if not overridden."""
        pass

    def log_checkpoint(self, step: int, path: str) -> None:
        """Keep a copy of the checkpoint saved at ``path``. No-op if not overridden."""
        pass

    def fetch_checkpoint(self, checkpoint_dir: str) -> int | None:
        """Put the newest kept checkpoint under ``checkpoint_dir``; return its step.

        Returns None if there is none. No-op if not overridden.
        """
        return None

    @abc.abstractmethod
    def close(self) -> None:
        """Close any resources held by the logger."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class WandBLogger(Logger):
    """Logger implementation using Weights and Biases (wandb).

    With ``upload_checkpoints=True``, every checkpoint is stored as a version of
    the artifact ``<run id>-checkpoints``, so training can continue from another
    machine: a run started with the same ``run_id`` resumes the wandb run and
    ``fetch_checkpoint`` finds its newest checkpoint.
    """

    def __init__(
        self,
        config_dict: dict,
        project_name: str,
        user_name: str,
        run_id: str | None = None,
        output_path: str | None = None,
        upload_checkpoints: bool = False,
        *args,
        **kwargs,
    ) -> None:
        import wandb

        self.config_dict = config_dict
        self.upload_checkpoints = upload_checkpoints
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        self.run = wandb.init(
            entity=user_name or None,  # None is the account's default entity.
            project=project_name,
            config=self.config_dict,
            id=run_id,
            dir=output_path,
            resume="allow",
        )
        self.run.define_metric("*", step_metric="global_step")

    def log(self, step: int, metrics: dict) -> None:
        metrics["global_step"] = step
        self.run.log(metrics)

    def log_images(self, step: int, key: str, images: np.ndarray) -> None:
        import wandb
        self.run.log({key: [wandb.Image(img) for img in images], "global_step": step})

    def log_checkpoint(self, step: int, path: str) -> None:
        if not self.upload_checkpoints:
            return
        import wandb

        artifact = wandb.Artifact(
            f"{self.run.id}-checkpoints", type="checkpoint", metadata={"step": step}
        )
        artifact.add_dir(path)
        # Uploads in the background; run.finish() waits for it. wandb moves the
        # "latest" alias to each new version.
        self.run.log_artifact(artifact, aliases=[f"step-{step}"])

    def fetch_checkpoint(self, checkpoint_dir: str) -> int | None:
        if not self.upload_checkpoints or self.run.offline:
            return None
        import wandb

        try:
            artifact = self.run.use_artifact(f"{self.run.id}-checkpoints:latest")
        except (ValueError, wandb.errors.CommError) as error:
            # A network failure must not pass for "nothing uploaded yet": the run
            # would silently start again from step 0.
            if "not found" not in str(error):
                raise
            return None
        step = int(artifact.metadata["step"])
        artifact.download(root=os.path.join(checkpoint_dir, str(step)))
        return step

    def close(self) -> None:
        self.run.finish()


class FileLogger(Logger):
    """Logger implementation that logs metrics to a local file."""

    def __init__(
        self,
        config_dict: dict,
        output_path: str,
        run_id: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.config_dict = config_dict
        label = run_id or datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        os.makedirs(output_path, exist_ok=True)
        # Append, so a resumed run keeps the log it has so far.
        self.file_ptr = open(os.path.join(output_path, f"logs_{label}.txt"), "a")
        self.file_ptr.write("\n")

    def log(self, step: int, metrics: dict) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["timestamp"] = timestamp
        metrics["step"] = step
        self.file_ptr.write(json.dumps(metrics, default=str) + "\n")
        self.file_ptr.flush()

    def log_images(self, step: int, key: str, images: np.ndarray) -> None:
        try:
            import math
            from PIL import Image as PILImage
            n, h, w, c = images.shape
            ncols = min(n, 4)
            nrows = math.ceil(n / ncols)
            grid = np.zeros((nrows * h, ncols * w, c), dtype=np.uint8)
            for i, img in enumerate(images):
                r, col = divmod(i, ncols)
                grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = img
            out_dir = os.path.dirname(self.file_ptr.name)
            fname = f"step{step:07d}_{key.replace('/', '_')}.png"
            PILImage.fromarray(grid).save(os.path.join(out_dir, fname))
        except ImportError:
            pass

    def close(self) -> None:
        self.file_ptr.close()


class StdoutLogger(Logger):
    """Logger implementation that logs metrics to standard output."""

    def __init__(self, config_dict: dict, *args, **kwargs) -> None:
        self.config_dict = config_dict

    def log(self, step: int, metrics: dict) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Step {step}: {metrics}")

    def log_images(self, step: int, key: str, images: np.ndarray) -> None:
        print(f"[Step {step}] {key}: {images.shape} uint8 images")

    def close(self) -> None:
        pass


class MultiLogger(Logger):
    """Send everything to several loggers."""

    def __init__(self, loggers: list[Logger]) -> None:
        self.loggers = loggers

    def log(self, step: int, metrics: dict) -> None:
        for sink in self.loggers:
            sink.log(step, dict(metrics))  # Sinks stamp their own fields on it.

    def log_images(self, step: int, key: str, images: np.ndarray) -> None:
        for sink in self.loggers:
            sink.log_images(step, key, images)

    def log_checkpoint(self, step: int, path: str) -> None:
        for sink in self.loggers:
            sink.log_checkpoint(step, path)

    def fetch_checkpoint(self, checkpoint_dir: str) -> int | None:
        for sink in self.loggers:
            step = sink.fetch_checkpoint(checkpoint_dir)
            if step is not None:
                return step
        return None

    def close(self) -> None:
        for sink in self.loggers:
            sink.close()
