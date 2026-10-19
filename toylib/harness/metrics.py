"""Metric interfaces for the training harness.

The domain-agnostic pieces shared by every model trained through ``Experiment``:

  - ``Metric`` Protocol              — interface every scalar metric implements.
  - ``Loss``                         — pass-through metric that surfaces the
                                       forward-fn loss.
  - ``VisualizationMetric`` Protocol — interface for generative image metrics
                                       that run outside JIT.

Model-specific metrics (accuracy, bits per byte, reconstruction error, ...)
live next to their model, since they're not infrastructure.
"""

import dataclasses
import typing

import jaxtyping as jt
import numpy as np


@typing.runtime_checkable
class Metric(typing.Protocol):
    """Protocol for computing and accumulating metrics."""

    def __call__(
        self,
        loss: jt.ArrayLike,
        aux: jt.PyTree,
        batch: jt.PyTree,
    ) -> dict[str, jt.ArrayLike]:
        """Compute final metric value(s) for the given inputs.

        Args:
            loss: The loss value returned by forward_fn.
            aux: The auxiliary jt.PyTree returned by forward_fn.
            batch: The input batch.
        """
        pass


@dataclasses.dataclass
class Loss:
    """Pass-through metric that returns the loss value."""

    def __call__(
        self,
        loss: jt.ArrayLike,
        aux: jt.PyTree,
        batch: jt.PyTree,
    ) -> dict[str, jt.ArrayLike]:
        del aux, batch
        return {"loss": loss}


@typing.runtime_checkable
class VisualizationMetric(typing.Protocol):
    """Protocol for generative image metrics that run outside JIT.

    Unlike ``Metric``, these don't consume an input batch — they generate images
    directly from the model (e.g. decoding random latents). Each call returns a
    dict of ``{name: (N, H, W, 3) uint8 ndarray}`` logged via ``logger.log_images``.

    Metrics that read real frames + their model outputs from ``aux`` (e.g. a VAE
    reconstruction comparison) should instead implement the standard ``Metric``
    protocol so they run inside the JIT-compiled eval step.
    """

    def __call__(
        self,
        model: typing.Any,
    ) -> dict[str, np.ndarray]:
        ...
