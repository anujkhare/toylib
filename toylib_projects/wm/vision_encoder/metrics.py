"""Generic metric scaffolding for the vision-encoder training loop.

Forked from `toylib_projects/tinystories/metrics.py`, keeping only the
domain-agnostic pieces:

  - ``Metric`` Protocol — interface every metric implements.
  - ``Loss``            — pass-through metric that surfaces the forward-fn loss.

The text-specific ``BitsPerByte`` is intentionally not forked. Image-specific
metrics (PSNR, SSIM, reconstruction MSE per pixel, per-mode FID, etc.) are
deferred to the model-author — they're not infrastructure.
"""

import dataclasses
import typing

import jaxtyping as jt


class Metric(typing.Protocol):
    """Protocol for computing and accumulating metrics."""

    def __call__(
        self,
        loss: float,
        aux: jt.PyTree,
        batch: jt.PyTree,
    ) -> dict[str, jt.Array]:
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
        loss: float,
        aux: jt.PyTree,
        batch: jt.PyTree,
    ) -> dict[str, jt.Array]:
        del aux, batch
        return {"loss": loss}
