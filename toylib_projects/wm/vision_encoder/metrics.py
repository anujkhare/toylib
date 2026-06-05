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

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np


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


class VisualizationMetric(typing.Protocol):
    """Protocol for generative image metrics that run outside JIT.

    Unlike ``Metric``, these don't consume an input batch — they generate images
    directly from the model (e.g. decoding random latents). Each call returns a
    dict of ``{name: (N, H, W, 3) uint8 ndarray}`` logged via ``logger.log_images``.

    For reconstruction visualization (encoding then decoding real frames), use
    ``ReconstructionVisualization`` which implements the standard ``Metric``
    protocol and runs inside the JIT-compiled eval step via ``aux``.
    """

    def __call__(
        self,
        model: typing.Any,
    ) -> dict[str, np.ndarray]:
        ...


@dataclasses.dataclass
class ReconstructionVisualization:
    """Return input frames and their VAE reconstructions from the eval forward pass.

    Implements the standard ``Metric`` protocol so it runs inside the JIT-compiled
    eval step. The eval ``forward_fn`` must include ``"recon"`` in the returned
    ``aux`` dict (float32, ``[-1, 1]``). Both outputs are converted to uint8
    ``[0, 255]`` before being returned for logging.

    Args:
        recon_aux_key: Key in ``aux`` where the eval forward_fn stores reconstructed
            frames as float32 in ``[-1, 1]``.
        num_images: How many images from the batch to return.
    """

    recon_aux_key: str = "recon"
    num_images: int = 8

    def __call__(
        self,
        loss: float,
        aux: jt.PyTree,
        batch: jt.PyTree,
    ) -> dict[str, jt.Array]:
        del loss
        inputs = batch[: self.num_images]  # already uint8 [0, 255]
        recon_f32 = aux[self.recon_aux_key][: self.num_images]  # float32 [-1, 1]
        recons = ((recon_f32 + 1.0) * 127.5).clip(0, 255).astype(jnp.uint8)
        return {"input_images": inputs, "recon_images": recons}


@dataclasses.dataclass
class PriorSamplingVisualization:
    """Log images decoded from randomly sampled latents.

    The PRNG key is derived from ``seed`` and held fixed across evals so
    outputs are directly comparable over the course of training.

    Args:
        sample_fn: ``(model, key, n) -> images`` where ``images`` is
            ``(n, H, W, 3)`` uint8.
        num_samples: Number of images to generate.
        seed: Fixed seed for the sampling key.
    """

    sample_fn: typing.Callable[..., np.ndarray]
    num_samples: int = 16
    seed: int = 42

    def __call__(
        self,
        model: typing.Any,
    ) -> dict[str, np.ndarray]:
        key = jax.random.key(self.seed)
        samples = self.sample_fn(model, key, self.num_samples)
        return {"prior_samples": np.asarray(samples)}
