"""Vision-encoder (VAE) specific metrics.

The domain-agnostic ``Metric`` / ``Loss`` / ``VisualizationMetric`` scaffolding
lives in ``toylib_projects.wm.metrics``. This module holds the image metrics
that are specific to the VAE codec:

  - ``ReconstructionVisualization`` — side-by-side input vs. reconstruction,
    implementing the ``Metric`` protocol (runs inside the JIT eval step via aux).
  - ``PriorSamplingVisualization``  — images decoded from random prior latents,
    implementing the ``VisualizationMetric`` protocol (runs outside JIT).
"""

import dataclasses
import typing

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np


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
    gap_px: int = 4

    def __call__(
        self,
        loss: float,
        aux: jt.PyTree,
        batch: jt.PyTree,
    ) -> dict[str, jt.Array]:
        del loss
        inputs = batch[: self.num_images]  # (N, H, W, 3) uint8
        recon_f32 = aux[self.recon_aux_key][: self.num_images]  # float32 [-1, 1]
        recons = ((recon_f32 + 1.0) * 127.5).clip(0, 255).astype(jnp.uint8)
        n, h = inputs.shape[:2]
        gap = jnp.full((n, h, self.gap_px, 3), 128, dtype=jnp.uint8)
        comparison = jnp.concatenate([inputs, gap, recons], axis=2)
        return {"recon_comparison": comparison}


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
