"""Figures for the post, from checkpoints in the GCS bucket.

    GOOGLE_CLOUD_PROJECT=practicalml-db49f uv run python -m toylib_projects.p01_cnn_mnist.figures

Writes PNGs to ``outputs/figures``.
"""

import pathlib

import jax
import matplotlib.pyplot as plt
import numpy as np

from toylib import viz
from toylib.harness import inference
from toylib_projects.p01_cnn_mnist import imagenette, model

CHECKPOINT_ROOT = "gs://tinystories-checkpoints/toylib-p01-imagenette"
IMAGE_SHAPE = (160, 160, 3)
OUTPUT_DIR = pathlib.Path(model.__file__).parent / "outputs" / "figures"


def as_images(weights: np.ndarray) -> np.ndarray:
    """``(H * W * C, N)`` weights -> ``(N, H, W, C)`` images, each scaled about grey."""
    images = np.moveaxis(weights.reshape(*IMAGE_SHAPE, -1), -1, 0)
    scale = np.abs(images).max(axis=(1, 2, 3), keepdims=True)
    return images / (2 * scale) + 0.5


def linear_templates() -> plt.Figure:
    """The one image per class that the linear model learned."""
    template = model.build_mlp(model.MLPConfig(IMAGE_SHAPE), jax.random.key(0))
    net = inference.load_model(template, f"{CHECKPOINT_ROOT}/linear_e200")
    return viz.image_grid(
        as_images(np.asarray(net.mlp.output_layer.weights)),
        titles=imagenette.CLASS_NAMES,
        suptitle="Linear model: one template per class",
    )


def mlp_hidden_weights(count: int = 32) -> plt.Figure:
    """The hidden units with the largest weights, as images."""
    config = model.MLPConfig(IMAGE_SHAPE, hidden_dims=(256,))
    template = model.build_mlp(config, jax.random.key(0))
    net = inference.load_model(template, f"{CHECKPOINT_ROOT}/mlp_e200")
    weights = np.asarray(net.mlp.hidden_layers[0].weights)
    largest = np.argsort(-np.linalg.norm(weights, axis=0))[:count]
    return viz.image_grid(
        as_images(weights[:, largest]),
        suptitle="MLP: the 32 hidden units with the largest weights",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    viz.use_style()
    viz.save(linear_templates(), OUTPUT_DIR / "linear_templates")
    viz.save(mlp_hidden_weights(), OUTPUT_DIR / "mlp_hidden_weights")


if __name__ == "__main__":
    main()
