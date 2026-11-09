"""Helpers for tests: a stand-in model and synthetic Imagenette.

The tests of the data, training and notebook code use these, so they pass before
the ViT exists and don't depend on what it learns.

* ``StandIn``: average each image's colour, then one linear layer. It has the
  ViT's interface, so ``build_stand_in`` replaces ``model.build_model``.
* ``cross_entropy``: replaces ``model.loss_fn``.
* ``SyntheticImagenette(n_train, n_test)``: a loader shaped like ``imagenette.load``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import optax

from toylib.nn import layers
from toylib.nn import module
from toylib_projects.p03_vit import data
from toylib_projects.p03_vit import imagenette
from toylib_projects.p03_vit import model


class StandIn(module.Module):
    num_classes: int
    key: jt.PRNGKeyArray

    def init(self) -> None:
        self.head = layers.Linear(
            in_features=3, out_features=self.num_classes, use_bias=True, key=self.key
        )

    def __call__(
        self, images: jt.Float[jt.Array, "B H W C"], return_intermediates: bool = False
    ):
        colour = jnp.mean(jnp.asarray(images), axis=(1, 2))
        logits = self.head(colour)
        if return_intermediates:
            return logits, {"colour": colour}
        return logits


def build_stand_in(config: model.ModelConfig, key: jax.Array) -> StandIn:
    net = StandIn(num_classes=config.num_classes, key=key)
    net.init()
    return net


def cross_entropy(
    logits: jt.Float[jt.Array, "batch classes"], labels: jt.Int[jt.Array, " batch"]
) -> jt.Float[jt.Array, ""]:
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


def use_stand_in(monkeypatch) -> None:
    """Make ``model`` build the stand-in and use cross-entropy, for one test."""
    monkeypatch.setattr(model, "build_model", build_stand_in)
    monkeypatch.setattr(model, "loss_fn", cross_entropy)


@dataclasses.dataclass
class SyntheticImagenette:
    """A loader like ``imagenette.load``, with random photos at the stored size.

    Each class gets its own mean colour, so even the stand-in can learn something.
    """

    n_train: int = 64
    n_test: int = 48
    seed: int = 0

    def __call__(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        n = {"train": self.n_train, "test": self.n_test}[split]
        rng = np.random.default_rng([self.seed, split == "test"])
        labels = rng.integers(0, len(data.CLASS_NAMES), n).astype(np.int32)
        size = imagenette.STORED_SIZE
        colours = np.linspace(30, 220, len(data.CLASS_NAMES))[labels]
        noise = rng.normal(0, 20, (n, size, size, 3))
        images = np.clip(colours[:, None, None, None] + noise, 0, 255).astype(np.uint8)
        return images, labels
