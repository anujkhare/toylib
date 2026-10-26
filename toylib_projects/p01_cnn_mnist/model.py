"""The image classifier for post 1. A skeleton: the layers and the loss are yours.

What is here already is the wiring the harness needs, so the pipeline runs end
to end before the model exists:

* ``ModelConfig`` and ``build_model(config, key)``: the ``model_factory``.
* ``CNN``: takes ``(B, H, W, C)`` images, returns ``(B, num_classes)`` logits.
  For now it is a single linear layer on the flattened pixels (a placeholder).
* ``loss_fn(logits, labels)``: a placeholder that is differentiable but
  meaningless; replace it with softmax cross-entropy.
* ``forward(model, batch) -> (loss, aux)``: the ``forward_fn``. ``aux`` carries
  the logits so metrics such as accuracy can read them.

The same code serves MNIST (28x28x1) and CIFAR-10 (32x32x3): the config's
``image_shape`` is the only difference.
"""

import dataclasses

import jax
import jax.numpy as jnp
import jaxtyping as jt

from toylib.nn import layers
from toylib.nn import module


@dataclasses.dataclass
class ModelConfig:
    image_shape: tuple[int, int, int] = (28, 28, 1)  # (H, W, C)
    num_classes: int = 10
    # TODO(you): add what the CNN needs, e.g. channels per conv layer, kernel
    # size, and whether the head flattens or global-average-pools.


class CNN(module.Module):
    image_shape: tuple[int, int, int]
    num_classes: int
    key: jt.PRNGKeyArray

    # Trainable sub-modules are created in init().
    # TODO(you): conv layers, pooling, and the head.
    head: layers.Linear | None = None

    def init(self) -> None:
        # PLACEHOLDER: one linear layer on the flattened pixels, so that
        # training, evaluation and checkpointing can be exercised now.
        h, w, c = self.image_shape
        self.head = layers.Linear(
            in_features=h * w * c,
            out_features=self.num_classes,
            use_bias=True,
            key=self.key,
        )

    def __call__(
        self,
        images: jt.Float[jt.Array, "batch height width channels"],
        return_intermediates: bool = False,
    ):
        """Logits for a batch of images.

        With ``return_intermediates=True``, returns ``(logits, acts)`` where
        ``acts`` maps layer names (``"conv1"``, ``"pool1"``, ...) to activations,
        for the post's figures. Under ``jit``, mark the flag static.
        """
        acts = {}
        # Accept NumPy batches straight from ArrayDataset (no-op under jit).
        x = jnp.asarray(images)
        # TODO(you): conv -> relu -> pool blocks, recording each in `acts`.
        x = x.reshape(x.shape[0], -1)
        acts["flatten"] = x
        logits = self.head(x)
        if return_intermediates:
            return logits, acts
        return logits


def build_model(config: ModelConfig, key: jax.Array) -> CNN:
    """The harness's ``model_factory``: construct and initialize the model."""
    model = CNN(image_shape=config.image_shape, num_classes=config.num_classes, key=key)
    model.init()
    return model


def loss_fn(
    logits: jt.Float[jt.Array, "batch classes"], labels: jt.Int[jt.Array, " batch"]
) -> jt.Float[jt.Array, ""]:
    """Mean loss over the batch.

    TODO(you): softmax cross-entropy. The placeholder below ignores the labels
    and pulls every logit towards zero; it only exists so gradients flow.
    """
    del labels
    return jnp.mean(logits**2)


def forward(model: CNN, batch: dict) -> tuple[jax.Array, dict]:
    """The harness's ``forward_fn``: ``(model, batch) -> (loss, aux)``."""
    logits = model(batch["image"])
    loss = loss_fn(logits, batch["label"])
    return loss, {"logits": logits}
