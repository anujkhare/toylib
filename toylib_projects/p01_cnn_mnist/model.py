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
import einops

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


class Block(module.Module):
    in_channels: int
    out_channels: int
    pool: bool = True
    key: jt.PRNGKeyArray

    def init(self) -> None:
        self.conv_layer = layers.Conv2D(
            in_channels=self.in_channels, out_channels=self.out_channels, key=self.key
        )

    def __call__(
        self, x: jt.Float[jt.Array, "B H W in_channels"]
    ) -> jt.Float[jt.Array, "B H_out W_out out_channels"]:
        x = self.conv_layer(x)
        x = jax.nn.relu(x)
        if self.pool:
            x = layers.max_pool(x, factor=2)
        return x


class CNN(module.Module):
    image_shape: tuple[int, int, int]
    num_classes: int
    key: jt.PRNGKeyArray

    def init(self) -> None:
        h, w, c = self.image_shape
        keys = jax.random.split(self.key, 4)

        self.conv1 = Block(in_channels=c, out_channels=32, key=keys[0])
        self.conv2 = Block(in_channels=32, out_channels=64, key=keys[1])
        self.conv3 = Block(in_channels=64, out_channels=64, pool=False, key=keys[2])

        self.head = layers.Linear(
            in_features=(h // 4) * (w // 4) * 64,
            out_features=self.num_classes,
            use_bias=True,
            key=keys[3],
        )

    def __call__(
        self,
        images: jt.Float[jt.Array, "B H W C"],
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        logits = self.head(einops.rearrange(x, "b h w c -> b (h w c)"))
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
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=-1))


def forward(model: CNN, batch: dict) -> tuple[jax.Array, dict]:
    """The harness's ``forward_fn``: ``(model, batch) -> (loss, aux)``."""
    logits = model(batch["image"])
    loss = loss_fn(logits, batch["label"])
    return loss, {"logits": logits}
