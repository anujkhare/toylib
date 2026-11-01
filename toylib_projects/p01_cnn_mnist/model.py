"""The image classifiers for post 1, from a linear model up to a small CNN.

* ``ModelConfig``, ``build_model`` and ``CNN``: convolutions and pooling, then a
  flatten and one linear layer. The stages come from the config, so the same class
  serves 28x28 MNIST, 32x32 CIFAR-10 and 160x160 Imagenette.
* ``MLPConfig``, ``build_mlp`` and ``MLPClassifier``: flatten the pixels, then fully
  connected layers. With no hidden layers it is a linear model.
* ``loss_fn(logits, labels)``: softmax cross-entropy.
* ``forward(model, batch) -> (loss, aux)``: the harness's ``forward_fn`` for every
  model. ``aux`` carries the logits so metrics such as accuracy can read them.

``resnet.py`` holds the larger model.
"""

import dataclasses
import einops

import jax
import jax.numpy as jnp
import jaxtyping as jt

from toylib.nn import layers
from toylib.nn import mlp
from toylib.nn import module


@dataclasses.dataclass
class ModelConfig:
    image_shape: tuple[int, int, int] = (28, 28, 1)  # (H, W, C)
    num_classes: int = 10
    channels: tuple[int, ...] = (32, 64, 64)  # One convolution per entry.
    pools: tuple[bool, ...] = (True, True, False)  # Halve the image after each?


@dataclasses.dataclass
class MLPConfig:
    image_shape: tuple[int, int, int] = (160, 160, 3)  # (H, W, C)
    num_classes: int = 10
    hidden_dims: tuple[int, ...] = ()  # Empty for a linear model.


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
    channels: tuple[int, ...] = (32, 64, 64)
    pools: tuple[bool, ...] = (True, True, False)
    key: jt.PRNGKeyArray

    def init(self) -> None:
        h, w, c = self.image_shape
        keys = jax.random.split(self.key, len(self.channels) + 1)

        in_channels = c
        for i, (out_channels, pool) in enumerate(zip(self.channels, self.pools)):
            block = Block(
                in_channels=in_channels,
                out_channels=out_channels,
                pool=pool,
                key=keys[i],
            )
            setattr(self, f"conv{i + 1}", block)
            in_channels = out_channels

        shrink = 2 ** sum(self.pools)
        self.head = layers.Linear(
            in_features=(h // shrink) * (w // shrink) * in_channels,
            out_features=self.num_classes,
            use_bias=True,
            key=keys[-1],
        )

    def __call__(
        self,
        images: jt.Float[jt.Array, "B H W C"],
        return_intermediates: bool = False,
    ):
        """Logits for a batch of images.

        With ``return_intermediates=True``, returns ``(logits, acts)`` where
        ``acts`` maps ``"conv1"``, ``"conv2"``, ... to each block's output, for the
        post's figures. Under ``jit``, mark the flag static.
        """
        acts = {}
        # Accept NumPy batches straight from ArrayDataset (no-op under jit).
        x = jnp.asarray(images)
        for i in range(len(self.channels)):
            name = f"conv{i + 1}"
            x = getattr(self, name)(x)
            acts[name] = x

        logits = self.head(einops.rearrange(x, "b h w c -> b (h w c)"))
        if return_intermediates:
            return logits, acts
        return logits


class MLPClassifier(module.Module):
    """Flatten the pixels, then fully connected layers."""

    image_shape: tuple[int, int, int]
    num_classes: int
    hidden_dims: tuple[int, ...] = ()
    key: jt.PRNGKeyArray

    def init(self) -> None:
        h, w, c = self.image_shape
        self.mlp = mlp.MLP(
            in_features=h * w * c,
            hidden_dims=list(self.hidden_dims),
            out_features=self.num_classes,
            key=self.key,
        )

    def __call__(
        self, images: jt.Float[jt.Array, "B H W C"]
    ) -> jt.Float[jt.Array, "B classes"]:
        return self.mlp(einops.rearrange(jnp.asarray(images), "b h w c -> b (h w c)"))


def build_model(config: ModelConfig, key: jax.Array) -> CNN:
    """The harness's ``model_factory``: construct and initialize the model."""
    model = CNN(
        image_shape=config.image_shape,
        num_classes=config.num_classes,
        channels=config.channels,
        pools=config.pools,
        key=key,
    )
    model.init()
    return model


def build_mlp(config: MLPConfig, key: jax.Array) -> MLPClassifier:
    """The ``model_factory`` for the linear model and the MLP."""
    model = MLPClassifier(
        image_shape=config.image_shape,
        num_classes=config.num_classes,
        hidden_dims=config.hidden_dims,
        key=key,
    )
    model.init()
    return model


def loss_fn(
    logits: jt.Float[jt.Array, "batch classes"], labels: jt.Int[jt.Array, " batch"]
) -> jt.Float[jt.Array, ""]:
    """Softmax cross-entropy, averaged over the batch."""
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=-1))


def forward(model: module.Module, batch: dict) -> tuple[jax.Array, dict]:
    """The harness's ``forward_fn``: ``(model, batch) -> (loss, aux)``."""
    logits = model(batch["image"])
    loss = loss_fn(logits, batch["label"])
    return loss, {"logits": logits}
