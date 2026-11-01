"""A small ResNet for CIFAR-10: how far past the three-layer CNN can we go?

The layout follows the CIFAR ResNets of He et al. (2015): a 3x3 stem, three stages
that double the channels and halve the image, global average pooling, and a
linear head. With ``blocks_per_stage=n`` the network has ``6n + 2`` weighted
layers, so ``n=3`` is ResNet-20.

Two differences from the paper:

* GroupNorm replaces BatchNorm. It needs no running statistics, so the model stays
  a pure function and works with the harness as is.
* ``residual=False`` drops the skip connections, leaving a plain network of the
  same depth, to measure what they contribute.

Larger images need a wider view, so the config can also add stages and give the
stem a larger, strided kernel. The Imagenette model has ``num_stages=4``,
``blocks_per_stage=2`` and a 7x7 stem with stride 2, the layout of ResNet-18.

It reuses ``model.forward`` and ``model.loss_fn``: only the architecture differs.
"""

import dataclasses

import jax
import jax.numpy as jnp
import jaxtyping as jt

from toylib.nn import layers
from toylib.nn import module


@dataclasses.dataclass
class ResNetConfig:
    image_shape: tuple[int, int, int] = (32, 32, 3)  # (H, W, C)
    num_classes: int = 10
    width: int = 16  # Channels in the first stage; must divide by ``num_groups``.
    blocks_per_stage: int = 3
    residual: bool = True
    num_groups: int = 8
    num_stages: int = 3
    stem_kernel_size: int = 3
    stem_stride: int = 1


class ResidualBlock(module.Module):
    """``x -> conv, norm, relu, conv, norm``, plus ``x`` itself, then relu."""

    in_channels: int
    out_channels: int
    stride: int = 1
    residual: bool = True
    num_groups: int = 8
    key: jt.PRNGKeyArray

    def init(self) -> None:
        keys = jax.random.split(self.key, 3)
        self.conv1 = layers.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=self.stride,
            use_bias=False,
            key=keys[0],
        )
        self.norm1 = layers.GroupNorm(
            num_features=self.out_channels, num_groups=self.num_groups
        )
        self.conv2 = layers.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            use_bias=False,
            key=keys[1],
        )
        self.norm2 = layers.GroupNorm(
            num_features=self.out_channels, num_groups=self.num_groups
        )
        # The skip needs a 1x1 conv only when the shape changes.
        self.project = None
        if self.residual and (
            self.stride != 1 or self.in_channels != self.out_channels
        ):
            self.project = layers.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=self.stride,
                use_bias=False,
                key=keys[2],
            )

    def __call__(
        self, x: jt.Float[jt.Array, "B H W in_channels"]
    ) -> jt.Float[jt.Array, "B H_out W_out out_channels"]:
        y = jax.nn.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.residual:
            y = y + (x if self.project is None else self.project(x))
        return jax.nn.relu(y)


class ResNet(module.Module):
    image_shape: tuple[int, int, int]
    num_classes: int
    width: int = 16
    blocks_per_stage: int = 3
    residual: bool = True
    num_groups: int = 8
    num_stages: int = 3
    stem_kernel_size: int = 3
    stem_stride: int = 1
    key: jt.PRNGKeyArray

    def init(self) -> None:
        c = self.image_shape[-1]
        stages = [self.width * 2**i for i in range(self.num_stages)]
        num_blocks = len(stages) * self.blocks_per_stage
        keys = jax.random.split(self.key, num_blocks + 2)

        self.stem = layers.Conv2D(
            in_channels=c,
            out_channels=self.width,
            kernel_size=self.stem_kernel_size,
            stride=self.stem_stride,
            use_bias=False,
            key=keys[0],
        )
        self.stem_norm = layers.GroupNorm(
            num_features=self.width, num_groups=self.num_groups
        )

        self.blocks = []
        in_channels = self.width
        for stage, out_channels in enumerate(stages):
            for i in range(self.blocks_per_stage):
                self.blocks.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2 if stage > 0 and i == 0 else 1,
                        residual=self.residual,
                        num_groups=self.num_groups,
                        key=keys[len(self.blocks) + 1],
                    )
                )
                in_channels = out_channels

        self.head = layers.Linear(
            in_features=in_channels,
            out_features=self.num_classes,
            use_bias=True,
            key=keys[-1],
        )

    def __call__(
        self,
        images: jt.Float[jt.Array, "B H W C"],
        return_intermediates: bool = False,
    ):
        """Logits for a batch of images; see ``model.CNN`` for the flag."""
        x = jnp.asarray(images)
        x = jax.nn.relu(self.stem_norm(self.stem(x)))
        acts = {"stem": x}
        for i, block in enumerate(self.blocks):
            x = block(x)
            acts[f"block{i}"] = x
        logits = self.head(jnp.mean(x, axis=(1, 2)))
        if return_intermediates:
            return logits, acts
        return logits


def build_resnet(config: ResNetConfig, key: jax.Array) -> ResNet:
    """The harness's ``model_factory``."""
    net = ResNet(
        image_shape=config.image_shape,
        num_classes=config.num_classes,
        width=config.width,
        blocks_per_stage=config.blocks_per_stage,
        residual=config.residual,
        num_groups=config.num_groups,
        num_stages=config.num_stages,
        stem_kernel_size=config.stem_kernel_size,
        stem_stride=config.stem_stride,
        key=key,
    )
    net.init()
    return net
