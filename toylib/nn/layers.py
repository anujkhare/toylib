import math
import jax
import jax.numpy as jnp
import jaxtyping as jt
import typing

from toylib.nn import module


class Linear(module.Module):
    """Defines a simple feedforward layer: which is a linear transformation."""

    # Hyperparameters
    in_features: int
    out_features: int
    key: jt.PRNGKeyArray
    use_bias: bool = False
    init_std: typing.Optional[float] = None

    # Trainable parameters
    weights: typing.Optional[jt.Float[jt.Array, "in_features out_features"]] = None
    bias: typing.Optional[jt.Float[jt.Array, " out_features"]] = None

    def init(self) -> None:
        w_key = self.key
        in_features = self.in_features
        out_features = self.out_features

        if self.init_std is not None:
            std = self.init_std
            # Initialize weights with a uniform distribution in the range [-std * sqrt(3), std * sqrt(3)]
            # For uniform distribution betweeen [-a, a], the variance is a^2 / 3.
            # a^2 / 3 = std^2 => a = std * sqrt(3)
            s = std * math.sqrt(3)
            self.weights = jax.random.uniform(
                key=w_key, shape=(in_features, out_features), minval=-s, maxval=s
            ).astype(self.param_dtype)
        else:
            # https://arxiv.org/pdf/2310.17813
            std = min(1.0, math.sqrt(out_features / in_features)) / math.sqrt(
                in_features
            )
            self.weights = (
                jax.random.normal(key=w_key, shape=(in_features, out_features)) * std
            ).astype(self.param_dtype)
        self.bias = (
            jax.numpy.zeros((out_features,), dtype=self.param_dtype)
            if self.use_bias
            else None
        )

    def __call__(
        self, x: jt.Float[jt.Array, "... in_features"]
    ) -> jt.Float[jt.Array, "... out_features"]:
        x = jax.numpy.dot(x.astype(self.dtype), self.weights.astype(self.dtype))
        if self.use_bias:
            x = x + self.bias.astype(self.dtype)
        return x


class Embedding(module.Module):
    """Defines an embedding layer that stores an embedding matrix for discrete tokens."""

    vocab_size: int
    embedding_dim: int
    key: jt.PRNGKeyArray

    # Trainable parameters
    weights: typing.Optional[jt.Float[jt.Array, "vocab_size embedding_dim"]] = None

    def init(
        self,
    ) -> None:
        # Initialize the embedding weights with a std normal distribution
        self.weights = jax.random.normal(
            self.key, (self.vocab_size, self.embedding_dim)
        ).astype(self.param_dtype)

    def __call__(
        self, tokens: jt.Integer[jt.Array, "... seq_len"]
    ) -> jt.Float[jt.Array, "... seq_len embedding_dim"]:
        return jax.numpy.take(self.weights, tokens, axis=0).astype(self.dtype)


class Conv2D(module.Module):
    """2D convolution with NHWC layout, optional bias, and 'SAME' or integer padding.

    Uses ``jax.lax.conv_general_dilated`` under the hood. Kernels are stored as
    ``(kernel_size, kernel_size, in_channels, out_channels)`` (HWIO layout) to
    match the NHWC input convention.

    Weight init follows the same pattern as Linear: if ``init_std`` is set,
    weights are drawn uniformly in ``[-init_std*sqrt(3), +init_std*sqrt(3)]``;
    otherwise a fan-in-aware default ``min(1, sqrt(out/in)) / sqrt(in)`` is
    used (matching arXiv:2310.17813), where ``in/out`` here are the effective
    fan-in / fan-out of the conv (``kernel_size^2 * channels``).
    """

    in_channels: int
    out_channels: int
    key: jt.PRNGKeyArray
    kernel_size: int = 3
    stride: int = 1
    # Either an int (symmetric pad on both spatial dims) or one of "SAME"/"VALID".
    padding: typing.Union[int, str] = "SAME"
    use_bias: bool = True
    init_std: typing.Optional[float] = None

    # Trainable
    weights: typing.Optional[
        jt.Float[jt.Array, "kh kw in_channels out_channels"]
    ] = None
    bias: typing.Optional[jt.Float[jt.Array, " out_channels"]] = None

    def init(self) -> None:
        k = self.kernel_size
        fan_in = k * k * self.in_channels
        fan_out = k * k * self.out_channels

        if self.init_std is not None:
            std = self.init_std
            s = std * math.sqrt(3)
            self.weights = jax.random.uniform(
                key=self.key,
                shape=(k, k, self.in_channels, self.out_channels),
                minval=-s,
                maxval=s,
            ).astype(self.param_dtype)
        else:
            std = min(1.0, math.sqrt(fan_out / fan_in)) / math.sqrt(fan_in)
            self.weights = (
                jax.random.normal(
                    key=self.key,
                    shape=(k, k, self.in_channels, self.out_channels),
                )
                * std
            ).astype(self.param_dtype)

        self.bias = (
            jnp.zeros((self.out_channels,), dtype=self.param_dtype)
            if self.use_bias
            else None
        )

    def _resolved_padding(self) -> typing.Union[str, list[tuple[int, int]]]:
        if isinstance(self.padding, str):
            return self.padding
        p = int(self.padding)
        return [(p, p), (p, p)]

    def __call__(
        self, x: jt.Float[jt.Array, "B H W in_channels"]
    ) -> jt.Float[jt.Array, "B H_out W_out out_channels"]:
        x = jax.lax.conv_general_dilated(
            x.astype(self.dtype),
            self.weights.astype(self.dtype),
            window_strides=(self.stride, self.stride),
            padding=self._resolved_padding(),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.use_bias:
            x = x + self.bias.astype(self.dtype)
        return x


class GroupNorm(module.Module):
    """Group Normalization over the channel dimension of an NHWC tensor.

    Splits the ``num_features`` channels into ``num_groups`` equal-sized groups
    and normalizes each group's activations (over the (H, W, C/G) volume per
    sample). Learnable per-channel scale and bias are applied after norm.

    Statistics are computed in float32 for numerical stability and cast back
    to ``self.dtype`` on the way out. Matches the convention used by
    ``rms_norm`` below.
    """

    num_features: int
    num_groups: int = 32
    eps: float = 1e-5

    # Trainable per-channel affine.
    scale: typing.Optional[jt.Float[jt.Array, " num_features"]] = None
    bias: typing.Optional[jt.Float[jt.Array, " num_features"]] = None

    def init(self) -> None:
        if self.num_features % self.num_groups != 0:
            raise ValueError(
                f"num_features ({self.num_features}) must be divisible by "
                f"num_groups ({self.num_groups})"
            )
        self.scale = jnp.ones((self.num_features,), dtype=self.param_dtype)
        self.bias = jnp.zeros((self.num_features,), dtype=self.param_dtype)

    def __call__(
        self, x: jt.Float[jt.Array, "B H W num_features"]
    ) -> jt.Float[jt.Array, "B H W num_features"]:
        orig_dtype = x.dtype
        B, H, W, C = x.shape
        G = self.num_groups
        # Reshape last axis into (G, C//G) to normalize per-group over (H, W, C/G).
        x32 = x.astype(jnp.float32).reshape(B, H, W, G, C // G)
        mean = jnp.mean(x32, axis=(1, 2, 4), keepdims=True)
        var = jnp.var(x32, axis=(1, 2, 4), keepdims=True)
        x32 = (x32 - mean) * jax.lax.rsqrt(var + self.eps)
        x32 = x32.reshape(B, H, W, C)
        # Per-channel affine.
        scale = self.scale.astype(jnp.float32).reshape(1, 1, 1, C)
        bias = self.bias.astype(jnp.float32).reshape(1, 1, 1, C)
        return (x32 * scale + bias).astype(orig_dtype)


def upsample_nearest(
    x: jt.Float[jt.Array, "B H W C"], factor: int = 2
) -> jt.Float[jt.Array, "B H_out W_out C"]:
    """Nearest-neighbor upsample of an NHWC image by ``factor`` along H and W.

    Pure function — no trainable parameters. Used in the VAE decoder to expand
    the spatial grid before a regular convolution. Avoids the checkerboard
    artifacts that transposed convolutions exhibit.
    """
    B, H, W, C = x.shape
    return jax.image.resize(
        x, (B, H * factor, W * factor, C), method="nearest"
    )


def rms_norm(
    x: jt.Float[jt.Array, "... dim"],
) -> jt.Float[jt.Array, "... dim"]:
    """Applies RMS Normalization over the last dimension of the input tensor.

    The mean-square computation is done in float32 for numerical stability,
    regardless of the input dtype. The output is cast back to the input dtype.

    Args:
        x: Input tensor

    Returns:
        The RMS normalized tensor of the same shape as input x.
    """
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-9)
    return (x / rms).astype(orig_dtype)
