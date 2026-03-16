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
