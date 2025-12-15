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
    use_bias: bool = False
    key: jt.PRNGKeyArray

    # Trainable parameters
    weights: jt.Float[jt.Array, "in_features out_features"] | None = None
    bias: typing.Optional[jt.Float[jt.Array, " out_features"]] | None = None

    def init(self) -> None:
        w_key = self.key
        in_features = self.in_features
        out_features = self.out_features

        # https://arxiv.org/pdf/2310.17813
        std = min(1.0, math.sqrt(out_features / in_features)) / math.sqrt(in_features)
        self.weights = jax.random.normal(w_key, (in_features, out_features)) * std
        self.bias = jax.numpy.zeros((out_features,)) if self.use_bias else None

    def __call__(
        self, x: jt.Float[jt.Array, "... in_features"]
    ) -> jt.Float[jt.Array, "... out_features"]:
        x = jax.numpy.dot(x, self.weights)
        if self.use_bias:
            x = x + self.bias
        return x


class Embedding(module.Module):
    """Defines an embedding layer that stores an embedding matrix for discrete tokens."""

    vocab_size: int
    embedding_dim: int
    key: jt.PRNGKeyArray

    # Trainable parameters
    weights: jt.Float[jt.Array, "vocab_size embedding_dim"] | None = None

    def init(
        self,
    ) -> None:
        # Initialize the embedding weights with a std normal distribution
        self.weights = jax.random.normal(
            self.key, (self.vocab_size, self.embedding_dim)
        )

    def __call__(
        self, tokens: jt.Int[jt.Array, "... seq_len"]
    ) -> jt.Float[jt.Array, "... seq_len embedding_dim"]:
        return jax.numpy.take(self.weights, tokens, axis=0)


def rms_norm(
    x: jt.Float[jt.Array, "... dim"],
) -> jt.Float[jt.Array, "... dim"]:
    """Applies RMS Normalization over the last dimension of the input tensor.

    Args:
        x: Input tensor

    Returns:
        The RMS normalized tensor of the same shape as input x.
    """
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-9)
    return x / rms
