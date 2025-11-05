import math
import jax
import jax.numpy as jnp
import jaxtyping as jt
import typing

from toylib.nn import module


@jax.tree_util.register_pytree_node_class
class Linear(module.Module):
    """Defines a simple feedforward layer: which is a linear transformation."""

    # Trainable parameters
    weights: jt.Float[jt.Array, "in_features out_features"]
    bias: typing.Optional[jt.Float[jt.Array, " out_features"]]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        key: jt.PRNGKeyArray,
    ) -> None:
        w_key = key

        # https://arxiv.org/pdf/2310.17813
        std = min(1.0, math.sqrt(out_features / in_features)) / math.sqrt(in_features)
        self.weights = jax.random.normal(w_key, (in_features, out_features)) * std
        self.bias = jax.numpy.zeros((out_features,)) if use_bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.key = key

    def __call__(
        self, x: jt.Float[jt.Array, "... in_features"]
    ) -> jt.Float[jt.Array, "... out_features"]:
        x = jax.numpy.dot(x, self.weights)
        if self.use_bias:
            x = x + self.bias
        return x


@jax.tree_util.register_pytree_node_class
class Embedding(module.Module):
    """Defines an embedding layer that stores an embedding matrix for discrete tokens."""

    # Trainable parameters
    weights: jt.Float[jt.Array, "vocab_size embedding_dim"]

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        key: jt.PRNGKeyArray,
    ) -> None:
        # Initialize the embedding weights with a std normal distribution
        self.weights = jax.random.normal(key, (vocab_size, embedding_dim))

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.key = key

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
