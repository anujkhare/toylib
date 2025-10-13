import math
import jax
import jaxtyping as jt
import typing

from toylib.nn import module


@jax.tree_util.register_pytree_node_class
class Linear(module.Module):
    """Defines a simple feedforward layer: which is a linear transformation."""

    # Trainable parameters
    weights: jt.Float[jt.Array, "in_features out_features"]
    bias: typing.Optional[jt.Float[jt.Array, "out_features"]]

    # Hyperparameters / metadata
    in_features: int
    out_features: int
    use_bias: bool

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        key: jt.PRNGKeyArray,
    ) -> None:
        # Split the random key for weights and bias
        w_key, b_key = jax.random.split(key, 2)

        # We initialize the weights with a uniform distribution with Xavier initialization
        # lim = 1 / math.sqrt(in_features)
        lim = math.sqrt(6 / (in_features + out_features))

        self.weights = jax.random.uniform(
            w_key, (in_features, out_features), minval=-lim, maxval=lim
        )
        self.bias = (
            jax.random.uniform(b_key, (out_features,), minval=-lim, maxval=lim)
            if use_bias
            else None
        )

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

    # Hyperparameters
    vocab_size: int
    embedding_dim: int

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        key: jt.PRNGKeyArray,
    ) -> None:
        # Initialize the embedding weights with a uniform distribution
        lim = 1 / math.sqrt(embedding_dim)
        self.weights = jax.random.uniform(
            key, (vocab_size, embedding_dim), minval=-lim, maxval=lim
        )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.key = key

    def __call__(
        self, tokens: jt.Int[jt.Array, "... num_tokens"]
    ) -> jt.Float[jt.Array, "... num_tokens embedding_dim"]:
        return jax.numpy.take(self.weights, tokens, axis=0)
