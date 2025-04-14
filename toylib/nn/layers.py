import math
import jax
import jaxtyping
import typing

from toylib.nn import module


@jax.tree_util.register_pytree_node_class
class Linear(module.Module):
    """Defines a simple feedforward layer: which is a linear transformation. """

    # Trainable parameters
    weights: jaxtyping.Array
    bias: typing.Optional[jaxtyping.Array]

    # Hyperparameters / metadata
    in_features: int
    out_features: int
    use_bias: bool

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True, *, key: jaxtyping.PRNGKeyArray) -> None:
        # Split the random key for weights and bias
        w_key, b_key = jax.random.split(key, 2)
        
        # We initialize the weights with a uniform distribution
        lim = 1 / math.sqrt(in_features)
        self.weights = jax.random.uniform(w_key, (in_features, out_features), minval=-lim, maxval=lim)
        if use_bias:
            self.bias = jax.random.uniform(b_key, (out_features,), minval=-lim, maxval=lim)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.key = key

    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        x = jax.numpy.dot(x, self.weights)
        if self.use_bias:
            x = x + self.bias
        return x
