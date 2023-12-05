import math
import jax

from typing import Optional
from jaxtyping import Array, PRNGKeyArray

from . import module


class Linear(module.Module):
    """Defines a simple feedforward layer: which is a linear transformation."""

    weights: Array
    bias: Optional[Array]

    in_features: int
    out_features: int
    use_bias: bool

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True, *, key: PRNGKeyArray) -> None:
        w_key, b_key = jax.random.split(key, 2)
        lim = 1 / math.sqrt(in_features)
        self.weights = jax.random.uniform(w_key, (out_features, in_features), minval=-lim, maxval=lim)
        if use_bias:
            self.bias = jax.random.uniform(b_key, (out_features,), minval=-lim, maxval=lim)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(self, x: Array) -> Array:
        x = jax.numpy.dot(self.weights, x)
        if self.use_bias:
            x = x + self.bias
        return x