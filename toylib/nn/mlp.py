import jax
import jaxtyping
import typing

from toylib.nn import layers
from toylib.nn import module


@jax.tree_util.register_pytree_node_class
class MLP(module.Module):
    output_layer: layers.Linear
    hidden_layers: typing.List[module.Module]

    in_features: int
    hidden_dims: list[int]
    out_features: int

    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int],
        out_features: int,
        *,
        key: jaxtyping.PRNGKeyArray,
    ) -> None:
        # Split the random key for weights and bias
        keys = jax.random.split(key, len(hidden_dims) + 1)

        # Create the layers
        hidden_layers = []
        input_dim = in_features
        for i, hidden_dim in enumerate(hidden_dims):
            layer = hidden_layers.Linear(input_dim, hidden_dim, key=keys[i])
            hidden_layers.append(layer)
            input_dim = hidden_dim

        # Create the output layer
        output_layer = layers.Linear(input_dim, out_features, key=keys[-1])

        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        for layer in self.hidden_layers:
            x = layer(x)
            x = jax.nn.relu(x)
        return self.output_layer(x)
