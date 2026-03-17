import jax
import jaxtyping
import typing

from toylib.nn import layers
from toylib.nn import module


class MLP(module.Module):
    output_layer: layers.Linear
    hidden_layers: typing.List[module.Module]

    in_features: int
    hidden_dims: list[int]
    out_features: int
    key: jaxtyping.PRNGKeyArray

    def init(self) -> None:
        # Split the random key for weights and bias
        keys = jax.random.split(self.key, len(self.hidden_dims) + 1)

        # Create the layers
        hidden_layers = []
        input_dim = self.in_features
        for i, hidden_dim in enumerate(self.hidden_dims):
            layer = layers.Linear(input_dim, hidden_dim, key=keys[i])
            hidden_layers.append(layer)
            input_dim = hidden_dim

        # Create the output layer
        output_layer = layers.Linear(input_dim, self.out_features, key=keys[-1])

        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        for layer in self.hidden_layers:
            x = layer(x)
            x = jax.nn.relu(x)
        return self.output_layer(x)
