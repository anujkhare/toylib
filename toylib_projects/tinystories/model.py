import dataclasses
import jax
import jaxtyping as jt
import jax.numpy as jnp
from toylib import nn


@dataclasses.dataclass
class ModelConfig:
    """Configuration for the DecoderOnlyTransformer model."""

    num_layers: int = 2
    num_heads: int = 5
    qkv_dim: int = 256

    vocab_size: int = 1000
    seq_len: int = 512


# @title Define a model
@jax.tree_util.register_pytree_node_class
class DecoderOnlyTransformer(nn.module.Module):
    """A simple decoder-only transformer model.

    Takes in a sequence of tokens, embeds them using a learned embedding layer,
    applies causal self-attention with ROPE embeddings, and outputs the logits
    for the next token prediction.
    """

    key: jt.PRNGKeyArray
    config: ModelConfig

    def __init__(self, config: ModelConfig, *, key: jt.PRNGKeyArray) -> None:
        # Generate keys - embedding, attention layers, output projection
        keys = list(jax.random.split(key, config.num_layers + 2))

        # Embedding layer
        self.embedding_layer = nn.layers.Embedding(
            vocab_size=config.vocab_size, embedding_dim=config.qkv_dim, key=keys.pop()
        )

        # Self-attention layers
        self.layers = []
        for _ in range(config.num_layers):
            self.layers.append(
                nn.attention.MultiHeadAttention(
                    num_heads=config.num_heads, qkv_dim=config.qkv_dim, key=keys.pop()
                )
            )

        # Output projection
        self.output_layer = nn.layers.Linear(
            in_features=config.qkv_dim, out_features=config.vocab_size, key=keys.pop()
        )

        self.config = config

    def _make_causal_mask(self) -> jt.Float[jt.Array, "seq_len seq_len"]:
        seq_len = self.config.seq_len
        return jnp.triu(jnp.ones((seq_len, seq_len)))

    def __call__(
        self, x: jt.Float[jt.Array, "batch_size seq_len"]
    ) -> jt.Float[jt.Array, "batch_size seq_len vocab_size"]:
        mask = None
        # Input projection to project the embeddings to the model dimension
        x = self.embedding_layer(x)

        # Apply each attention layer
        # TODO: check where the non-linearity should go
        for layer in self.layers:
            x, _ = layer(Q=x, K=x, V=x, mask=mask)
            x = jax.nn.relu(x)

        x = self.output_layer(x)
        return jax.nn.log_softmax(x, axis=-1)
