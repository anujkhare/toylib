from jax import numpy as jnp
import chex
import dataclasses
import jax

from toylib.nn import layers
from toylib.nn import module


def scaled_dot_product_attention(q, k, v, mask=None):
    """Compute scaled dot product attention.

    Given query (`q`), key (`k`), and value (`v`) tensors, this function first computes the
    attention weights as the softmax of the dot product of `q` and `k`, scaled by the square
    root of the dimension of the keys. If a mask is provided, it is applied to the attention
    logits before the softmax is computed.

    Finally, the attention weights are used to compute the weighted average of the given values.

    NOTE: the batch dimension is not explicitly handled in this function.

    Args:
        q: [..., seq_len_q, qk_dim]
        k: [..., seq_len_kv, qk_dim]
        v: [..., seq_len_kv, v_dim]
        mask: optional boolean mask of shape [..., seq_len_q, seq_len_kv] to apply to the attention logits

    Returns:
        tuple of final values [seq_len_q, v_dim] and attention weights [seq_len_q, seq_len_kv]

    """
    d_k = q.shape[-1]
    assert q.shape[-1] == k.shape[-1], "q and k must have the same feature dimension"

    attention_logits = jnp.matmul(q, k.swapaxes(-1, -2)) / jnp.sqrt(d_k)
    if mask is not None:
        attention_logits = jnp.where(mask, attention_logits, -jnp.inf)
    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    values = jnp.matmul(attention_weights, v)
    return values, attention_weights


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class MultiHeadAttention(module.Module):
    """
    The MultiHeadAttention defines `num_heads` attention heads. For the given input `Q`, `K`, `V`
    tensors, `num_head` linear projections of dim `qkv_dim / num_heads` are produced.

    An attention weight is then computed using the scaled dot product attention method. The
    weighted average of the values are then concatenated from the various heads to produce a
    single output value vector. A final linear layer is applied on top of this with non-linearity.
    """

    key: jax.random.PRNGKey

    num_heads: int = 4
    qkv_dim: int = 32

    def __post_init__(self) -> None:
        keys = jax.random.split(self.key, 4)

        # Input projections - different "heads" will be split out from the same tensor
        self.q_projection = layers.Linear(
            in_features=self.qkv_dim, out_features=self.qkv_dim, use_bias=False, key=keys[0]
        )
        self.k_projection = layers.Linear(
            in_features=self.qkv_dim, out_features=self.qkv_dim, use_bias=False, key=keys[1]
        )
        self.v_projection = layers.Linear(
            in_features=self.qkv_dim, out_features=self.qkv_dim, use_bias=False, key=keys[2]
        )

        # Output linear layer
        self.linear = layers.Linear(
            in_features=self.qkv_dim, out_features=self.qkv_dim, key=keys[3]
        )

    def __call__(self, Q, K, V, mask):
        chex.assert_equal_rank((Q, K, V))
        assert Q.shape[-1] == self.qkv_dim, (
            f"Final dimension in input tensors must be `qkv`: {self.qkv_dim}, found: {Q.shape}"
        )
        assert K.shape[-1] == self.qkv_dim, (
            f"Final dimension in input tensors must be `qkv`: {self.qkv_dim}, found: {K.shape}"
        )
        assert V.shape[-1] == self.qkv_dim, (
            f"Final dimension in input tensors must be `qkv`: {self.qkv_dim}, found: {V.shape}"
        )

        Q = self.q_projection(Q)  # [seq_len, qkv_dim]
        K = self.k_projection(K)  # [seq_len, qkv_dim]
        V = self.v_projection(V)  # [seq_len, qkv_dim]
        print("Q", Q.shape)

        # Reshape the input tensors to split out the heads
        Q = jnp.reshape(Q, (Q.shape[0], self.num_heads, -1)).transpose(
            1, 0, 2
        )  # [num_heads, seq_len, qkv_dim / num_heads]
        K = jnp.reshape(K, (K.shape[0], self.num_heads, -1)).transpose(
            1, 0, 2
        )  # [num_heads, seq_len, qkv_dim / num_heads]
        V = jnp.reshape(V, (V.shape[0], self.num_heads, -1)).transpose(
            1, 0, 2
        )  # [num_heads, seq_len, qkv_dim / num_heads]
        print("Q reshaped", Q.shape)

        # Apply self atttention to each head, get the output values
        # values: [num_heads, seq_len, qkv_dim], attention_weights: [num_heads, seq_len, seq_len]
        values, attention_weights = scaled_dot_product_attention(
            q=Q, k=K, v=V, mask=mask
        )

        # Reshape to [seq_len, num_heads, ...] and then collapse the last two dimensions
        values = jax.lax.collapse(values.transpose(1, 0, 2), -2)

        # Apply linear: [seq_len, qkv_dim]
        values = self.linear(values)

        # return the attention weights and the output values
        return values, attention_weights
