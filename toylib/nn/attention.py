import einops
from jax import numpy as jnp
import jax
import jaxtyping as jt
import typing

from toylib.nn import layers
from toylib.nn import module


def scaled_dot_product_attention(
    q: jt.Float[jt.Array, "... seq_len qkv_dim"],
    k: jt.Float[jt.Array, "... seq_len qkv_dim"],
    v: jt.Float[jt.Array, "... seq_len qkv_dim"],
    mask: typing.Optional[jt.Float[jt.Array, "... seq_len qkv_dim"]],
) -> tuple[
    jt.Float[jt.Array, "... seq_len qkv_dim"], jt.Float[jt.Array, "... seq_len seq_len"]
]:
    """Compute scaled dot product attention.

    Given query (`q`), key (`k`), and value (`v`) tensors, this function first computes the
    attention weights as the softmax of the dot product of `q` and `k`, scaled by the square
    root of the dimension of the keys. If a mask is provided, it is applied to the attention
    logits before the softmax is computed.

    Finally, the attention weights are used to compute the weighted average of the given values.

    NOTE: the batch dimension is not explicitly handled in this function.

    Args:
        q: query tensor
        k: keys tensor
        v: values tensor
        mask: optional boolean mask to apply to the attention logits

    Returns:
        tuple of final values and attention weights

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
class MultiHeadAttention(module.Module):
    """
    The MultiHeadAttention defines `num_heads` attention heads. For the given input `Q`, `K`, `V`
    tensors, `num_head` linear projections of dim `qkv_dim / num_heads` are produced.

    An attention weight is then computed using the scaled dot product attention method. The
    weighted average of the values are then concatenated from the various heads to produce a
    single output value vector. A final linear layer is applied on top of this with non-linearity.
    """

    def __init__(
        self, qkv_dim: int, num_heads: int, *, key: jax.random.PRNGKey
    ) -> None:
        keys = jax.random.split(key, 4)

        # Input projections - different "heads" will be split out from the same tensor
        self.q_projection = layers.Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[0],
        )
        self.k_projection = layers.Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[1],
        )
        self.v_projection = layers.Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[2],
        )

        # Output linear layer
        self.linear = layers.Linear(
            in_features=qkv_dim, out_features=qkv_dim, key=keys[3]
        )

        self.qkv_dim = qkv_dim
        self.num_heads = num_heads

    def __call__(
        self,
        Q: jt.Float[jt.Array, "... seq_len qkv_dim"],
        K: jt.Float[jt.Array, "... seq_len qkv_dim"],
        V: jt.Float[jt.Array, "... seq_len qkv_dim"],
        mask: typing.Optional[jt.Float[jt.Array, "... seq_len qkv_dim"]] = None,
    ):
        Q = self.q_projection(Q)
        K = self.k_projection(K)
        V = self.v_projection(V)
        print("Q", Q.shape)

        # Reshape the input tensors to split out the heads
        Q = einops.rearrange(
            Q,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        K = einops.rearrange(
            K,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        V = einops.rearrange(
            V,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        print("Q reshaped", Q.shape)

        # Apply self atttention to each head, get the output values
        # values: [... num_heads, seq_len, qkv_dim/num_heads], attention_weights: [... num_heads, seq_len, qkv_dim/num_heads]
        values, attention_weights = scaled_dot_product_attention(
            q=Q, k=K, v=V, mask=mask
        )

        values = einops.rearrange(
            values,
            "... num_heads seq_len d -> ... seq_len (num_heads d)",
        )

        # Apply linear: [..., seq_len, qkv_dim]
        values = self.linear(values)

        # return the attention weights and the output values
        return values, attention_weights
