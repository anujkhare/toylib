import einops
import jax
import jax.numpy as jnp
import jaxtyping as jt
import math
import typing

from toylib.nn import layers
from toylib.nn import module


class RotaryPositionalEmbedding(module.Module):
    """Implements Rotary Positional Embeddings (RoPE) as described in https://arxiv.org/abs/2104.09864."""

    seq_len: int = 1024
    qkv_dim: int = 128
    base: int = 100_000

    def init(self) -> None:
        # Construct the frequencies
        positions = jnp.arange(0, self.seq_len)
        freqs = self.base ** (jnp.arange(0, self.qkv_dim, 2) / self.qkv_dim)
        # [seq_len, qkv_dim // 2]
        self.gamma = einops.einsum(positions, 1.0 / freqs, "t, d -> t d")
        self.cos = jnp.cos(self.gamma).astype(jnp.bfloat16)
        self.sin = jnp.sin(self.gamma).astype(jnp.bfloat16)

    def __call__(
        self, x: jt.Float[jt.Array, "... seq_len qkv_dim"], t0: int = 0
    ) -> jt.Float[jt.Array, "... seq_len qkv_dim"]:
        t, d = x.shape[-2:]
        # The position embeddings are cached for `seq_len` which may be larger
        # than the maximum sequence length of the model. We index the cached
        # embeddings equal to the current sequence length `t`. During training,
        # this length is (likely) equal to the model's `seq_len`. During
        # inference, this would vary based on the input sequence. `t0` is
        # used when the starting token is not at time 0 - for instance,
        # when using a KV cache.
        if t0 + t > self.seq_len:
            raise ValueError(
                "Position index out of range of RoPE cache:"
                f"t0 ({t0}) + t ({t}) > seq_len ({self.seq_len})"
            )
        sin, cos = self.sin[t0 : t0 + t, :], self.cos[t0 : t0 + t, :]

        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        # element-wise multiplication: rotate dims clockwise pair-wise
        es_shape = "... t d, t d -> ... t d"
        y1 = einops.einsum(x1, cos, es_shape) + einops.einsum(x2, sin, es_shape)
        y2 = -einops.einsum(x1, sin, es_shape) + einops.einsum(x2, cos, es_shape)
        return jnp.concatenate([y1, y2], axis=-1)


def scaled_dot_product_attention(
    q: jt.Float[jt.Array, "... seq_len qkv_dim"],
    k: jt.Float[jt.Array, "... seq_len qkv_dim"],
    v: jt.Float[jt.Array, "... seq_len qkv_dim"],
    mask: typing.Optional[jt.Float[jt.Array, "... seq_len seq_len"]],
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
        # Use a large negative value to mask out attention logits instead of -jnp.inf
        attention_logits = jnp.where(mask, attention_logits, -1e9)

    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    values = jnp.matmul(attention_weights, v)
    return values, attention_weights


class MultiHeadAttention(module.Module):
    """
    The MultiHeadAttention defines `num_heads` attention heads. For the given input `Q`, `K`, `V`
    tensors, `num_head` linear projections of dim `qkv_dim / num_heads` are produced.

    An attention weight is then computed using the scaled dot product attention method. The
    weighted average of the values are then concatenated from the various heads to produce a
    single output value vector. A final linear layer is applied on top of this with non-linearity.
    """

    qkv_dim: int
    num_heads: int
    use_qk_norm: bool = True
    key: jt.PRNGKeyArray

    def init(self) -> None:
        qkv_dim = self.qkv_dim
        keys = jax.random.split(self.key, 4)

        # init_std for the q, k, v projections is set to 1/sqrt(qkv_dim) to ensure that the initial variance
        # of the projected queries, keys, and values is around 1. This helps stabilize training at the start
        # by preventing excessively large or small values in the attention computations.
        init_std = 1 / math.sqrt(qkv_dim)
        # Input projections - different "heads" will be split out from the same tensor
        self.q_projection = layers.Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[0],
            init_std=init_std,
        )
        self.k_projection = layers.Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[1],
            init_std=init_std,
        )
        self.v_projection = layers.Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[2],
            init_std=init_std,
        )

        # Output linear layer
        self.linear = layers.Linear(
            in_features=qkv_dim,
            out_features=qkv_dim,
            use_bias=False,
            key=keys[3],
            # Initialize weights to zero to stabilize training at the start
            init_std=0.0,
        )

    def __call__(
        self,
        Q: jt.Float[jt.Array, "... seq_len qkv_dim"],
        K: jt.Float[jt.Array, "... seq_len qkv_dim"],
        V: jt.Float[jt.Array, "... seq_len qkv_dim"],
        mask: typing.Optional[jt.Float[jt.Array, "... seq_len seq_len"]] = None,
        *,
        rope: typing.Optional[RotaryPositionalEmbedding] = None,
        return_attention_weights: bool = False,
    ) -> typing.Union[
        tuple[
            jt.Float[jt.Array, "... seq_len qkv_dim"],
            jt.Float[jt.Array, "... seq_len seq_len"],
        ],
        jt.Float[jt.Array, "... seq_len qkv_dim"],
    ]:
        Q = self.q_projection(Q)
        K = self.k_projection(K)
        V = self.v_projection(V)

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
        if mask is not None:
            mask = einops.rearrange(
                mask, "... seq_len1 seq_len2 -> ... 1 seq_len1 seq_len2"
            )

        if rope is not None:
            Q = rope(Q)
            K = rope(K)

        if self.use_qk_norm:
            Q = layers.rms_norm(Q)
            K = layers.rms_norm(K)

        # Apply self atttention to each head, get the output values
        # values: [... num_heads, seq_len, qkv_dim/num_heads], attention_weights: [... num_heads, seq_len, seq_len]
        values, attention_weights = scaled_dot_product_attention(
            q=Q, k=K, v=V, mask=mask
        )

        values = einops.rearrange(
            values,
            "... num_heads seq_len d -> ... seq_len (num_heads d)",
        )

        # Apply linear: [..., seq_len, qkv_dim]
        values = self.linear(values)

        # return the output values and attention weights if specified
        if return_attention_weights:
            return values, attention_weights
        return values
