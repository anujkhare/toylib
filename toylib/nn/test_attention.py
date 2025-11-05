"""Tests for model.py."""

import jax.numpy as jnp

from toylib.nn import attention


class TestROPE:
    def test_smoke(self):
        qkv_dim, seq_len = 8, 16
        rope = attention.RotaryPositionalEmbedding(qkv_dim=qkv_dim, seq_len=seq_len)
        # [batch_size, num_heads, seq_len, qkv_dim]
        x = jnp.ones((3, 4, seq_len, qkv_dim))
        actual = rope(x)
        assert actual.shape == (3, 4, seq_len, qkv_dim)
