"""Tests for model.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from toylib.nn import attention


class TestROPE:
    def test_smoke(self):
        qkv_dim, seq_len = 8, 16
        rope = attention.RotaryPositionalEmbedding(qkv_dim=qkv_dim, seq_len=seq_len)
        rope.init()
        # [batch_size, num_heads, seq_len, qkv_dim]
        x = jnp.ones((3, 4, seq_len, qkv_dim))
        actual = rope(x)
        assert actual.shape == (3, 4, seq_len, qkv_dim)


class TestRotaryPositionalEmbeddingIsFixed:
    def _rope(self, qkv_dim=8, seq_len=16):
        rope = attention.RotaryPositionalEmbedding(qkv_dim=qkv_dim, seq_len=seq_len)
        rope.init()
        return rope

    def test_has_no_trainable_parameters(self):
        assert jax.tree.leaves(self._rope()) == []

    def test_position_zero_is_unrotated(self):
        x = jax.random.normal(jax.random.key(0), (1, 8))
        np.testing.assert_allclose(self._rope()(x), x, atol=1e-6)

    def test_offset_matches_a_longer_sequence(self):
        """`t0` picks up where the sequence left off, as a KV cache would need."""
        rope = self._rope()
        x = jax.random.normal(jax.random.key(0), (6, 8))
        np.testing.assert_allclose(rope(x)[4:], rope(x[4:], t0=4), atol=1e-6)

    def test_dot_products_depend_only_on_relative_position(self):
        """The property RoPE exists for: q(i) . k(j) depends on i - j, not on i."""
        rope = self._rope()
        q, k = jax.random.normal(jax.random.key(0), (2, 1, 8))

        def dot_at(i):
            return jnp.vdot(rope(q, t0=i), rope(k, t0=i + 3))

        np.testing.assert_allclose(dot_at(0), dot_at(9), atol=1e-5)

    def test_rejects_positions_past_seq_len(self):
        with pytest.raises(ValueError, match="out of range"):
            self._rope(seq_len=4)(jnp.zeros((5, 8)))
