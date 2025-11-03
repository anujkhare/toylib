"""Tests for layer.py."""

import pytest
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from beartype import typing

from toylib.nn.layers import Linear, Embedding


class TestLinear:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(0)

    @pytest.mark.parametrize(
        "in_features,out_features",
        [
            (5, 3),
            (10, 10),
            (1, 1),
        ],
    )
    def test_initialization_shapes(self, key, in_features, out_features):
        """Test various layer sizes."""
        layer = Linear(in_features, out_features, key=key)
        assert layer.weights.shape == (in_features, out_features)
        assert layer.bias.shape == (out_features,)

    @pytest.mark.parametrize(
        "batch_shape,in_features,out_features",
        [
            ((), 10, 5),  # Single vector input
            ((5,), 10, 5),  # 2D input
            ((3, 4), 10, 5),  # 3D input
            ((2, 3, 4), 10, 5),  # 4D input
        ],
    )
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_batched_forward_pass(
        self,
        key: jt.PRNGKeyArray,
        batch_shape: typing.Iterable[int],
        in_features: int,
        out_features: int,
        use_bias: bool,
    ):
        """Test forward pass with various batch dimensions."""
        layer = Linear(in_features, out_features, key=key, use_bias=use_bias)
        x = jax.random.normal(key, batch_shape + (in_features,))

        output = layer(x)

        expected_shape = batch_shape + (out_features,)
        assert output.shape == expected_shape


class TestEmbedding:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    @pytest.mark.parametrize(
        "vocab_size,embedding_dim",
        [
            (10, 5),
            (100, 64),
            (1000, 128),
            (50000, 300),
        ],
    )
    def test_initialization_shapes(self, key, vocab_size, embedding_dim):
        """Test various embedding sizes."""
        layer = Embedding(vocab_size, embedding_dim, key=key)
        assert layer.weights.shape == (vocab_size, embedding_dim)

    @pytest.mark.parametrize(
        "sequence_shape,vocab_size,embedding_dim",
        [
            ((4,), 10, 6),  # 1D sequence
            ((8, 4), 10, 7),  # 2D batch
            ((2, 8, 4), 10, 8),  # 3D batch
        ],
    )
    def test_forward_pass_shapes(self, key, sequence_shape, vocab_size, embedding_dim):
        """Test embedding lookup with various input shapes."""
        layer = Embedding(vocab_size, embedding_dim, key=key)

        # Generate random valid token indices
        tokens = jax.random.randint(key, sequence_shape, minval=0, maxval=vocab_size)

        output = layer(tokens)

        expected_shape = sequence_shape + (embedding_dim,)
        assert output.shape == expected_shape

    @pytest.mark.parametrize("token_idx", [0, 5, 9])
    def test_specific_token_lookup(self, key, token_idx):
        """Test that specific tokens return correct embeddings."""
        layer = Embedding(vocab_size=10, embedding_dim=5, key=key)
        output = layer(jnp.array([token_idx]))

        np.testing.assert_allclose(output[0], layer.weights[token_idx], rtol=1e-6)
