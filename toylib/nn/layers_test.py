"""Tests for layer.py."""

import pytest
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from beartype import typing

from toylib.nn.layers import Conv2D, Embedding, GroupNorm, Linear, upsample_nearest


class TestLinear:
    @pytest.fixture
    def key(self):
        return jax.random.key(0)

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
        layer = Linear(
            in_features=in_features, out_features=out_features, key=key, use_bias=True
        )
        layer.init()
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
        layer = Linear(
            in_features=in_features,
            out_features=out_features,
            key=key,
            use_bias=use_bias,
        )
        layer.init()
        x = jax.random.normal(key, batch_shape + (in_features,))

        output = layer(x)

        expected_shape = batch_shape + (out_features,)
        assert output.shape == expected_shape


class TestEmbedding:
    @pytest.fixture
    def key(self):
        return jax.random.key(42)

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
        layer = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim, key=key)
        layer.init()
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
        layer = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim, key=key)
        layer.init()

        # Generate random valid token indices
        tokens = jax.random.randint(key, sequence_shape, minval=0, maxval=vocab_size)

        output = layer(tokens)

        expected_shape = sequence_shape + (embedding_dim,)
        assert output.shape == expected_shape

    @pytest.mark.parametrize("token_idx", [0, 5, 9])
    def test_specific_token_lookup(self, key, token_idx):
        """Test that specific tokens return correct embeddings."""
        layer = Embedding(vocab_size=10, embedding_dim=5, key=key)
        layer.init()
        output = layer(jnp.array([token_idx]))

        np.testing.assert_allclose(output[0], layer.weights[token_idx], rtol=1e-6)


class TestConv2D:
    @pytest.fixture
    def key(self):
        return jax.random.key(0)

    @pytest.mark.parametrize(
        "in_c,out_c,k,stride,in_hw,out_hw",
        [
            (3, 64, 3, 1, 64, 64),    # SAME, stride 1 preserves spatial
            (64, 128, 3, 2, 64, 32),  # SAME, stride 2 halves spatial
            (4, 8, 1, 1, 8, 8),       # 1x1 conv preserves spatial
        ],
    )
    def test_shapes(self, key, in_c, out_c, k, stride, in_hw, out_hw):
        conv = Conv2D(
            in_channels=in_c, out_channels=out_c, kernel_size=k,
            stride=stride, padding="SAME", key=key, use_bias=True,
        )
        conv.init()
        x = jnp.zeros((2, in_hw, in_hw, in_c))
        y = conv(x)
        assert y.shape == (2, out_hw, out_hw, out_c)
        assert conv.weights.shape == (k, k, in_c, out_c)
        assert conv.bias.shape == (out_c,)

    def test_no_bias(self, key):
        conv = Conv2D(
            in_channels=4, out_channels=8, kernel_size=3,
            padding="SAME", key=key, use_bias=False,
        )
        conv.init()
        assert conv.bias is None
        y = conv(jnp.zeros((1, 8, 8, 4)))
        assert y.shape == (1, 8, 8, 8)

    def test_explicit_integer_padding(self, key):
        """padding=1 must produce the same output shape as padding='SAME' for k=3, s=1."""
        conv_same = Conv2D(
            in_channels=4, out_channels=4, kernel_size=3,
            padding="SAME", key=key, use_bias=False,
        )
        conv_int = Conv2D(
            in_channels=4, out_channels=4, kernel_size=3,
            padding=1, key=key, use_bias=False,
        )
        conv_same.init()
        conv_int.init()
        x = jnp.ones((1, 8, 8, 4))
        assert conv_same(x).shape == conv_int(x).shape == (1, 8, 8, 4)

    def test_gradient_flow(self, key):
        conv = Conv2D(
            in_channels=4, out_channels=8, kernel_size=3,
            padding="SAME", key=key, use_bias=True,
        )
        conv.init()

        def loss_fn(c, x):
            return jnp.sum(c(x))

        x = jax.random.normal(key, (1, 8, 8, 4))
        grads = jax.grad(loss_fn)(conv, x)
        assert jnp.any(grads.weights != 0)
        assert jnp.any(grads.bias != 0)


class TestGroupNorm:
    @pytest.fixture
    def key(self):
        return jax.random.key(0)

    def test_init_shapes(self):
        gn = GroupNorm(num_features=64, num_groups=32)
        gn.init()
        assert gn.scale.shape == (64,) and gn.bias.shape == (64,)
        # Default scale = 1, bias = 0 → norm is identity-ish before training.
        np.testing.assert_array_equal(np.asarray(gn.scale), np.ones(64))
        np.testing.assert_array_equal(np.asarray(gn.bias), np.zeros(64))

    def test_normalizes_per_group(self):
        """After GroupNorm, each (sample, group) chunk has mean≈0 and var≈1."""
        gn = GroupNorm(num_features=8, num_groups=4)  # 2 channels per group
        gn.init()
        # Construct input where each group has wildly different stats.
        x = jax.random.normal(jax.random.key(1), (2, 4, 4, 8)) * 5.0 + 7.0
        y = gn(x)
        # Reshape to (B, H, W, G, C/G) and verify per-(B,G) stats.
        y_grouped = y.reshape(2, 4, 4, 4, 2)
        per_group_mean = jnp.mean(y_grouped, axis=(1, 2, 4))  # (B, G)
        per_group_var = jnp.var(y_grouped, axis=(1, 2, 4))    # (B, G)
        np.testing.assert_allclose(per_group_mean, jnp.zeros((2, 4)), atol=1e-5)
        np.testing.assert_allclose(per_group_var, jnp.ones((2, 4)), atol=1e-3)

    def test_invalid_groups_raises(self):
        gn = GroupNorm(num_features=10, num_groups=3)  # 10 % 3 != 0
        with pytest.raises(ValueError, match="divisible"):
            gn.init()


class TestUpsampleNearest:
    def test_default_2x(self):
        x = jnp.arange(2 * 4 * 4 * 3).reshape(2, 4, 4, 3).astype(jnp.float32)
        y = upsample_nearest(x, factor=2)
        assert y.shape == (2, 8, 8, 3)
        # Top-left 2x2 block of the output should be a copy of x[:, 0, 0, :].
        np.testing.assert_array_equal(y[:, 0, 0, :], x[:, 0, 0, :])
        np.testing.assert_array_equal(y[:, 0, 1, :], x[:, 0, 0, :])
        np.testing.assert_array_equal(y[:, 1, 0, :], x[:, 0, 0, :])

    def test_factor_3(self):
        x = jnp.zeros((1, 8, 8, 4))
        assert upsample_nearest(x, factor=3).shape == (1, 24, 24, 4)
