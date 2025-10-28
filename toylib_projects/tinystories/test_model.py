"""Tests for layer.py."""

import pytest
import jax
import jax.numpy as jnp

from toylib_projects.tinystories.model import DecoderOnlyTransformer, ModelConfig


class TestDecoderOnlyTransformer:
    @pytest.mark.parametrize(
        "input_tokens,model_config",
        [
            (
                jnp.array([[1, 2, 3], [4, 5, 6]]),
                ModelConfig(
                    num_layers=2, num_heads=2, qkv_dim=16, vocab_size=10, seq_len=5
                ),
            ),
        ],
    )
    def test_smoke(self, input_tokens: jnp.ndarray, model_config: ModelConfig):
        """Test a simple forward pass of the model."""
        model = DecoderOnlyTransformer(config=model_config, key=jax.random.PRNGKey(42))
        actual = model(input_tokens)
        assert actual.shape == (
            input_tokens.shape[0],
            input_tokens.shape[1],
            model_config.vocab_size,
        )

        # Assert that the output is a valid log-probability distribution
        probs = jnp.exp(actual)
        probs_sum = probs.sum(axis=-1)
        assert jnp.allclose(probs_sum, jnp.ones_like(probs_sum), atol=1e-5)
