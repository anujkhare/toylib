"""Tests for model.py."""

import pytest
import jax
import jax.numpy as jnp

from toylib_projects.tinystories import decoder_only_model


class TestDecoderOnlyTransformer:
    @pytest.mark.parametrize(
        "input_tokens,model_config",
        [
            (
                jnp.array([[1, 2, 3], [4, 5, 6]]),
                decoder_only_model.ModelConfig(
                    num_layers=2, num_heads=2, qkv_dim=16, vocab_size=10, seq_len=3
                ),
            ),
        ],
    )
    def test_smoke(
        self, input_tokens: jnp.ndarray, model_config: decoder_only_model.ModelConfig
    ):
        """Test a simple forward pass of the model."""
        model = decoder_only_model.DecoderOnlyTransformer(
            config=model_config, key=jax.random.PRNGKey(42)
        )
        actual = model(input_tokens)
        assert actual.shape == (
            input_tokens.shape[0],
            input_tokens.shape[1],
            model_config.vocab_size,
        )


class TestTrainStep:
    @pytest.mark.parametrize(
        "input_tokens,target_tokens,model_config",
        [
            (
                jnp.array([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]]),
                jnp.array([[2, 3, 4, 0, 0], [5, 6, 7, 0, 0]]),
                decoder_only_model.ModelConfig(
                    num_layers=2, num_heads=2, qkv_dim=16, vocab_size=10, seq_len=5
                ),
            ),
        ],
    )
    def test_smoke(
        self,
        input_tokens: jnp.ndarray,
        target_tokens: jnp.ndarray,
        model_config: decoder_only_model.ModelConfig,
    ):
        """Test a simple training step."""
        model = decoder_only_model.DecoderOnlyTransformer(
            config=model_config, key=jax.random.PRNGKey(42)
        )
        loss = decoder_only_model.train_step(
            model, input_tokens, target_tokens, jnp.ones_like(target_tokens)
        )

        assert loss >= 0.0
