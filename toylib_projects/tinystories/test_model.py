"""Tests for layer.py."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from toylib_projects.tinystories.model import DecoderOnlyTransformer, ModelConfig


class TestLinear:
    @pytest.mark.parametrize(
        "input_tokens,model_config",
        [
            (
                np.array([[1, 2, 3], [4, 5, 6]]),
                ModelConfig(
                    num_layers=2, num_heads=2, qkv_dim=16, vocab_size=10, seq_len=5
                ),
            ),
        ],
    )
    def test_forward(self, input_tokens: np.ndarray, model_config: ModelConfig):
        """Test a simple forward pass of the model."""
        model = DecoderOnlyTransformer(config=model_config, key=jax.random.PRNGKey(42))
        actual = model(input_tokens)
        print(actual)
