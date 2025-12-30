"""Tests for model.py."""

import pytest
import jax
import jax.numpy as jnp
import jaxtyping as jt
from unittest.mock import Mock, patch

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
        "batch,model_config",
        [
            (
                {
                    "inputs": jnp.array([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]]),
                    "targets": jnp.array([[2, 3, 4, 0, 0], [5, 6, 7, 0, 0]]),
                    "mask": jnp.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
                },
                decoder_only_model.ModelConfig(
                    num_layers=2, num_heads=2, qkv_dim=16, vocab_size=10, seq_len=5
                ),
            ),
        ],
    )
    def test_smoke(
        self,
        batch: jt.PyTree,
        model_config: decoder_only_model.ModelConfig,
    ):
        """Test a simple training step."""
        model = decoder_only_model.DecoderOnlyTransformer(
            config=model_config, key=jax.random.PRNGKey(42)
        )
        (loss, _), _ = jax.jit(
            jax.value_and_grad(decoder_only_model.train_step, has_aux=True)
        )(model, batch)

        assert loss >= 0.0


class TestSampling:
    @pytest.mark.parametrize(
        "model_config,context,max_length",
        [
            (
                decoder_only_model.ModelConfig(
                    num_layers=2, num_heads=2, qkv_dim=16, vocab_size=10, seq_len=16
                ),
                [1, 2, 3],
                5,
            ),
        ],
    )
    def test_smoke(
        self,
        model_config: decoder_only_model.ModelConfig,
        context: jnp.ndarray,
        max_length: int,
    ):
        """Test that sampling runs with a random small model."""
        model = decoder_only_model.DecoderOnlyTransformer(
            config=model_config, key=jax.random.PRNGKey(42)
        )
        sampled = list(
            decoder_only_model.sample(
                model=model,
                input_tokens=context,
                key=jax.random.PRNGKey(0),
                max_output_tokens=max_length,
                temperature=1.0,
                top_k=5,
            )
        )

        assert len(sampled) == max_length

    def test_top_k(self):
        """Test that top-k sampling only samples from the top k logits."""
        vocab_size = 10
        logits = jnp.array([1.0, 5.0, 2.0, 8.0, 3.0, 0.5, 0.1, 0.2, 0.3, 0.4])
        top_k = 3

        mock_model = Mock()
        mock_model.return_value = logits.reshape(1, vocab_size)

        with patch(
            "toylib_projects.tinystories.decoder_only_model.jax.random.categorical"
        ) as mock_categorical:
            mock_categorical.return_value = jnp.array(3)

            list(
                decoder_only_model.sample(
                    model=mock_model,
                    input_tokens=[1],
                    key=jax.random.PRNGKey(0),
                    max_output_tokens=1,
                    temperature=1.0,
                    top_k=top_k,
                )
            )

            # Check that categorical was called with masked logits
            call_args = mock_categorical.call_args
            passed_logits = call_args.kwargs["logits"]

            # Top 3 values are at indices 3 (8.0), 1 (5.0), 4 (3.0)
            # All other logits should be -inf
            assert jnp.isfinite(passed_logits[3])
            assert jnp.isfinite(passed_logits[1])
            assert jnp.isfinite(passed_logits[4])
            assert jnp.all(jnp.isinf(passed_logits[jnp.array([0, 2, 5, 6, 7, 8, 9])]))

    def test_temperature_scaling(self):
        """Test that temperature correctly scales the logits."""
        vocab_size = 5
        logits = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        temperature = 2.0

        mock_model = Mock()
        mock_model.return_value = logits.reshape(1, vocab_size)

        with patch(
            "toylib_projects.tinystories.decoder_only_model.jax.random.categorical"
        ) as mock_categorical:
            mock_categorical.return_value = jnp.array(4)

            list(
                decoder_only_model.sample(
                    model=mock_model,
                    input_tokens=[1],
                    key=jax.random.PRNGKey(0),
                    max_output_tokens=1,
                    temperature=temperature,
                    top_k=None,
                )
            )

            # Check that logits were scaled by temperature
            call_args = mock_categorical.call_args
            passed_logits = call_args.kwargs["logits"]

            expected_logits = logits / temperature
            assert jnp.allclose(passed_logits, expected_logits)

    def test_temperature_one_no_scaling(self):
        """Test that temperature=1.0 does not modify logits."""
        vocab_size = 5
        logits = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mock_model = Mock()
        mock_model.return_value = logits.reshape(1, vocab_size)

        with patch(
            "toylib_projects.tinystories.decoder_only_model.jax.random.categorical"
        ) as mock_categorical:
            mock_categorical.return_value = jnp.array(4)

            list(
                decoder_only_model.sample(
                    model=mock_model,
                    input_tokens=[1],
                    key=jax.random.PRNGKey(0),
                    max_output_tokens=1,
                    temperature=1.0,
                    top_k=None,
                )
            )

            call_args = mock_categorical.call_args
            passed_logits = call_args.kwargs["logits"]

            assert jnp.allclose(passed_logits, logits)

    def test_top_k_one_greedy(self):
        """Test that top_k=1 results in greedy sampling (only max logit is valid)."""
        vocab_size = 5
        logits = jnp.array([1.0, 2.0, 5.0, 3.0, 4.0])  # max at index 2

        mock_model = Mock()
        mock_model.return_value = logits.reshape(1, vocab_size)

        with patch(
            "toylib_projects.tinystories.decoder_only_model.jax.random.categorical"
        ) as mock_categorical:
            mock_categorical.return_value = jnp.array(2)

            list(
                decoder_only_model.sample(
                    model=mock_model,
                    input_tokens=[1],
                    key=jax.random.PRNGKey(0),
                    max_output_tokens=1,
                    temperature=1.0,
                    top_k=1,
                )
            )

            call_args = mock_categorical.call_args
            passed_logits = call_args.kwargs["logits"]

            # Only index 2 should be finite
            assert jnp.isfinite(passed_logits[2])
            assert jnp.all(jnp.isinf(passed_logits[jnp.array([0, 1, 3, 4])]))
