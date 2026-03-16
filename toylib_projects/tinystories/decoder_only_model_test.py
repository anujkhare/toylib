"""Tests for model.py."""

import pytest
import jax
import jax.numpy as jnp
import jaxtyping as jt

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
            config=model_config, key=jax.random.key(42)
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
            config=model_config, key=jax.random.key(42)
        )
        (loss, _), _ = jax.jit(
            jax.value_and_grad(decoder_only_model.train_step, has_aux=True)
        )(model, batch)

        assert loss >= 0.0


def _make_model_and_padded_prompt(model_config, context):
    """Helper: create a model and a padded prompt array."""
    model = decoder_only_model.DecoderOnlyTransformer(
        config=model_config, key=jax.random.key(42)
    )
    padded = jnp.zeros(model_config.seq_len, dtype=jnp.uint16)
    padded = padded.at[: len(context)].set(jnp.array(context, dtype=jnp.uint16))
    return model, padded


class TestSampling:
    MODEL_CONFIG = decoder_only_model.ModelConfig(
        num_layers=2, num_heads=2, qkv_dim=16, vocab_size=10, seq_len=16
    )

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
        context: list[int],
        max_length: int,
    ):
        """Test that sampling runs with a random small model."""
        model, padded = _make_model_and_padded_prompt(model_config, context)
        sampled = decoder_only_model.sample(
            model=model,
            input_tokens=padded,
            prompt_len=len(context),
            key=jax.random.key(0),
            max_output_tokens=max_length,
            temperature=1.0,
            top_k=5,
        )
        assert sampled.shape == (max_length,)

    def test_top_k(self):
        """Test that top-k sampling only samples tokens within the top-k set."""
        context = [1]
        top_k = 3
        model, padded = _make_model_and_padded_prompt(self.MODEL_CONFIG, context)

        # Get the top-k token indices according to the model's logits
        logits = model(padded)  # [seq_len, vocab_size]
        _, top_k_indices = jax.lax.top_k(logits[len(context) - 1], top_k)
        top_k_set = set(top_k_indices.tolist())

        # Verify every sample lands in the top-k set across multiple keys
        for seed in range(20):
            generated = decoder_only_model.sample(
                model=model,
                input_tokens=padded,
                prompt_len=len(context),
                key=jax.random.key(seed),
                max_output_tokens=1,
                temperature=1.0,
                top_k=top_k,
            )
            assert int(generated[0]) in top_k_set

    def test_temperature_scaling(self):
        """Test that very low temperature converges sampling to greedy (argmax)."""
        context = [1, 2]
        model, padded = _make_model_and_padded_prompt(self.MODEL_CONFIG, context)

        greedy = decoder_only_model.sample(
            model=model,
            input_tokens=padded,
            prompt_len=len(context),
            key=jax.random.key(0),
            max_output_tokens=1,
            temperature=1.0,
            top_k=1,
        )
        low_temp = decoder_only_model.sample(
            model=model,
            input_tokens=padded,
            prompt_len=len(context),
            key=jax.random.key(0),
            max_output_tokens=1,
            temperature=1e-6,
            top_k=None,
        )
        # Near-zero temperature should reproduce greedy selection
        assert int(low_temp[0]) == int(greedy[0])

    def test_temperature_one_no_scaling(self):
        """Test that temperature=1.0 gives the same result as top_k=1 greedy baseline."""
        context = [1]
        model, padded = _make_model_and_padded_prompt(self.MODEL_CONFIG, context)

        # With the same key, two runs at temp=1.0 should be identical (determinism)
        result_a = decoder_only_model.sample(
            model=model,
            input_tokens=padded,
            prompt_len=len(context),
            key=jax.random.key(7),
            max_output_tokens=3,
            temperature=1.0,
            top_k=None,
        )
        result_b = decoder_only_model.sample(
            model=model,
            input_tokens=padded,
            prompt_len=len(context),
            key=jax.random.key(7),
            max_output_tokens=3,
            temperature=1.0,
            top_k=None,
        )
        assert jnp.array_equal(result_a, result_b)

    def test_top_k_one_greedy(self):
        """Test that top_k=1 always returns the argmax token."""
        context = [1, 2, 3]
        model, padded = _make_model_and_padded_prompt(self.MODEL_CONFIG, context)

        logits = model(padded)  # [seq_len, vocab_size]
        expected_token = int(jnp.argmax(logits[len(context) - 1]))

        # top_k=1 is deterministic regardless of key
        for seed in range(5):
            generated = decoder_only_model.sample(
                model=model,
                input_tokens=padded,
                prompt_len=len(context),
                key=jax.random.key(seed),
                max_output_tokens=1,
                temperature=1.0,
                top_k=1,
            )
            assert int(generated[0]) == expected_token
