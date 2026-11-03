"""Tests for model.py."""

import numpy as np
import optax
import pytest
import jax
import jax.numpy as jnp
import jaxtyping as jt

from toylib_projects.p02_gpt_web import decoder_only_model


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
        model.init()
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
        model.init()
        (loss, _), _ = jax.jit(
            jax.value_and_grad(decoder_only_model.train_step, has_aux=True)
        )(model, batch)

        assert loss >= 0.0


def _make_model_and_padded_prompt(model_config, context):
    """Helper: create a model and a padded prompt array."""
    model = decoder_only_model.DecoderOnlyTransformer(
        config=model_config, key=jax.random.key(42)
    )
    model.init()
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


SMALL_CONFIG = decoder_only_model.ModelConfig(
    num_layers=2, num_heads=2, qkv_dim=16, vocab_size=10, seq_len=8
)


def _make_model(config=SMALL_CONFIG, seed=42):
    model = decoder_only_model.DecoderOnlyTransformer(
        config=config, key=jax.random.key(seed)
    )
    model.init()
    return model


def _trained_a_little(config=SMALL_CONFIG):
    """Blocks start as the identity, so give them weights that do something."""
    model = _make_model(config)
    tokens = jax.random.randint(jax.random.key(0), (4, 8), 0, 10)
    batch = {"inputs": tokens, "targets": tokens, "mask": jnp.ones_like(tokens)}
    optimizer = optax.adam(1e-2)
    state = optimizer.init(model)

    @jax.jit
    def step(model, state):
        grads = jax.grad(lambda m: decoder_only_model.train_step(m, batch)[0])(model)
        updates, state = optimizer.update(grads, state, model)
        return optax.apply_updates(model, updates), state

    for _ in range(10):
        model, state = step(model, state)
    return model


class TestModelConfig:
    def test_rejects_uneven_head_split(self):
        with pytest.raises(ValueError, match="heads"):
            decoder_only_model.ModelConfig(num_heads=3, qkv_dim=16)

    def test_rejects_odd_head_dim(self):
        with pytest.raises(ValueError, match="even"):
            decoder_only_model.ModelConfig(num_heads=4, qkv_dim=12)


class TestModelStructure:
    def test_parameters_are_exactly_the_trained_layers(self):
        """RoPE contributes no leaves, so its tables can't drift during training."""
        paths = [
            jax.tree_util.keystr(path)
            for path, _ in jax.tree_util.tree_leaves_with_path(_make_model())
        ]
        assert not [p for p in paths if "rope" in p]
        # Embedding, output, and six matrices per block.
        assert len(paths) == 2 + 2 * 6

    def test_blocks_start_as_the_identity(self):
        """Zero-initialized output projections let signal pass straight through."""
        model = _make_model()
        tokens = jnp.arange(8)
        _, acts = model(tokens, return_intermediates=True)
        np.testing.assert_allclose(acts["blocks.1.hidden"], acts["embedding"], atol=1e-6)

    def test_is_causal(self):
        """A later token can't change the logits at earlier positions."""
        model = _trained_a_little()
        tokens = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
        changed = tokens.at[5].set(9)
        before, after = model(tokens), model(changed)
        np.testing.assert_allclose(before[:5], after[:5], atol=1e-5)
        assert not np.allclose(before[5:], after[5:], atol=1e-5)

    def test_shorter_sequences_match_a_prefix(self):
        model = _trained_a_little()
        tokens = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_allclose(model(tokens[:5]), model(tokens)[:5], atol=1e-5)

    def test_batched_matches_unbatched(self):
        model = _trained_a_little()
        tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
        batched = model(tokens)
        np.testing.assert_allclose(batched[1], model(tokens[1]), atol=1e-5)

    def test_logits_are_softcapped(self):
        model = _make_model()
        huge = jax.tree.map(lambda x: x * 1e6, model)
        logits = huge(jnp.arange(8))
        assert jnp.abs(logits).max() <= SMALL_CONFIG.logit_softcap + 1e-3

    def test_remat_does_not_change_outputs(self):
        tokens = jnp.arange(8)
        config = decoder_only_model.ModelConfig(
            num_layers=2,
            num_heads=2,
            qkv_dim=16,
            vocab_size=10,
            seq_len=8,
            remat_policy=jax.checkpoint_policies.nothing_saveable,
        )
        np.testing.assert_allclose(_make_model(config)(tokens), _make_model()(tokens))


class TestReturnIntermediates:
    def test_matches_the_plain_forward_pass(self):
        model = _trained_a_little()
        tokens = jnp.arange(8)
        logits, acts = model(tokens, return_intermediates=True)
        np.testing.assert_allclose(logits, model(tokens), atol=1e-6)
        np.testing.assert_allclose(model.logits(acts["blocks.1.hidden"]), logits, atol=1e-6)

    def test_names_and_shapes(self):
        tokens = jnp.zeros((3, 8), dtype=jnp.int32)
        _, acts = _make_model()(tokens, return_intermediates=True)
        assert set(acts) == {
            "embedding",
            "blocks.0.hidden",
            "blocks.0.attn_weights",
            "blocks.1.hidden",
            "blocks.1.attn_weights",
        }
        assert acts["embedding"].shape == (3, 8, 16)
        assert acts["blocks.0.attn_weights"].shape == (3, 2, 8, 8)

    def test_attention_is_a_causal_distribution(self):
        _, acts = _trained_a_little()(jnp.arange(8), return_intermediates=True)
        weights = acts["blocks.0.attn_weights"]
        np.testing.assert_allclose(weights.sum(-1), 1.0, atol=1e-5)
        np.testing.assert_allclose(jnp.triu(weights, k=1), 0.0, atol=1e-5)

    def test_works_under_jit(self):
        model = _make_model()
        forward = jax.jit(lambda m, t: m(t, return_intermediates=True))
        logits, acts = forward(model, jnp.arange(8))
        assert logits.shape == (8, 10)
        assert "blocks.1.attn_weights" in acts


class TestLoss:
    def test_uniform_logits_give_log_vocab(self):
        logits = jnp.zeros((2, 3, 10))
        targets = jnp.zeros((2, 3), dtype=jnp.int32)
        loss, per_token = decoder_only_model.loss_fn(logits, targets, jnp.ones((2, 3)))
        np.testing.assert_allclose(loss, np.log(10), rtol=1e-6)
        assert per_token.shape == (2, 3)

    def test_mask_excludes_tokens_from_the_mean(self):
        # Position 0 is certain and position 1 is uniform, in a vocabulary of two.
        logits = jnp.array([[[100.0, -100.0], [0.0, 0.0]]])
        targets = jnp.array([[0, 0]])
        all_tokens, _ = decoder_only_model.loss_fn(logits, targets, jnp.ones((1, 2)))
        second, _ = decoder_only_model.loss_fn(logits, targets, jnp.array([[0, 1]]))
        np.testing.assert_allclose(all_tokens, np.log(2) / 2, rtol=1e-5)
        np.testing.assert_allclose(second, np.log(2), rtol=1e-5)

    def test_one_batch_overfits(self):
        config = decoder_only_model.ModelConfig(
            num_layers=2, num_heads=2, qkv_dim=32, vocab_size=10, seq_len=8
        )
        model = _make_model(config)
        tokens = jax.random.randint(jax.random.key(1), (4, 9), 0, 10)
        batch = {
            "inputs": tokens[:, :-1],
            "targets": tokens[:, 1:],
            "mask": jnp.ones((4, 8)),
        }
        optimizer = optax.adam(3e-3)
        state = optimizer.init(model)

        @jax.jit
        def step(model, state):
            (loss, _), grads = jax.value_and_grad(
                decoder_only_model.train_step, has_aux=True
            )(model, batch)
            updates, state = optimizer.update(grads, state, model)
            return optax.apply_updates(model, updates), state, loss

        for _ in range(200):
            model, state, loss = step(model, state)
        assert loss < 0.5


class TestSamplingWindow:
    def _greedy_by_hand(self, model, tokens, count, seq_len):
        tokens = list(tokens)
        for _ in range(count):
            context = jnp.array(tokens[-seq_len:])
            tokens.append(int(jnp.argmax(model(context)[-1])))
        return tokens[-count:]

    def test_short_prompt_matches_stepping_by_hand(self):
        model = _trained_a_little()
        prompt = [1, 2, 3]
        padded = jnp.zeros(8, dtype=jnp.uint16).at[:3].set(jnp.array(prompt, jnp.uint16))
        generated = decoder_only_model.sample(
            model, padded, 3, jax.random.key(0), max_output_tokens=4, temperature=0
        )
        assert generated.tolist() == self._greedy_by_hand(model, prompt, 4, 8)

    def test_slides_when_the_buffer_fills(self):
        """Past seq_len the oldest tokens drop off, as in nanoGPT's generate."""
        model = _trained_a_little()
        prompt = [1, 2, 3, 4, 5, 6]
        padded = jnp.zeros(8, dtype=jnp.uint16).at[:6].set(jnp.array(prompt, jnp.uint16))
        generated = decoder_only_model.sample(
            model, padded, 6, jax.random.key(0), max_output_tokens=7, temperature=0
        )
        assert generated.tolist() == self._greedy_by_hand(model, prompt, 7, 8)

    def test_full_buffer_prompt(self):
        model = _trained_a_little()
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        padded = jnp.array(prompt, dtype=jnp.uint16)
        generated = decoder_only_model.sample(
            model, padded, 8, jax.random.key(0), max_output_tokens=3, temperature=0
        )
        assert generated.tolist() == self._greedy_by_hand(model, prompt, 3, 8)

    def test_top_k_larger_than_the_vocabulary(self):
        model = _make_model()
        padded = jnp.zeros(8, dtype=jnp.uint16)
        generated = decoder_only_model.sample(
            model, padded, 1, jax.random.key(0), max_output_tokens=2, top_k=1000
        )
        assert generated.shape == (2,)

    def test_jit_sample_matches_sample(self):
        model = _trained_a_little()
        padded = jnp.zeros(8, dtype=jnp.uint16).at[0].set(3)
        expected = decoder_only_model.sample(
            model, padded, 1, jax.random.key(3), max_output_tokens=5, top_k=4
        )
        actual = decoder_only_model.jit_sample(
            model, padded, 1, jax.random.key(3), max_output_tokens=5, top_k=4
        )
        np.testing.assert_array_equal(actual, expected)


# Every nanochat addition on: windows of 2, 2, 2 and 8, value embeddings on layers 1
# and 3, the residual scalars, smear and backout.
NANOCHAT_CONFIG = decoder_only_model.ModelConfig(
    num_layers=4, num_heads=2, qkv_dim=32, vocab_size=10, seq_len=8,
    **decoder_only_model.NANOCHAT,
)


def _trained_nanochat():
    """Train enough that smear and the gates move off their initial values."""
    return _trained_a_little(NANOCHAT_CONFIG)


class TestNanochatAdditions:
    def test_windows_tile_the_pattern_and_end_long(self):
        assert NANOCHAT_CONFIG.window_sizes() == [2, 2, 2, 8]
        config = decoder_only_model.ModelConfig(
            num_layers=12, num_heads=6, qkv_dim=768, seq_len=2_048, window_pattern="SSSL"
        )
        assert config.window_sizes() == [512] * 3 + [2_048] + [512] * 3 + [2_048] + [512] * 3 + [2_048]

    def test_rejects_unknown_window_letters(self):
        with pytest.raises(ValueError, match="window_pattern"):
            decoder_only_model.ModelConfig(window_pattern="SXL")

    def test_value_embeddings_alternate_and_include_the_last_layer(self):
        assert [NANOCHAT_CONFIG.has_value_embedding(i) for i in range(4)] == [False, True, False, True]
        odd = decoder_only_model.ModelConfig(num_layers=3, value_embeddings=True)
        assert [odd.has_value_embedding(i) for i in range(3)] == [True, False, True]

    def test_adds_only_the_new_parameters(self):
        paths = {
            jax.tree_util.keystr(path)
            for path, _ in jax.tree_util.tree_leaves_with_path(_make_model(NANOCHAT_CONFIG))
        }
        extra = {p for p in paths if not p.startswith((".blocks", ".embedding_layer", ".output_layer"))}
        assert extra == {
            ".value_embeds[0].weights",
            ".value_embeds[1].weights",
            ".resid_lambdas",
            ".x0_lambdas",
            ".smear_gate",
            ".smear_lambda",
            ".backout_lambda",
        }
        gates = {p for p in paths if "value_gate" in p}
        assert gates == {".blocks[1].causal_attn.value_gate", ".blocks[3].causal_attn.value_gate"}

    def test_off_by_default(self):
        model = _make_model()
        assert not hasattr(model, "value_embeds") and not hasattr(model, "smear_gate")

    def test_is_causal(self):
        model = _trained_nanochat()
        tokens = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
        changed = tokens.at[5].set(9)
        before, after = model(tokens), model(changed)
        np.testing.assert_allclose(before[:5], after[:5], atol=1e-5)
        assert not np.allclose(before[5:], after[5:], atol=1e-5)

    def test_smear_is_learned_and_mixes_the_previous_token(self):
        model = _trained_nanochat()
        assert float(jnp.abs(model.smear_lambda[0])) > 0

    def test_short_windows_ignore_older_tokens(self):
        _, acts = _trained_nanochat()(jnp.arange(8), return_intermediates=True)
        weights = acts["blocks.0.attn_weights"]  # [heads, query, key]
        distance = jnp.arange(8)[:, None] - jnp.arange(8)[None, :]
        np.testing.assert_allclose(jnp.where(distance > 2, weights, 0.0), 0.0, atol=1e-6)
        np.testing.assert_allclose(weights.sum(-1), 1.0, atol=1e-5)
        last = acts["blocks.3.attn_weights"]
        assert float(last[:, 7, 0].min()) > 0  # The long layer still sees the first token.

    def test_batched_matches_unbatched(self):
        model = _trained_nanochat()
        tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
        np.testing.assert_allclose(model(tokens)[1], model(tokens[1]), atol=1e-5)

    def test_shorter_sequences_match_a_prefix(self):
        model = _trained_nanochat()
        tokens = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_allclose(model(tokens[:5]), model(tokens)[:5], atol=1e-5)

    def test_intermediates_match_the_plain_forward_pass(self):
        model = _trained_nanochat()
        logits, acts = model(jnp.arange(8), return_intermediates=True)
        np.testing.assert_allclose(logits, model(jnp.arange(8)), atol=1e-5)
        assert set(acts) == {"embedding"} | {
            f"blocks.{i}.{name}" for i in range(4) for name in ("hidden", "attn_weights")
        }

    def test_remat_does_not_change_outputs(self):
        model = _trained_nanochat()
        remat = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(
                _make_model(
                    decoder_only_model.ModelConfig(
                        **{**NANOCHAT_CONFIG.__dict__,
                           "remat_policy": jax.checkpoint_policies.nothing_saveable}
                    )
                )
            ),
            jax.tree_util.tree_leaves(model),
        )
        np.testing.assert_allclose(remat(jnp.arange(8)), model(jnp.arange(8)), atol=1e-6)

    @pytest.mark.parametrize("batched", [False, True])
    def test_fused_attention_matches_the_masked_matmuls(self, batched):
        model = _trained_nanochat()
        fused_config = decoder_only_model.ModelConfig(
            **{**NANOCHAT_CONFIG.__dict__, "attention_impl": "xla"}
        )
        fused = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(_make_model(fused_config)),
            jax.tree_util.tree_leaves(model),
        )
        tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
        tokens = tokens if batched else tokens[0]
        np.testing.assert_allclose(fused(tokens), model(tokens), atol=1e-5)

    def test_one_batch_overfits(self):
        model = _make_model(NANOCHAT_CONFIG)
        tokens = jax.random.randint(jax.random.key(1), (2, 8), 0, 10)
        batch = {"inputs": tokens[:, :-1], "targets": tokens[:, 1:], "mask": jnp.ones((2, 7))}
        optimizer = optax.adam(3e-2)
        state = optimizer.init(model)

        @jax.jit
        def step(model, state):
            loss, grads = jax.value_and_grad(
                lambda m: decoder_only_model.train_step(m, batch)[0]
            )(model)
            updates, state = optimizer.update(grads, state, model)
            return optax.apply_updates(model, updates), state, loss

        for _ in range(400):
            model, state, loss = step(model, state)
        assert float(loss) < 0.1

    def test_jit_sample_runs(self):
        model = _trained_nanochat()
        padded = jnp.zeros(8, dtype=jnp.uint16).at[0].set(3)
        expected = decoder_only_model.sample(
            model, padded, 1, jax.random.key(3), max_output_tokens=10, temperature=0
        )
        actual = decoder_only_model.jit_sample(
            model, padded, 1, jax.random.key(3), max_output_tokens=10, temperature=0
        )
        np.testing.assert_array_equal(actual, expected)
