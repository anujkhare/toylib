"""Tests for generate.py: the cached forward must match the model's own."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from toylib.nn import module
from toylib_projects.p02_gpt_web import decoder_only_model
from toylib_projects.p02_gpt_web import generate

# Every nanochat addition on, with windows of 4, 4, 4 and 16, so prompts longer than a
# window test the sliding mask.
CONFIG = decoder_only_model.ModelConfig(
    num_layers=4, num_heads=2, qkv_dim=32, vocab_size=50, seq_len=16,
    **decoder_only_model.NANOCHAT,
)


def _model(config=CONFIG, seed=0):
    """A model with every trainable weight perturbed, so smear, gates and backout all matter."""
    model = decoder_only_model.DecoderOnlyTransformer(config=config, key=jax.random.key(seed))
    model.init()
    leaves, treedef = jax.tree.flatten(model)
    mask = jax.tree.leaves(module.trainable_mask(model))
    keys = jax.random.split(jax.random.key(seed + 1), len(leaves))
    leaves = [
        x + 0.3 * jax.random.normal(k, x.shape, x.dtype) if trainable else x
        for x, k, trainable in zip(leaves, keys, mask)
    ]
    return jax.tree.unflatten(treedef, leaves)


def _tokens(n, seed=0):
    return jax.random.randint(jax.random.key(seed), (n,), 0, CONFIG.vocab_size)


def _expected(model, tokens):
    return np.asarray(model(tokens))


class TestForward:
    def test_whole_prompt_matches_the_model(self):
        model, tokens = _model(), _tokens(16)
        hidden, _ = generate.forward(model, tokens, jnp.int32(0), generate.empty_cache(model))
        np.testing.assert_allclose(model.logits(hidden), _expected(model, tokens), atol=1e-4)

    @pytest.mark.parametrize("split", [1, 5, 11])
    def test_one_token_at_a_time_matches_the_model(self, split):
        """A prompt, then single steps past the windows of 4, as sampling runs."""
        model, tokens = _model(), _tokens(16)
        expected = _expected(model, tokens)
        hidden, cache = generate.forward(model, tokens[:split], jnp.int32(0), generate.empty_cache(model))
        np.testing.assert_allclose(model.logits(hidden), expected[:split], atol=1e-4)
        for pos in range(split, 16):
            hidden, cache = generate.forward(model, tokens[pos : pos + 1], jnp.int32(pos), cache)
            np.testing.assert_allclose(model.logits(hidden[0]), expected[pos], atol=1e-4)

    def test_padding_after_the_prompt_changes_nothing(self):
        model, tokens = _model(), _tokens(16)
        expected = _expected(model, tokens)
        padded = jnp.zeros(8, jnp.int32).at[:5].set(tokens[:5])
        hidden, cache = generate.forward(model, padded, jnp.int32(0), generate.empty_cache(model), jnp.int32(5))
        np.testing.assert_allclose(model.logits(hidden[:5]), expected[:5], atol=1e-4)
        for pos in range(5, 16):
            hidden, cache = generate.forward(model, tokens[pos : pos + 1], jnp.int32(pos), cache)
            np.testing.assert_allclose(model.logits(hidden[0]), expected[pos], atol=1e-4)

    def test_plain_architecture(self):
        """Without nanochat's additions too, as the older runs were trained."""
        config = dataclasses.replace(
            CONFIG, window_pattern="L", value_embeddings=False, residual_scalars=False,
            smear=False, backout=False, qk_scale=1.0,
        )
        model, tokens = _model(config), _tokens(16)
        hidden, cache = generate.forward(model, tokens[:3], jnp.int32(0), generate.empty_cache(model))
        for pos in range(3, 16):
            hidden, cache = generate.forward(model, tokens[pos : pos + 1], jnp.int32(pos), cache)
        np.testing.assert_allclose(model.logits(hidden[0]), _expected(model, tokens)[-1], atol=1e-4)


class TestGenerator:
    def test_greedy_matches_sample(self):
        model, prompt = _model(), [int(t) for t in _tokens(3)]
        padded = jnp.zeros(CONFIG.seq_len, jnp.int32).at[:3].set(jnp.array(prompt))
        expected = decoder_only_model.sample(
            model, padded, 3, jax.random.key(0), max_output_tokens=10, temperature=0.0
        )
        actual = list(generate.Generator(model).stream(prompt, max_new_tokens=10, temperature=0.0))
        assert actual == expected.tolist()

    def test_stops_when_the_context_is_full(self):
        model, prompt = _model(), [int(t) for t in _tokens(10)]
        out = list(generate.Generator(model).stream(prompt, max_new_tokens=100, temperature=0.0))
        # The last token may fill the 16th position, but nothing past it.
        assert len(out) == CONFIG.seq_len - len(prompt) + 1

    def test_stop_token_is_not_yielded(self):
        model, prompt = _model(), [int(t) for t in _tokens(3)]
        gen = generate.Generator(model)
        full = list(gen.stream(prompt, max_new_tokens=8, temperature=0.0))
        cut = list(gen.stream(prompt, max_new_tokens=8, temperature=0.0, stop_tokens={full[2]}))
        assert cut == full[: full.index(full[2])]

    def test_same_seed_same_sample(self):
        model, prompt = _model(), [int(t) for t in _tokens(3)]
        gen = generate.Generator(model)
        a = list(gen.stream(prompt, max_new_tokens=8, seed=7, top_k=10))
        b = list(gen.stream(prompt, max_new_tokens=8, seed=7, top_k=10))
        assert a == b and len(a) == 8

    def test_rejects_an_empty_or_full_prompt(self):
        gen = generate.Generator(_model())
        with pytest.raises(ValueError):
            next(gen.stream([]))
        with pytest.raises(ValueError):
            next(gen.stream([1] * CONFIG.seq_len))


class TestGreedyBatch:
    def _single(self, model, prompt, n, **kwargs):
        return list(generate.Generator(model).stream(prompt, max_new_tokens=n, temperature=0.0, **kwargs))

    def test_rows_match_one_prompt_at_a_time(self):
        """Prompts of different lengths, each decoding at its own position."""
        model = _model()
        prompts = [[int(t) for t in _tokens(n, seed=n)] for n in (1, 3, 6)]
        outs = generate.greedy_batch(model, prompts, max_new_tokens=6)
        assert outs == [self._single(model, p, 6) for p in prompts]

    def test_rows_end_at_their_own_stop_token(self):
        model = _model()
        prompts = [[int(t) for t in _tokens(3, seed=s)] for s in (0, 1)]
        full = generate.greedy_batch(model, prompts, max_new_tokens=8)
        stop = full[0][3]
        outs = generate.greedy_batch(model, prompts, max_new_tokens=8, stop_tokens={stop})
        assert outs == [row[: row.index(stop)] if stop in row else row for row in full]

    def test_stops_when_the_context_is_full(self):
        model = _model()
        outs = generate.greedy_batch(model, [[1] * 10, [2] * 4], max_new_tokens=100)
        assert [len(o) for o in outs] == [CONFIG.seq_len - 10 + 1, CONFIG.seq_len - 4 + 1]

    def test_is_done_ends_a_row_early(self):
        model = _model()
        outs = generate.greedy_batch(model, [[1, 2]], max_new_tokens=12, is_done=lambda out: len(out) >= 2, check_every=4)
        assert len(outs[0]) == 4
