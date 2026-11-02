"""Tests for probes.py, on a small untrained or lightly trained model."""

import collections

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from toylib_projects.p02_tinystories import decoder_only_model
from toylib_projects.p02_tinystories import experiment
from toylib_projects.p02_tinystories import probes
from toylib_projects.p02_tinystories.tokenizer import bpe

CONFIG = decoder_only_model.ModelConfig(
    num_layers=2, num_heads=2, qkv_dim=16, vocab_size=32, seq_len=128
)


@pytest.fixture(scope="module")
def model():
    model = experiment.build_model(CONFIG, jax.random.key(0))
    tokens = jax.random.randint(jax.random.key(1), (4, 17), 0, 32)
    batch = {
        "inputs": tokens[:, :-1],
        "targets": tokens[:, 1:],
        "mask": jnp.ones((4, 16)),
    }
    optimizer = optax.adam(1e-2)
    state = optimizer.init(model)
    for _ in range(5):
        grads = jax.grad(lambda m: decoder_only_model.train_step(m, batch)[0])(model)
        updates, state = optimizer.update(grads, state, model)
        model = optax.apply_updates(model, updates)
    return model


class TestNgramBits:
    def test_uniform_is_log_vocab(self):
        tokens = np.arange(100) % 8
        assert probes.ngram_bits(tokens, tokens, 8)["uniform"] == 3.0

    def test_longer_contexts_win_on_patterned_text(self):
        tokens = np.tile([0, 1, 2, 3, 2, 1], 200)
        bits = probes.ngram_bits(tokens, tokens, 8)
        assert bits["trigram"] < bits["bigram"] < bits["unigram"] < bits["uniform"]

    def test_unseen_contexts_stay_finite(self):
        train = np.tile([0, 1], 50)
        val = np.array([1, 1, 0, 0, 3, 3])
        assert np.isfinite(list(probes.ngram_bits(train, val, 4).values())).all()

    def test_conditional_matches_counts_by_hand(self):
        # After 0 comes 1 three times and 2 once, and val only asks about 0 -> 1.
        train = np.array([0, 1, 0, 1, 0, 1, 0, 2])
        val = np.array([0, 0, 0, 1])
        bits = probes.ngram_bits(train, val, 3)
        unigram = 4 / 8, 3 / 8
        expected = -np.mean(
            [
                np.log2(0.9 * 1 / 4 * 0 + 0.099 * unigram[0] + 0.001 / 3),
                np.log2(0.9 * 3 / 4 + 0.099 * unigram[1] + 0.001 / 3),
            ]
        )
        assert bits["bigram"] == pytest.approx(expected)


def test_token_nats_matches_the_training_loss(model):
    tokens = np.arange(17)[None]
    logits = model(jnp.asarray(tokens[:, :-1]))
    _, per_token = decoder_only_model.loss_fn(
        logits, jnp.asarray(tokens[:, 1:]), jnp.ones((1, 16))
    )
    np.testing.assert_allclose(probes.token_nats(model, tokens), per_token, rtol=1e-5)


def test_attention_by_distance_shape_and_scale(model):
    tokens = np.arange(9).reshape(1, 9)
    result = probes.attention_by_distance(model, tokens, max_distance=8)
    assert result.shape == (2, 2, 9)
    # Each token's weights sum to one, so the mean weight per distance can't exceed it.
    assert (result >= 0).all() and (result <= 1).all()


def test_attention_by_distance_finds_a_known_pattern(monkeypatch):
    weights = jnp.eye(4)[None, None]  # Everyone attends only to itself.
    fake = type("M", (), {"config": type("C", (), {"num_layers": 1})})()
    monkeypatch.setattr(
        probes, "_forward_acts", lambda m, t: (None, {"blocks.0.attn_weights": weights})
    )
    result = probes.attention_by_distance(fake, np.zeros((1, 4)), max_distance=2)
    np.testing.assert_allclose(result[0, 0], [1, 0, 0])


def test_lens_last_entry_is_the_real_output(model):
    tokens = np.arange(8)
    lens = probes.lens_logits(model, tokens)
    assert len(lens) == CONFIG.num_layers + 1
    np.testing.assert_allclose(lens[-1], model(jnp.asarray(tokens)), atol=1e-5)


def test_lens_nats_has_one_value_per_depth(model):
    assert probes.lens_nats(model, np.arange(9)[None]).shape == (3,)


def test_context_curve_uses_only_the_last_k_tokens(model):
    windows = np.arange(10)[None] % 32
    other = windows.copy()
    other[0, :5] = 31  # Changed, but more than 4 tokens before the target.
    same = probes.context_curve(model, windows, [1, 4])
    assert same == pytest.approx(probes.context_curve(model, other, [1, 4]))


def test_repeat_test_keys(model):
    result = probes.repeat_test(
        model, length=8, num_sequences=2, vocab_range=(2, 30)
    )
    assert set(result) == {"repeated", "shuffled"}


class TestNameProbes:
    class Tokenizer:
        def __call__(self, texts, **_):
            return {"input_ids": [[ord(c) % 32 for c in t] for t in texts]}

    def test_shapes(self, model):
        result = probes.name_probes(
            model, self.Tokenizer(), ["ab", "c"], [("xy", "z")]
        )
        assert result["recall"].shape == (2,)
        assert np.isfinite(result["finish"][0])
        assert np.isnan(result["finish"][1])

    def test_margins_are_the_differences_of_their_log_probs(self, model):
        result = probes.name_probes(
            model, self.Tokenizer(), ["ab", "cd"], [("xy", "z")]
        )
        np.testing.assert_allclose(
            result["recall"], result["recall_seen"] - result["recall_other"]
        )
        assert np.all(result["recall_seen"] < 0)
        assert np.all(result["recall_none"] < 0)

    def test_num_controls_limits_the_comparison(self, model):
        names, templates = ["ab", "cd", "ef"], [("xy", "z")]
        limited = probes.name_probes(
            model, self.Tokenizer(), names, templates, num_controls=1
        )
        full = probes.name_probes(model, self.Tokenizer(), names, templates)
        assert not np.allclose(limited["recall_other"], full["recall_other"])
        np.testing.assert_allclose(limited["recall_seen"], full["recall_seen"], atol=1e-5)

    def test_last_depth_matches_the_output(self, model):
        args = (model, self.Tokenizer(), ["ab", "cd"], [("xy", "z")])
        np.testing.assert_allclose(
            probes.name_probes(*args, depth=CONFIG.num_layers)["recall"],
            probes.name_probes(*args)["recall"],
            atol=1e-4,
        )

    def test_name_reuse_is_a_fraction(self, model):
        class Tokenizer(self.Tokenizer):
            def decode(self, ids):
                return "".join(chr(ord("a") + i % 26) for i in ids)

        reuse = probes.name_reuse(
            model, Tokenizer(), ["a"], [("x", "y")], num_samples=2, max_output_tokens=8
        )
        assert 0 <= reuse <= 1


class TestFirstVsRepeat:
    STORIES = [np.array([0, 5, 6, 5, 6, 7]), np.array([0, 6, 8, 8])]

    def test_counts_each_band(self, model):
        counts = np.zeros(32, dtype=int)
        counts[[5, 6]] = 10
        bands = probes.first_vs_repeat(model, self.STORIES, counts, [0, 5, 100], length=6)
        assert len(bands) == 2
        for band in bands:
            assert band["first"] > 0 and band["repeat"] > 0

    def test_matches_token_nats_by_hand(self, model):
        counts = np.ones(32, dtype=int)
        story = np.array([0, 5, 5, 5])
        bands = probes.first_vs_repeat(model, [story], counts, [0, 2], length=4)
        bits = probes.token_nats(model, story[None])[0] / np.log(2)
        assert bands[0]["first"] == pytest.approx(bits[0], abs=1e-5)
        assert bands[0]["repeat"] == pytest.approx(bits[1:].mean(), abs=1e-5)


class TestFindInventedNames:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        text = " Bl Sn Tr Gr Dr Fl Kr Sk Pl St" * 3
        return bpe.train(collections.Counter(bpe.split(text)), 400)

    def test_names_follow_the_rules(self, tokenizer):
        names = probes.find_invented_names(
            tokenizer, np.array([0, 1, 2], dtype=np.uint16), count=2
        )
        assert len(names) == 2
        ids = [tokenizer([name])["input_ids"][0] for name in names]
        assert all(2 <= len(i) <= 4 for i in ids)
        assert not set(ids[0]) & set(ids[1])

    def test_rejects_pairs_seen_in_training(self, tokenizer):
        names = probes.find_invented_names(
            tokenizer, np.array([0, 1, 2], dtype=np.uint16), count=1
        )
        ids = tokenizer(names)["input_ids"][0]
        training = np.array(ids, dtype=np.uint16)
        assert names[0] not in probes.find_invented_names(tokenizer, training, count=2)

    def test_is_deterministic(self, tokenizer):
        tokens = np.array([0, 1, 2], dtype=np.uint16)
        assert probes.find_invented_names(
            tokenizer, tokens, count=2
        ) == probes.find_invented_names(tokenizer, tokens, count=2)


def test_nearest_tokens_excludes_itself():
    weights = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    assert probes.nearest_tokens(weights, 0, k=1) == [1]


def test_pca_2d_finds_the_main_axis():
    x = np.linspace(-1, 1, 11)
    points = np.stack([x, 0 * x, 0.01 * x**2], axis=1)
    projected = probes.pca_2d(points)
    assert projected.shape == (11, 2)
    np.testing.assert_allclose(np.abs(projected[:, 0]), np.abs(x), atol=1e-2)
