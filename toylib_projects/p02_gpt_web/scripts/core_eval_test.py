"""Tests for scripts/core_eval.py, on a character tokenizer and hand-built forwards."""

import random

import numpy as np
import pytest

from toylib_projects.p02_gpt_web.scripts import core_eval

BOS = 0


def encode(text: str) -> list[int]:
    return [ord(c) for c in text]


def _task(task_type, data, num_fewshot=0, delimiter=" ", baseline=25.0):
    return core_eval.Task("t", task_type, num_fewshot, delimiter, baseline, data)


def _mc(gold=1):
    return {"query": "Q", "choices": ["ab", "cd", "ef"], "gold": gold}


def _next_token(tokens):
    """A forward that predicts every next token perfectly, at zero loss."""
    return np.zeros(tokens.shape), np.roll(tokens, -1, axis=1)


class TestPrompts:
    def test_multiple_choice_judges_only_the_choice(self):
        prompts = core_eval.task_prompts(_task("multiple_choice", [_mc()]), encode, BOS, 100)
        assert [p.option for p in prompts] == [0, 1, 2]
        for p, choice in zip(prompts, ["ab", "cd", "ef"]):
            assert p.tokens == [BOS] + encode("Q " + choice)
            assert p.tokens[p.start : p.end] == encode(choice)

    def test_schema_judges_the_shared_continuation(self):
        item = {"context_options": ["He ran", "She sat"], "continuation": "fast.", "gold": 0}
        prompts = core_eval.task_prompts(_task("schema", [item]), encode, BOS, 100)
        for p in prompts:
            assert p.tokens[p.start : p.end] == encode(" fast.")

    def test_language_modeling_judges_the_continuation(self):
        item = {"context": "The sky is  ", "continuation": "blue"}
        (p,) = core_eval.task_prompts(_task("language_modeling", [item]), encode, BOS, 100)
        # The context is trimmed before the delimiter, and nanochat strips the prompt
        # without the continuation, so the delimiter is judged with it.
        assert p.tokens == [BOS] + encode("The sky is blue")
        assert p.tokens[p.start : p.end] == encode(" blue")

    def test_fewshot_examples_match_nanochat(self):
        """nanochat draws item i's examples with Random(1234 + i), excluding item i."""
        data = [{"query": f"q{i}", "choices": ["x", "y"], "gold": i % 2} for i in range(6)]
        prompts = core_eval.task_prompts(_task("multiple_choice", data, num_fewshot=2), encode, BOS, 1000)
        picked = random.Random(1234 + 3).sample([0, 1, 2, 4, 5], 2)
        expected = "".join(f"q{j} {'xy'[j % 2]}\n\n" for j in picked) + "q3 x"
        assert [p for p in prompts if p.item == 3][0].tokens == [BOS] + encode(expected)

    def test_long_prompts_keep_their_end(self):
        item = {"query": "Q" * 50, "choices": ["ab", "cd"], "gold": 0}
        prompts = core_eval.task_prompts(_task("multiple_choice", [item]), encode, BOS, 10)
        for p, choice in zip(prompts, ["ab", "cd"]):
            assert len(p.tokens) == 10 and p.tokens[p.start : p.end] == encode(choice)


class TestScoring:
    def test_multiple_choice_picks_the_lowest_mean_loss(self):
        task = _task("multiple_choice", [_mc(gold=1), _mc(gold=2)])
        prompts = core_eval.task_prompts(task, encode, BOS, 100)
        good = encode("cd")

        def forward(tokens):  # Cheap only for the tokens of choice "cd".
            nxt = np.roll(tokens, -1, axis=1)
            return np.where(np.isin(nxt, good), 0.1, 5.0), nxt

        loss, match = core_eval.score_prompts(forward, prompts, BOS, token_budget=128)
        assert core_eval.task_accuracy(task, prompts, loss, match) == 0.5

    def test_language_modeling_needs_every_token(self):
        task = _task("language_modeling", [{"context": "a", "continuation": "bc"}])
        prompts = core_eval.task_prompts(task, encode, BOS, 100)

        def misses_c(tokens):
            losses, nxt = _next_token(tokens)
            return losses, np.where(nxt == ord("c"), 1, nxt)

        for forward, expected in [(_next_token, 1.0), (misses_c, 0.0)]:
            loss, match = core_eval.score_prompts(forward, prompts, BOS, 128)
            assert core_eval.task_accuracy(task, prompts, loss, match) == expected

    def test_batching_does_not_change_scores(self):
        data = [{"query": "Q" * (i * 7 % 90), "choices": ["ab", "cdef"], "gold": 0} for i in range(30)]
        prompts = core_eval.task_prompts(_task("multiple_choice", data), encode, BOS, 1000)

        def forward(tokens):  # Depends only on each position's own tokens, like a causal model.
            return (tokens % 7).astype(float), np.roll(tokens, -1, axis=1)

        small = core_eval.score_prompts(forward, prompts, BOS, token_budget=64)
        large = core_eval.score_prompts(forward, prompts, BOS, token_budget=4096)
        np.testing.assert_array_equal(small[0], large[0])
        np.testing.assert_array_equal(small[1], large[1])

    def test_bucket(self):
        assert [core_eval.bucket(n) for n in (1, 64, 65, 300, 2048)] == [64, 64, 128, 512, 2048]

    def test_centered(self):
        assert core_eval.centered(0.25, 25.0) == 0.0
        assert core_eval.centered(1.0, 25.0) == 1.0
        assert core_eval.centered(0.5, 0.0) == 0.5


class TestBundle:
    def test_csv_round_trip(self, tmp_path):
        out = {"results": {"a": 0.5, "b": 0.25}, "centered_results": {"a": 0.4, "b": 0.0}, "core_metric": 0.2}
        core_eval.write_csv(tmp_path / "x.csv", out)
        assert core_eval.read_csv(tmp_path / "x.csv") == {
            "a": (0.5, 0.4), "b": (0.25, 0.0), "CORE": (None, 0.2)
        }

    @pytest.mark.skipif(
        not (core_eval.EVAL_BUNDLE_DIR / "core.yaml").exists(), reason="eval bundle not downloaded"
    )
    def test_bundle_has_22_tasks_and_gpt2_reference(self):
        tasks = core_eval.load_tasks(core_eval.EVAL_BUNDLE_DIR, max_per_task=5)
        assert len(tasks) == 22
        ref = core_eval.read_csv(core_eval.EVAL_BUNDLE_DIR / "openai-community-gpt2.csv")
        assert set(ref) == {t.label for t in tasks} | {"CORE"}
        assert ref["CORE"][1] == pytest.approx(np.mean([ref[t.label][1] for t in tasks]), abs=1e-5)
