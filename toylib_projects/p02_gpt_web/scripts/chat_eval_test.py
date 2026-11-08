"""Tests for scripts/chat_eval.py, on a character tokenizer and hand-built models."""

import json

import numpy as np
import pytest

from toylib_projects.p02_gpt_web.scripts import chat_eval

BOS = 0
FMT = chat_eval.Transcript(lambda text: [ord(c) for c in text], BOS)


def encode(text: str) -> list[int]:
    return [ord(c) for c in text]


def _mc(answer="B", letters=("A", "B", "C", "D"), question="Q?"):
    return chat_eval.Problem(chat_eval.render_mc(question, letters, ["w", "x", "y", "z"]), answer, letters)


class TestData:
    def test_render_mc_matches_nanochat(self):
        text = chat_eval.render_mc("Which?", ["A", "B"], ["red", "blue"])
        assert text == "Multiple Choice question: Which?\n- red=A\n- blue=B\n\nRespond only with the letter of the correct answer."

    def test_gsm8k_number(self):
        assert chat_eval.gsm8k_number("so 3 + 4 = 7\n#### 1,234") == "1234"
        assert chat_eval.gsm8k_number("#### -2.5 dollars") == "-2.5"
        assert chat_eval.gsm8k_number("the answer is 7") is None

    def test_load_hub_reads_the_cache_in_nanochats_order(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        folder = tmp_path / "org--name" / "default" / "test"
        folder.mkdir(parents=True)
        pq.write_table(pa.table({"x": list(range(5))}), folder / "00000.parquet")
        pq.write_table(pa.table({"x": list(range(5, 8))}), folder / "00001.parquet")
        (folder / "manifest.json").write_text(json.dumps(["00000.parquet", "00001.parquet"]))
        rows = chat_eval.load_hub("org/name", "default", "test", data_dir=tmp_path)
        assert [r["x"] for r in rows] == list(np.random.default_rng(42).permutation(8))


class TestPrompts:
    def test_prompt_is_a_transcript_ending_at_the_reply(self):
        ids = FMT.prompt("Hi there", max_len=1000)
        text = "".join(map(chr, ids[1:]))
        assert ids[0] == BOS and text.endswith("User: Hi there\nAssistant:")

    def test_long_prompts_keep_their_end(self):
        ids = FMT.prompt("x" * 500, max_len=50)
        assert len(ids) == 50 and "".join(map(chr, ids)).endswith("\nAssistant:")

    def test_choice_tokens_share_the_space(self):
        shared, ids = chat_eval.choice_tokens(FMT, ["A", "B", "C"])
        assert shared == encode(" ") and ids == encode("ABC")

    def test_choice_tokens_reject_answers_that_differ_twice(self):
        with pytest.raises(ValueError):
            chat_eval.choice_tokens(FMT, ["A", "AB"])


class TestCategorical:
    def _prefers(self, letter):
        """A model whose logits at every position favor one character."""

        def last_logits(tokens, positions):
            assert (tokens[np.arange(len(tokens)), positions] == ord(" ")).all()  # Reads the end.
            logits = np.zeros((len(tokens), 256))
            logits[:, ord(letter)] = 1.0
            return logits

        return last_logits

    def test_picks_the_highest_letter_among_the_choices(self):
        problems = [_mc("B"), _mc("C"), _mc("B", letters=("A", "C", "D", "E"))]
        correct = chat_eval.categorical_correct(self._prefers("B"), problems, FMT, 4096, 4096)
        # The third problem doesn't offer "B", so its pick falls among the others.
        assert correct[:2] == [True, False]

    def test_batching_does_not_change_results(self):
        problems = [_mc("ABCD"[i % 3], question="Q" * (i * 37 % 300)) for i in range(40)]

        def by_length(tokens, positions):  # Depends only on each row's own prompt.
            logits = np.zeros((len(tokens), 256))
            for row, pos in enumerate(positions):
                logits[row, ord("ABCD"[pos % 4])] = 1.0
            return logits

        small = chat_eval.categorical_correct(by_length, problems, FMT, 4096, 256)
        large = chat_eval.categorical_correct(by_length, problems, FMT, 4096, 65536)
        assert small == large and any(small) and not all(small)


class TestGenerative:
    def test_gsm8k_grades_the_final_number(self):
        p = chat_eval.Problem("q", "18")
        assert chat_eval.grade("GSM8K", p, "16 - 3 - 4 = 9 eggs, 9 * 2 = 18\n#### 18")
        assert not chat_eval.grade("GSM8K", p, "The answer is 18.")

    def _humaneval(self):
        prompt = 'import math\n\n\ndef root(x):\n    """Square root."""\n'
        test = "def check(f):\n    assert f(9) == 3\n    assert f(16) == 4\n"
        return chat_eval.Problem(prompt, "", entry_point="root", test=test)

    def test_humaneval_runs_the_tests(self):
        p = self._humaneval()
        good = "def root(x):\n    return int(math.sqrt(x))"
        assert chat_eval.passes_tests(p, good)
        assert chat_eval.passes_tests(p, f"Here you go:\n```python\n{good}\n```\nDone.")
        assert not chat_eval.passes_tests(p, "def root(x):\n    return x")
        assert not chat_eval.passes_tests(p, "this is not python")

    def test_humaneval_times_out(self):
        assert not chat_eval.passes_tests(self._humaneval(), "def root(x):\n    while True: pass", timeout=1.0)

    def test_results_stay_in_problem_order(self):
        problems = [chat_eval.Problem("q" * n, str(n)) for n in (30, 5, 12)]

        def complete(prompts):  # Answers each prompt with its own length.
            return [f"#### {len(p) - len(FMT.prompt('', 10**6))}" for p in prompts]

        results = chat_eval.generative_results(complete, "GSM8K", problems, FMT, 10**6, batch_size=2, verbose=False)
        assert [text for text, _ in results] == ["#### 30", "#### 5", "#### 12"]
        assert all(ok for _, ok in results)


def test_summarize_reproduces_nanochats_d20_sft_chatcore():
    """nanochat's October 2025 d20 report card: SFT row, ChatCORE 0.0884."""
    sft = {"ARC-Easy": 0.3876, "ARC-Challenge": 0.2807, "MMLU": 0.3151, "GSM8K": 0.0455, "HumanEval": 0.0854}
    out = chat_eval.summarize(sft)
    assert out["ChatCORE"] == pytest.approx(0.0884, abs=1e-4)
    assert "ChatCORE" not in chat_eval.summarize({"MMLU": 0.3})


def test_chat_tokens_prompt_like_nanochat():
    from toylib_projects.p02_gpt_web import chat_format
    from toylib_projects.p02_gpt_web.tokenizer import bpe

    tok = chat_format.ChatTokenizer(bpe.Tokenizer([]))
    fmt = chat_eval.ChatTokens(tok)
    assert tok.decode(fmt.prompt("Hi", 100)) == "<|endoftext|><|user_start|>Hi<|user_end|><|assistant_start|>"
    shared, ids = chat_eval.choice_tokens(fmt, ["A", "B", "1"])
    assert shared == [] and ids == tok.encode("AB1")  # No space after <|assistant_start|>.
