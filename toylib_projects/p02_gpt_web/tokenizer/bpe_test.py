"""Tests for bpe.py."""

import collections

import numpy as np
import pytest

from toylib_projects.p02_gpt_web.tokenizer import bpe

TEXT = "Once upon a time, Lily's dog said “hello” to 12 friends.\n\nThe end_of_it!  "


def tokenizer_for(text: str, vocab_size: int = 300) -> bpe.Tokenizer:
    return bpe.train(collections.Counter(bpe.split(text)), vocab_size)


def slow_train(chunk_counts: dict[str, int], vocab_size: int) -> list[bpe.Pair]:
    """Recount every pair for every merge, as minbpe does."""
    words = {chunk: [b + 1 for b in chunk.encode()] for chunk in chunk_counts}
    merges = []
    for new_id in range(bpe.FIRST_MERGE_ID, vocab_size):
        counts = collections.Counter()
        for chunk, word in words.items():
            for pair in zip(word, word[1:]):
                counts[pair] += chunk_counts[chunk]
        if not counts:
            break
        best = max(counts, key=lambda pair: (counts[pair], pair))
        merges.append(best)
        words = {chunk: bpe.merge(word, best, new_id) for chunk, word in words.items()}
    return merges


class TestSplit:
    def test_loses_nothing(self):
        assert "".join(bpe.split(TEXT)) == TEXT

    def test_boundaries(self):
        assert bpe.split('Lily\'s dog said, "Hi!" 123 times_two.') == [
            "Lily", "'s", " dog", " said", ",", ' "', "Hi", '!"', " ",
            "12", "3", " times", "_two", ".",
        ]  # fmt: skip

    def test_a_quote_joins_the_word_after_it(self):
        assert bpe.split('"Hi') == ['"Hi']

    def test_keeps_accented_letters_in_one_word(self):
        assert bpe.split("a café") == ["a", " café"]

    @pytest.mark.parametrize(
        "text",
        [
            TEXT,
            "Don'T STOP, it's 1,234.56 \r\n\r\n   tabs\there\n",
            'She said: "Wow..." -- (really?) [x] a_b __c ___ 2nd',
            "日本語 🙂 naïve ٣٤ ١٢٣ end\n\n\n  ",
        ],
    )
    def test_matches_the_nanochat_pattern(self, text):
        regex = pytest.importorskip("regex")
        nanochat = (
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"""
            r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )
        assert bpe.split(text) == regex.findall(nanochat, text)


def test_merge_is_greedy_from_the_left():
    assert bpe.merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert bpe.merge([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]


class TestTrain:
    def test_merges_the_most_frequent_pair_first(self):
        tokenizer = bpe.train({"ab": 5, "cd": 3}, bpe.FIRST_MERGE_ID + 1)
        assert tokenizer.vocab[bpe.FIRST_MERGE_ID] == b"ab"

    def test_counts_are_weighted_by_chunk_frequency(self):
        tokenizer = bpe.train({"ab": 2, "cd": 100}, bpe.FIRST_MERGE_ID + 1)
        assert tokenizer.vocab[bpe.FIRST_MERGE_ID] == b"cd"

    def test_a_merged_token_can_merge_again(self):
        tokenizer = bpe.train({"abc": 5}, bpe.FIRST_MERGE_ID + 2)
        assert tokenizer.vocab[-1] == b"abc"

    def test_stops_when_nothing_is_left_to_merge(self):
        assert len(bpe.train({"ab": 1}, 400)) == bpe.FIRST_MERGE_ID + 1

    def test_matches_recounting_from_scratch(self):
        rng = np.random.default_rng(0)
        counts = {}
        for _ in range(200):
            word = "".join(rng.choice(list("aabbc"), size=rng.integers(1, 9)))
            counts[word] = int(rng.integers(1, 20))
        assert bpe.train(counts, 340).merges == slow_train(counts, 340)

    def test_rejects_a_vocabulary_without_all_the_bytes(self):
        with pytest.raises(ValueError):
            bpe.train({"ab": 1}, 100)


class TestTokenizer:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        return tokenizer_for(TEXT * 3)

    def test_round_trip(self, tokenizer):
        assert tokenizer.decode(tokenizer.encode(TEXT)) == TEXT

    def test_merges_shorten_the_text(self, tokenizer):
        assert len(tokenizer.encode(TEXT)) < len(TEXT.encode("utf-8")) / 2

    @pytest.mark.parametrize("text", ["zygomorphic", "日本語 🙂", "\t\n  "])
    def test_unseen_text_still_round_trips(self, tokenizer, text):
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_call_batches(self, tokenizer):
        result = tokenizer(["a", "b"])["input_ids"]
        assert result == [tokenizer.encode("a"), tokenizer.encode("b")]

    def test_eot_decodes_as_its_text(self, tokenizer):
        assert tokenizer.decode([bpe.EOT_ID]) == bpe.EOT_TEXT

    def test_bytes_sum_to_the_text_length(self, tokenizer):
        ids = tokenizer.encode(TEXT)
        assert tokenizer.bytes_per_token()[ids].sum() == len(TEXT.encode("utf-8"))

    def test_bytes_mark_only_eot(self, tokenizer):
        table = tokenizer.bytes_per_token()
        assert table[bpe.EOT_ID] == -1
        assert (table[1:] > 0).all()

    def test_save_and_load(self, tokenizer, tmp_path):
        path = tmp_path / "tokenizer.json"
        tokenizer.save(path)
        loaded = bpe.Tokenizer.load(path)
        assert loaded.encode(TEXT) == tokenizer.encode(TEXT)
        assert len(loaded) == len(tokenizer)
