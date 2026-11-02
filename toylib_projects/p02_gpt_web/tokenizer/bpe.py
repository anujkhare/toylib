"""Byte Pair Encoding: a tokenizer trained on the text it will read.

The split pattern comes from nanochat (https://github.com/karpathy/nanochat), which
trains with its compiled `rustbpe` and encodes with `tiktoken`. The Python here is
adapted from minbpe (https://github.com/karpathy/minbpe), mainly `regex.py` and
`base.py`. Both are MIT licensed, Copyright (c) Andrej Karpathy. Changes from them:

* Training counts each distinct chunk once, updates pair counts as it merges and
  keeps them in a heap, since rescanning 2 GB of text for every merge would take days.
* Only the standard library's `re`, so there is nothing to install.
* Id 0 is <|endoftext|>, so byte `b` has id `b + 1`.

Every byte has an id, so any text can be encoded and there is no unknown token: a
word the tokenizer has never seen is spelled in smaller pieces.
"""

import collections
import heapq
import json
import pathlib
import re
import typing

import numpy as np

EOT_ID = 0  # Starts every story.
EOT_TEXT = "<|endoftext|>"
FIRST_MERGE_ID = 257  # After <|endoftext|> and the 256 bytes.

# nanochat's split, so merges never cross from one word to the next. Numbers go in
# ones and twos, so a small vocabulary doesn't spend tokens on them. Without `\p{L}`,
# letters are `[^\W\d_]`, and "neither letter nor number" has to name `_` itself. The
# one difference: numerals that aren't digits, such as ½, count as letters here.
SPLIT_PATTERN = re.compile(
    r"""'(?i:[sdmt]|ll|ve|re)"""
    r"""|(?:[^\r\n\w]|_)?[^\W\d_]+"""
    r"""|\d{1,2}"""
    r"""| ?(?:[^\s\w]|_)+[\r\n]*"""
    r"""|\s*[\r\n]"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)

Pair = tuple[int, int]


def split(text: str) -> list[str]:
    """Split text into the chunks that are tokenized separately."""
    return SPLIT_PATTERN.findall(text)


def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace each occurrence of `pair` in `ids` with `new_id`."""
    merged, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class Tokenizer:
    """Follows the parts of the Hugging Face interface the project uses: calling it on
    a list of strings returns `{"input_ids": ...}`, and `decode` returns text.
    """

    bos_token_id = EOT_ID
    eos_token_id = EOT_ID

    def __init__(self, merges: typing.Iterable[Pair]):
        self.merges = [(int(a), int(b)) for a, b in merges]
        self.ranks = {pair: FIRST_MERGE_ID + i for i, pair in enumerate(self.merges)}
        self.vocab = [EOT_TEXT.encode("utf-8")] + [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._cache: dict[str, list[int]] = {}  # Stories reuse few distinct chunks.

    def __len__(self) -> int:
        return len(self.vocab)

    def __call__(self, texts: list[str], **_) -> dict[str, list[list[int]]]:
        return {"input_ids": [self.encode(text) for text in texts]}

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = [b + 1 for b in chunk.encode("utf-8")]
        while len(ids) >= 2:
            # Merges apply in the order they were learned.
            pair = min(
                zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf"))
            )
            if pair not in self.ranks:
                break
            ids = merge(ids, pair, self.ranks[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        ids = []
        for chunk in split(text):
            if chunk not in self._cache:
                self._cache[chunk] = self._encode_chunk(chunk)
            ids.extend(self._cache[chunk])
        return ids

    def decode(self, ids: typing.Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_strings(self, ids: typing.Iterable[int]) -> list[str]:
        """Return each token's text, for labelling figures."""
        return [self.decode([i]) for i in ids]

    def bytes_per_token(self) -> np.ndarray:
        """Return UTF-8 bytes per id; -1 marks <|endoftext|>, which bits per byte skips."""
        table = np.array([len(token) for token in self.vocab])
        table[EOT_ID] = -1
        return table

    def save(self, path: pathlib.Path) -> None:
        path.write_text(json.dumps({"merges": self.merges}))

    @classmethod
    def load(cls, path: pathlib.Path) -> "Tokenizer":
        return cls(json.loads(path.read_text())["merges"])


def train(chunk_counts: typing.Mapping[str, int], vocab_size: int) -> Tokenizer:
    """Learn merges from how often each chunk appears, most frequent pair first.

    Stops early if every chunk has merged into one token, so the result can be smaller
    than `vocab_size`. Ties go to the larger pair, which makes training deterministic.
    """
    if vocab_size < FIRST_MERGE_ID:
        raise ValueError(f"vocab_size must be at least {FIRST_MERGE_ID}: {vocab_size}")

    words = [[b + 1 for b in chunk.encode("utf-8")] for chunk in chunk_counts]
    weights = list(chunk_counts.values())
    pair_counts: collections.Counter[Pair] = collections.Counter()
    pair_words: dict[Pair, set[int]] = collections.defaultdict(set)
    for index, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += weights[index]
            pair_words[pair].add(index)

    def heap_entry(pair: Pair) -> tuple[int, int, int]:
        return (-pair_counts[pair], -pair[0], -pair[1])  # Pops the largest (count, pair).

    heap = [heap_entry(pair) for pair in pair_counts]
    heapq.heapify(heap)

    merges = []
    for new_id in range(FIRST_MERGE_ID, vocab_size):
        best = None
        while heap and best is None:
            count, a, b = heapq.heappop(heap)
            if pair_counts.get((-a, -b)) == -count:  # Otherwise it's out of date.
                best = (-a, -b)
        if best is None:
            break
        merges.append(best)
        # Only the words holding this pair change, so only their pairs are recounted.
        changed = set()
        for index in pair_words.pop(best):
            for pair in zip(words[index], words[index][1:]):
                pair_counts[pair] -= weights[index]
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                changed.add(pair)
            words[index] = merge(words[index], best, new_id)
            for pair in zip(words[index], words[index][1:]):
                pair_counts[pair] += weights[index]
                pair_words[pair].add(index)
                changed.add(pair)
        for pair in changed:
            if pair in pair_counts:
                heapq.heappush(heap, heap_entry(pair))
    return Tokenizer(merges)
