"""TinyStories: download, train a tokenizer, and store as flat files of token ids.

TinyStories uses few of GPT-2's 50,257 tokens, and in an untied model each unused one
costs two rows of `qkv_dim` parameters and a slower softmax. So we train a tokenizer
on the stories themselves. `prepare` writes:

    data/tokenizer_<N>.json  the merges of a tokenizer with N tokens
    data/train_<N>.bin       token ids as uint16, each story preceded by <|endoftext|>
    data/val_<N>.bin

Run once with:
    uv run python -m toylib_projects.p02_tinystories.tinystories --vocab-size 8192
"""

import argparse
import collections
import pathlib
import typing

import numpy as np

from toylib.data import download
from toylib_projects.p02_tinystories.tokenizer import bpe

DATA_DIR = pathlib.Path(__file__).parent / "data"
URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-{split}.txt"
)
SPLIT_FILES = {"train": "train", "val": "valid"}
DELIMITER = bpe.EOT_TEXT
EOT_ID = bpe.EOT_ID
CHUNK_CHARS = 64_000_000


def read_stories(path: pathlib.Path) -> typing.Iterator[list[str]]:
    """Yield lists of stories from a delimited text file, without loading all of it."""
    leftover = ""
    with open(path, encoding="utf-8") as f:
        while chunk := f.read(CHUNK_CHARS):
            parts = (leftover + chunk).split(DELIMITER)
            leftover = parts.pop()
            yield [s for p in parts if (s := p.strip())]
    if leftover.strip():
        yield [leftover.strip()]


def count_chunks(path: pathlib.Path) -> collections.Counter[str]:
    """Count the chunks of a story file, which is all that training a tokenizer needs."""
    counts: collections.Counter[str] = collections.Counter()
    for stories in read_stories(path):
        for story in stories:
            counts.update(bpe.split(story))
    return counts


def write_tokens(
    path: pathlib.Path, out_path: pathlib.Path, tokenizer: bpe.Tokenizer
) -> int:
    """Write a story file as token ids, one continuous stream; return the token count."""
    total = 0
    with open(out_path, "wb") as out:
        for stories in read_stories(path):
            ids = []
            for story in stories:
                ids.append(EOT_ID)
                ids.extend(tokenizer.encode(story))
            np.array(ids, dtype=np.uint16).tofile(out)
            total += len(ids)
    return total


def text_path(split: str, data_dir: pathlib.Path = DATA_DIR) -> pathlib.Path:
    return data_dir / f"TinyStoriesV2-GPT4-{SPLIT_FILES[split]}.txt"


def bin_path(split: str, vocab_size: int, data_dir: pathlib.Path = DATA_DIR) -> pathlib.Path:
    return data_dir / f"{split}_{vocab_size}.bin"


def tokenizer_path(vocab_size: int, data_dir: pathlib.Path = DATA_DIR) -> pathlib.Path:
    return data_dir / f"tokenizer_{vocab_size}.json"


def load_tokenizer(vocab_size: int, data_dir: pathlib.Path = DATA_DIR) -> bpe.Tokenizer:
    return bpe.Tokenizer.load(tokenizer_path(vocab_size, data_dir))


def prepare(vocab_size: int, data_dir: pathlib.Path = DATA_DIR) -> None:
    """Download TinyStories, train a tokenizer and write the token files.

    Skips work already done.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, name in SPLIT_FILES.items():
        download.fetch(URL.format(split=name), text_path(split, data_dir))

    path = tokenizer_path(vocab_size, data_dir)
    if not path.exists():
        print("Training the tokenizer")
        counts = count_chunks(text_path("train", data_dir))
        print(f"{len(counts):,} distinct chunks")
        bpe.train(counts, vocab_size).save(path)
    tokenizer = bpe.Tokenizer.load(path)

    for split in SPLIT_FILES:
        out_path = bin_path(split, vocab_size, data_dir)
        if not out_path.exists():
            total = write_tokens(text_path(split, data_dir), out_path, tokenizer)
            print(f"Wrote {out_path}: {total:,} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=8192)
    prepare(parser.parse_args().vocab_size)
