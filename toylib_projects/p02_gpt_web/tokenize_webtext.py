"""ClimbMix: the web text nanochat trains on, stored as flat files of token ids.

The parquet shards are small enough to fetch but too big to tokenize at training time
in Python, so `prepare` writes these, and training reads them with `TokenBinDataset`:

    <data_dir>/tokenizer_<N>.json  the merges of a tokenizer with N tokens
    <data_dir>/train_<N>.bin       token ids as uint16, each document after <|endoftext|>
    <data_dir>/val_<N>.bin

As in nanochat, the last shard is the validation set. Shards hold about 250M characters.

Run once with:
    uv run python -m toylib_projects.p02_gpt_web.tokenize_webtext --num-train-shards 8
"""

import argparse
import collections
import functools
import multiprocessing
import os
import pathlib
import typing

import numpy as np
import pyarrow.parquet as pq

from toylib_projects.p02_gpt_web.scripts import download_data
from toylib_projects.p02_gpt_web.tokenizer import bpe

DATA_DIR = pathlib.Path(__file__).parent / "data" / "climbmix"
EOT_ID = bpe.EOT_ID
VAL_SHARD = download_data.MAX_SHARD
NUM_TOKENIZER_SHARDS = 8  # About 2B characters, as nanochat uses.
MIN_CHUNK_COUNT = 2  # Half the distinct chunks appear once, and hardly affect merges.


def data_dir_for(num_train_shards: int) -> pathlib.Path:
    return DATA_DIR / f"shards{num_train_shards}"


def bin_path(split: str, vocab_size: int, data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / f"{split}_{vocab_size}.bin"


def tokenizer_path(vocab_size: int, data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / f"tokenizer_{vocab_size}.json"


def load_tokenizer(vocab_size: int, data_dir: pathlib.Path) -> bpe.Tokenizer:
    return bpe.Tokenizer.load(tokenizer_path(vocab_size, data_dir))


def shard_path(index: int, data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / "shards" / download_data.shard_filename(index)


def fetch_shards(
    indices: list[int], data_dir: pathlib.Path, workers: int = 8
) -> list[pathlib.Path]:
    """Download consecutive shards that are missing, and return their paths."""
    paths = [shard_path(i, data_dir) for i in indices]
    download_data.download(len(indices), paths[0].parent, workers, indices[0])
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} shards failed to download: {missing[:5]}")
    return paths


def read_texts(path: pathlib.Path) -> typing.Iterator[list[str]]:
    """Yield lists of documents from a shard, one row group at a time."""
    file = pq.ParquetFile(path)
    for index in range(file.num_row_groups):
        yield file.read_row_group(index, columns=["text"]).column("text").to_pylist()


def count_chunks(
    paths: list[pathlib.Path], min_count: int = MIN_CHUNK_COUNT
) -> dict[str, int]:
    """Count the chunks of some shards, keeping those seen at least `min_count` times."""
    counts: collections.Counter[str] = collections.Counter()
    for path in paths:
        for texts in read_texts(path):
            for text in texts:
                counts.update(bpe.split(text))
    return {chunk: count for chunk, count in counts.items() if count >= min_count}


def encode_shard(path: pathlib.Path, tokenizer_file: pathlib.Path) -> np.ndarray:
    """Return a shard as one stream of token ids."""
    tokenizer = _load_tokenizer(tokenizer_file)
    ids = []
    for texts in read_texts(path):
        for text in texts:
            ids.append(EOT_ID)
            ids.extend(tokenizer.encode(text))
    return np.array(ids, dtype=np.uint16)


@functools.lru_cache(maxsize=1)
def _load_tokenizer(path: pathlib.Path) -> bpe.Tokenizer:
    """Keep one tokenizer per process, so its cache of chunks carries across shards."""
    return bpe.Tokenizer.load(path)


def write_tokens(
    paths: list[pathlib.Path],
    out_path: pathlib.Path,
    tokenizer_file: pathlib.Path,
    workers: int,
) -> int:
    """Write shards as token ids in order, and return the token count."""
    partial = out_path.with_suffix(".partial")
    encode = functools.partial(encode_shard, tokenizer_file=tokenizer_file)
    total = 0
    # Spawned, because forking after JAX has started can hang.
    with multiprocessing.get_context("spawn").Pool(workers) as pool, open(partial, "wb") as out:
        for done, ids in enumerate(pool.imap(encode, paths), start=1):
            ids.tofile(out)
            total += len(ids)
            print(f"Tokenized {done} of {len(paths)} shards: {total:,} tokens")
    partial.rename(out_path)
    return total


def prepare(
    vocab_size: int,
    num_train_shards: int,
    data_dir: pathlib.Path,
    num_tokenizer_shards: int = NUM_TOKENIZER_SHARDS,
    workers: int | None = None,
) -> None:
    """Download shards, train a tokenizer and write the token files.

    Skips work already done. `vocab_size` must fit in uint16.
    """
    if vocab_size > 2**16:
        raise ValueError(f"vocab_size must fit in uint16: {vocab_size}")
    if not 0 < num_train_shards <= VAL_SHARD:
        raise ValueError(f"num_train_shards must be in 1..{VAL_SHARD}: {num_train_shards}")
    workers = workers or os.cpu_count() or 1
    data_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_file = tokenizer_path(vocab_size, data_dir)
    train_file = bin_path("train", vocab_size, data_dir)
    val_file = bin_path("val", vocab_size, data_dir)
    if tokenizer_file.exists() and train_file.exists() and val_file.exists():
        return

    train_shards = fetch_shards(list(range(num_train_shards)), data_dir)
    val_shards = fetch_shards([VAL_SHARD], data_dir)

    if not tokenizer_file.exists():
        print("Training the tokenizer")
        counts = count_chunks(train_shards[:num_tokenizer_shards])
        print(f"{len(counts):,} distinct chunks")
        bpe.train(counts, vocab_size).save(tokenizer_file)

    for shards, out_path in ((val_shards, val_file), (train_shards, train_file)):
        if not out_path.exists():
            total = write_tokens(shards, out_path, tokenizer_file, workers)
            print(f"Wrote {out_path}: {total:,} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--num-train-shards", type=int, default=8)
    args = parser.parse_args()
    prepare(args.vocab_size, args.num_train_shards, data_dir_for(args.num_train_shards))
