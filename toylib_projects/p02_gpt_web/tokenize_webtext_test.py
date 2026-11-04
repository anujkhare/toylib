"""Tests for tokenize_webtext.py."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toylib_projects.p02_gpt_web import tokenize_webtext
from toylib_projects.p02_gpt_web.tokenizer import bpe

DOCS = {
    0: ["the cat sat on the mat", "the dog sat on the log"],
    1: ["the cat saw the dog", "a mat and a log"],
    tokenize_webtext.VAL_SHARD: ["the cat sat"],
}


def write_shard(index: int, data_dir) -> None:
    path = tokenize_webtext.shard_path(index, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # One document per row group, to exercise reading more than one group.
    pq.write_table(pa.table({"text": DOCS[index]}), path, row_group_size=1)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    for index in DOCS:
        write_shard(index, tmp_path)
    monkeypatch.setattr(tokenize_webtext.download_data, "download", lambda *args: None)
    return tmp_path


def test_read_texts_visits_every_row_group(data_dir):
    groups = list(tokenize_webtext.read_texts(tokenize_webtext.shard_path(0, data_dir)))
    assert groups == [[DOCS[0][0]], [DOCS[0][1]]]


def test_fetch_shards_reports_failures(data_dir):
    with pytest.raises(RuntimeError, match="1 shards failed"):
        tokenize_webtext.fetch_shards([1, 2], data_dir)


def test_count_chunks_drops_rare_chunks(data_dir):
    counts = tokenize_webtext.count_chunks([tokenize_webtext.shard_path(0, data_dir)], min_count=2)
    assert counts[" the"] == 2
    assert " cat" not in counts


def test_write_tokens_keeps_documents_in_order(data_dir):
    tokenizer_file = data_dir / "tokenizer.json"
    bpe.train({"the": 5}, 300).save(tokenizer_file)
    paths = [tokenize_webtext.shard_path(i, data_dir) for i in (0, 1)]
    out = data_dir / "out.bin"
    total = tokenize_webtext.write_tokens(paths, out, tokenizer_file, workers=2)

    ids = np.fromfile(out, dtype=np.uint16)
    assert len(ids) == total
    text = bpe.Tokenizer.load(tokenizer_file).decode(ids.tolist())
    expected = "".join(bpe.EOT_TEXT + doc for i in (0, 1) for doc in DOCS[i])
    assert text == expected


def test_prepare_writes_files_and_skips_finished_work(data_dir, monkeypatch):
    tokenize_webtext.prepare(262, num_train_shards=2, data_dir=data_dir, workers=1)
    train = np.fromfile(tokenize_webtext.bin_path("train", 262, data_dir), dtype=np.uint16)
    val = np.fromfile(tokenize_webtext.bin_path("val", 262, data_dir), dtype=np.uint16)
    assert (train == bpe.EOT_ID).sum() == 4
    assert (val == bpe.EOT_ID).sum() == 1
    assert len(tokenize_webtext.load_tokenizer(262, data_dir)) == 262

    monkeypatch.setattr(tokenize_webtext, "fetch_shards", lambda *args: pytest.fail("Refetched"))
    tokenize_webtext.prepare(262, num_train_shards=2, data_dir=data_dir, workers=1)


@pytest.mark.parametrize(
    ("vocab_size", "shards"),
    [(2**16 + 1, 1), (300, 0), (300, tokenize_webtext.VAL_SHARD + 1)],
)
def test_prepare_rejects_bad_settings(vocab_size, shards, tmp_path):
    with pytest.raises(ValueError):
        tokenize_webtext.prepare(vocab_size, shards, tmp_path)
