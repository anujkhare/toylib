"""Tests for tinystories.py."""

import numpy as np

from toylib_projects.p02_tinystories import tinystories


class TestReadStories:
    def test_splits_across_chunk_boundaries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tinystories, "CHUNK_CHARS", 7)
        path = tmp_path / "stories.txt"
        path.write_text("first story\n<|endoftext|>\nsecond\n<|endoftext|>third")
        stories = [s for chunk in tinystories.read_stories(path) for s in chunk]
        assert stories == ["first story", "second", "third"]

    def test_drops_empty_stories(self, tmp_path):
        path = tmp_path / "stories.txt"
        path.write_text("one<|endoftext|>\n\n<|endoftext|>two<|endoftext|>")
        stories = [s for chunk in tinystories.read_stories(path) for s in chunk]
        assert stories == ["one", "two"]


def test_count_chunks(tmp_path):
    path = tmp_path / "stories.txt"
    path.write_text("the cat the<|endoftext|>the dog<|endoftext|>")
    counts = tinystories.count_chunks(path)
    assert counts == {"the": 2, " cat": 1, " the": 1, " dog": 1}


def test_prepare_writes_token_files(tmp_path):
    for split in ("train", "val"):
        text = "The cat sat.<|endoftext|>The dog ran.<|endoftext|>"
        tinystories.text_path(split, tmp_path).write_text(text)

    tinystories.prepare(vocab_size=262, data_dir=tmp_path)

    tokens = np.fromfile(tinystories.bin_path("train", 262, tmp_path), dtype=np.uint16)
    tokenizer = tinystories.load_tokenizer(262, tmp_path)
    assert len(tokenizer) == 262
    assert tokens[0] == tinystories.EOT_ID
    assert tokenizer.decode(tokens.tolist()) == (
        "<|endoftext|>The cat sat.<|endoftext|>The dog ran."
    )


def test_prepare_skips_finished_work(tmp_path, monkeypatch):
    for split in ("train", "val"):
        tinystories.text_path(split, tmp_path).write_text("A cat.<|endoftext|>")
    tinystories.prepare(vocab_size=258, data_dir=tmp_path)
    monkeypatch.setattr(tinystories, "count_chunks", None)  # Would crash if called.
    tinystories.prepare(vocab_size=258, data_dir=tmp_path)
