"""Tests for sft_data.py."""

import random

import numpy as np
import pytest

from toylib_projects.p02_gpt_web import sft_data


def test_gsm8k_calls_become_python_parts():
    parts = sft_data.gsm8k_parts("She earns 12/60 = $<<12/60=0.2>>0.2 a minute.\n#### 10")
    assert parts == [
        {"type": "text", "text": "She earns 12/60 = $"},
        {"type": "python", "text": "12/60"},
        {"type": "python_output", "text": "0.2"},
        {"type": "text", "text": "0.2 a minute.\n#### 10"},
    ]


def test_mixture_order_matches_nanochats_task_mixture():
    order = sft_data.mixture_order([3, 2])
    expected = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    random.Random(42).shuffle(expected)
    assert order == expected


def _conversations(lengths, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for n in lengths:
        ids = np.concatenate([[0], rng.integers(1, 100, n - 1)]).astype(np.uint16)
        out.append((ids, rng.integers(0, 2, n).astype(np.uint8)))
    return out


def test_packing_keeps_every_conversation_whole():
    convs = _conversations(np.random.default_rng(1).integers(2, 17, 200))
    rows, masks = sft_data.pack(iter(convs), row_len=17, buffer_size=10)
    assert rows.shape == masks.shape and rows.shape[1] == 17
    # Each conversation starts with 0 (BOS) and has no other 0s, so rows split back into them.
    found = []
    for row, mask in zip(rows, masks):
        assert row[0] == 0  # Rows start a conversation.
        starts = list(np.flatnonzero(row == 0))
        content = len(row)
        pads = [s for s in starts if s > 0 and (row[s:] == 0).all()]
        if pads:
            content = pads[0]
            assert (mask[content:] == 0).all()
        bounds = [s for s in starts if s < content] + [content]
        found += [tuple(row[a:b]) for a, b in zip(bounds, bounds[1:])]
    assert sorted(found) == sorted(tuple(ids) for ids, _ in convs)


def test_packing_takes_the_longest_that_fits():
    convs = _conversations([6, 3, 5, 4])
    rows, _ = sft_data.pack(iter(convs), row_len=10, buffer_size=4)
    # Row 1: 6, then the longest of 3/5/4 that fits in 4 is 4. Row 2: 5 then 3.
    assert (rows[0][:6] == convs[0][0]).all() and (rows[0][6:] == convs[3][0]).all()
    assert (rows[1][:5] == convs[2][0]).all() and (rows[1][5:8] == convs[1][0]).all()
    assert (rows[1][8:] == 0).all()


def _write(tmp_path, rows):
    tokens = np.arange(rows * 5, dtype=np.uint16).reshape(rows, 5)
    mask = (tokens % 2).astype(np.uint8)
    np.save(tmp_path / "t.npy", tokens)
    np.save(tmp_path / "m.npy", mask)
    return tokens, mask


def test_dataset_batches_rows_in_order_and_resumes(tmp_path):
    tokens, mask = _write(tmp_path, rows=7)
    ds = sft_data.PackedRowsDataset(str(tmp_path / "t.npy"), str(tmp_path / "m.npy"), batch_size=3)
    assert ds.steps_per_epoch == 2
    it = iter(ds)
    first, second, third = next(it), next(it), next(it)
    np.testing.assert_array_equal(first["inputs"], tokens[:3, :-1])
    np.testing.assert_array_equal(first["targets"], tokens[:3, 1:])
    np.testing.assert_array_equal(first["mask"], mask[:3, 1:])
    np.testing.assert_array_equal(second["inputs"], tokens[3:6, :-1])
    np.testing.assert_array_equal(third["inputs"], first["inputs"])  # A new epoch.
    again = sft_data.PackedRowsDataset(str(tmp_path / "t.npy"), str(tmp_path / "m.npy"), batch_size=3)
    again.restore_state(ds.get_state())
    np.testing.assert_array_equal(next(iter(again))["inputs"], next(it)["inputs"])


def test_repeatable_dataset_restarts(tmp_path):
    _write(tmp_path, rows=6)
    ds = sft_data.PackedRowsDataset(str(tmp_path / "t.npy"), str(tmp_path / "m.npy"), batch_size=2, repeatable=True)
    a = [b["inputs"] for _, b in zip(range(2), ds)]
    b = [b["inputs"] for _, b in zip(range(2), ds)]
    np.testing.assert_array_equal(a, b)


def test_dataset_needs_a_full_batch(tmp_path):
    _write(tmp_path, rows=2)
    with pytest.raises(ValueError, match="fewer than a batch"):
        sft_data.PackedRowsDataset(str(tmp_path / "t.npy"), str(tmp_path / "m.npy"), batch_size=3)
