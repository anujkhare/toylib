"""Tests for ArrayDataset."""

import numpy as np
import pytest

from toylib.harness.data import ArrayDataset


def _arrays(n: int = 10) -> dict[str, np.ndarray]:
    return {"x": np.arange(n * 2).reshape(n, 2), "y": np.arange(n)}


def _epoch(ds: ArrayDataset) -> list[np.ndarray]:
    return [batch["y"] for batch in ds]


def test_batches_have_fixed_shape_and_drop_remainder():
    ds = ArrayDataset(_arrays(10), batch_size=3, shuffle=False)
    batches = list(ds)
    assert len(batches) == 3 == ds.batches_per_epoch
    for b in batches:
        assert b["x"].shape == (3, 2) and b["y"].shape == (3,)
        np.testing.assert_array_equal(b["x"][:, 0] // 2, b["y"])  # rows stay aligned


def test_each_epoch_is_a_new_permutation_and_reproducible():
    a = ArrayDataset(_arrays(10), batch_size=5, seed=0)
    b = ArrayDataset(_arrays(10), batch_size=5, seed=0)
    first, second = np.concatenate(_epoch(a)), np.concatenate(_epoch(a))
    assert sorted(first) == list(range(10))
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first, np.concatenate(_epoch(b)))


def test_break_resumes_at_next_batch():
    ds = ArrayDataset(_arrays(8), batch_size=2, shuffle=False)
    for i, batch in enumerate(ds):
        if i == 1:
            break
    assert ds.get_state() == {"epoch": 0, "position": 2}
    np.testing.assert_array_equal(next(iter(ds))["y"], [4, 5])


def test_fully_consumed_epoch_rolls_over_on_next_iter():
    """Stopping right after the last batch must not yield an empty pass next time."""
    ds = ArrayDataset(_arrays(4), batch_size=2, shuffle=False)
    it = iter(ds)
    next(it), next(it)  # consume both batches, never exhaust the generator
    assert ds.get_state() == {"epoch": 0, "position": 2}
    assert len(_epoch(ds)) == 2
    assert ds.get_state() == {"epoch": 2, "position": 0}


def test_restore_state_continues_same_order():
    ds = ArrayDataset(_arrays(12), batch_size=3, seed=5)
    it = iter(ds)
    next(it)
    state = ds.get_state()
    rest = [b["y"] for b in it]

    fresh = ArrayDataset(_arrays(12), batch_size=3, seed=5)
    fresh.restore_state(state)
    np.testing.assert_array_equal(np.concatenate(_epoch(fresh)), np.concatenate(rest))


def test_restart_each_iter_for_eval():
    ds = ArrayDataset(_arrays(8), batch_size=2, shuffle=False, restart_each_iter=True)
    next(iter(ds))
    np.testing.assert_array_equal(next(iter(ds))["y"], [0, 1])


def test_validation():
    with pytest.raises(ValueError, match="same length"):
        ArrayDataset({"x": np.zeros(3), "y": np.zeros(4)}, batch_size=1)
    with pytest.raises(ValueError, match="larger than the dataset"):
        ArrayDataset(_arrays(2), batch_size=3)
    with pytest.raises(ValueError, match="at least one"):
        ArrayDataset({}, batch_size=1)
