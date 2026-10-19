"""Datasets as NumPy arrays, ready for ``toylib.harness.data.ArrayDataset``.

Each module downloads once, caches under ``~/.cache/toylib/``, and returns
plain arrays, so it depends on nothing else in the repo.
"""
