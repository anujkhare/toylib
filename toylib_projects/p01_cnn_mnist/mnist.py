"""MNIST as NumPy arrays.

::

    images, labels = mnist.load("train")   # or "test"

    images: uint8, shape (N, 28, 28, 1), values 0-255
            (N = 60,000 for "train", 10,000 for "test")
    labels: int32, shape (N,), values 0-9

The original IDX files are downloaded once from a mirror and cached under
``~/.cache/toylib/mnist`` (override the root with ``$TOYLIB_CACHE``), so later
runs work offline.
"""

import gzip
import pathlib

import numpy as np

from toylib.data import download

# Google's CVDF mirror of the files on yann.lecun.com, which is often unavailable.
_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}

CLASS_NAMES = [str(i) for i in range(10)]


def parse_idx(path: pathlib.Path) -> np.ndarray:
    """Read a gzipped IDX file (the format MNIST ships in) into an array.

    Header: two zero bytes, a dtype code (0x08 = uint8), the number of
    dimensions, then each dimension as a big-endian int32. The data follows.
    """
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if raw[:2] != b"\x00\x00" or raw[2] != 0x08:
        raise ValueError(f"{path} is not a uint8 IDX file")
    ndim = raw[3]
    shape = tuple(np.frombuffer(raw, dtype=">i4", count=ndim, offset=4))
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    return data.reshape(shape)


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)`` for ``split`` ("train" or "test")."""
    if split not in _FILES:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    root = download.cache_dir("mnist")
    image_file, label_file = _FILES[split]
    images = parse_idx(download.fetch(_BASE_URL + image_file, root / image_file))
    labels = parse_idx(download.fetch(_BASE_URL + label_file, root / label_file))
    return images[..., None], labels.astype(np.int32)
