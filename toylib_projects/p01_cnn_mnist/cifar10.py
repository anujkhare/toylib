"""CIFAR-10 as NumPy arrays.

::

    images, labels = cifar10.load("train")   # or "test"

    images: uint8, shape (N, 32, 32, 3), values 0-255, RGB
            (N = 50,000 for "train", 10,000 for "test")
    labels: int32, shape (N,), values 0-9 (names in ``CLASS_NAMES``)

The Python-pickle archive (about 160 MB) is downloaded once from the dataset's
home page and cached under ``~/.cache/toylib/cifar10`` (override the root with
``$TOYLIB_CACHE``), so later runs work offline.
"""

import pickle
import tarfile

import numpy as np

from toylib.data import download

_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_ARCHIVE = "cifar-10-python.tar.gz"
_BATCHES = {
    "train": [f"data_batch_{i}" for i in range(1, 6)],
    "test": ["test_batch"],
}

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_batch(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode one pickled batch into NHWC images and labels.

    Each batch is a dict with ``b"data"``, a (N, 3072) uint8 array laid out as
    1024 red, then 1024 green, then 1024 blue values per image, and
    ``b"labels"``, a list of N ints.
    """
    batch = pickle.loads(raw, encoding="bytes")
    data = np.asarray(batch[b"data"], dtype=np.uint8)
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"], dtype=np.int32)
    return images, labels


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)`` for ``split`` ("train" or "test")."""
    if split not in _BATCHES:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    root = download.cache_dir("cifar10")
    cached = root / f"{split}.npz"
    if not cached.exists():
        archive = download.fetch(_URL, root / _ARCHIVE)
        parts = []
        with tarfile.open(archive, "r:gz") as tar:
            for name in _BATCHES[split]:
                member = tar.extractfile(f"cifar-10-batches-py/{name}")
                parts.append(parse_batch(member.read()))
        images = np.concatenate([p[0] for p in parts])
        labels = np.concatenate([p[1] for p in parts])
        # Unpacking the archive takes seconds; the arrays load in milliseconds.
        np.savez(cached, images=images, labels=labels)
    with np.load(cached) as f:
        return f["images"], f["labels"]
