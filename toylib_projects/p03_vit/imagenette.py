"""Imagenette as NumPy arrays: ten easy classes of ImageNet, at higher resolution.

::

    images, labels = imagenette.load("train")   # or "test"

    images: uint8, shape (N, 200, 200, 3), values 0-255, RGB
            (N = 9,469 for "train", 3,925 for "test")
    labels: int32, shape (N,), values 0-9 (names in ``CLASS_NAMES``)

Photos vary in shape, so each is cropped to a centred square and resized to
``STORED_SIZE``, which leaves room for random crops during training. Test images
are shuffled with a fixed seed: the harness drops a partial last batch, and in
folder order that would remove one class only.

The archive (about 340 MB) is downloaded once from fast.ai's mirror. It and the
decoded arrays are cached under ``~/.cache/toylib/imagenette`` (override the root
with ``$TOYLIB_CACHE``), so later runs work offline.
"""

import io
import tarfile

import numpy as np
from PIL import Image
from PIL import ImageOps

from toylib.data import download

_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
_ARCHIVE = "imagenette2-320.tgz"
_SPLIT_FOLDERS = {"train": "train", "test": "val"}

# ImageNet ids, sorted, so a label indexes both lists.
_WNIDS = [
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
]

CLASS_NAMES = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]

STORED_SIZE = 200


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        square = ImageOps.fit(
            image.convert("RGB"),
            (STORED_SIZE, STORED_SIZE),
            method=Image.Resampling.BILINEAR,
        )
        return np.asarray(square, dtype=np.uint8)


def _read_archive(path, split: str) -> tuple[np.ndarray, np.ndarray]:
    folder = _SPLIT_FOLDERS[split]
    images, labels = [], []
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/")  # imagenette2-320/<folder>/<wnid>/<file>
            if not member.isfile() or len(parts) != 4 or parts[1] != folder:
                continue
            images.append(_decode(tar.extractfile(member).read()))
            labels.append(_WNIDS.index(parts[2]))
    order = np.random.default_rng(0).permutation(len(images))
    return np.stack(images)[order], np.asarray(labels, dtype=np.int32)[order]


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)`` for ``"train"`` or ``"test"``."""
    if split not in _SPLIT_FOLDERS:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    root = download.cache_dir("imagenette")
    cached = root / f"{split}_{STORED_SIZE}.npz"
    if not cached.exists():
        archive = download.fetch(_URL, root / _ARCHIVE)
        images, labels = _read_archive(archive, split)
        np.savez(cached, images=images, labels=labels)
    with np.load(cached) as arrays:
        return arrays["images"], arrays["labels"]
