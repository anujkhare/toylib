"""MNIST as NumPy arrays. Post 1, milestone 1: yours to implement.

The rest of the repo expects this contract::

    images, labels = mnist.load("train")   # or "test"

    images: uint8, shape (N, 28, 28, 1), values 0-255
            (N = 60,000 for "train", 10,000 for "test")
    labels: int32, shape (N,), values 0-9

Cache the download under ``~/.cache/toylib/mnist`` so later runs work offline.
Hugging Face ``datasets`` is installed with the ``data`` extra and is one way
to get the files; any source works.
"""

import numpy as np


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "Post 1, milestone 1: see toylib_posts/p01_cnn_mnist/README.md"
    )
