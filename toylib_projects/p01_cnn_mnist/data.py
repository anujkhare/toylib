"""MNIST and CIFAR-10 as harness datasets.

``mnist`` and ``cifar10`` return raw uint8 arrays; this module turns them into what the
model trains on (float32, standardized per channel, NHWC) and wraps them in
``ArrayDataset`` so ``Experiment`` can iterate them. Each batch is a dict::

    {"image": float32 (B, H, W, C), "label": int32 (B,)}
"""

import dataclasses

import numpy as np

from toylib_projects.p01_cnn_mnist import cifar10, mnist
from toylib.harness.data import ArrayDataset


@dataclasses.dataclass(frozen=True)
class DatasetInfo:
    load: object  # split -> (uint8 images, int32 labels)
    image_shape: tuple[int, int, int]  # (H, W, C)
    class_names: list[str]
    # Per-channel statistics of the training images, after scaling to [0, 1].
    mean: tuple[float, ...]
    std: tuple[float, ...]


DATASETS = {
    "mnist": DatasetInfo(
        load=mnist.load,
        image_shape=(28, 28, 1),
        class_names=mnist.CLASS_NAMES,
        mean=(0.1307,),
        std=(0.3081,),
    ),
    "cifar10": DatasetInfo(
        load=cifar10.load,
        image_shape=(32, 32, 3),
        class_names=cifar10.CLASS_NAMES,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
}


def preprocess(images: np.ndarray, info: DatasetInfo) -> np.ndarray:
    """uint8 [0, 255] -> float32, zero mean and unit variance per channel."""
    x = images.astype(np.float32) / 255.0
    return (x - np.asarray(info.mean, np.float32)) / np.asarray(info.std, np.float32)


def unpreprocess(images: np.ndarray, info: DatasetInfo) -> np.ndarray:
    """Undo ``preprocess`` back to floats in [0, 1], for plotting."""
    x = np.asarray(images) * np.asarray(info.std, np.float32) + np.asarray(info.mean, np.float32)
    return np.clip(x, 0.0, 1.0)


def load_arrays(name: str, split: str, limit: int | None = None) -> dict[str, np.ndarray]:
    """Preprocessed ``{"image", "label"}`` arrays for one split, optionally the first ``limit``."""
    info = DATASETS[name]
    images, labels = info.load(split)
    if limit is not None:
        images, labels = images[:limit], labels[:limit]
    return {"image": preprocess(images, info), "label": labels}


def make_datasets(
    name: str,
    batch_size: int,
    eval_batch_size: int = 500,
    seed: int = 0,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[ArrayDataset, ArrayDataset]:
    """Train and test ``ArrayDataset``s for ``name`` ("mnist" or "cifar10").

    Training batches are shuffled and resume mid-epoch after a checkpoint
    restore; test batches come in a fixed order and restart for every
    evaluation, so each eval sees the same examples. ``*_limit`` keeps only the
    first N examples, for quick runs.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    train = ArrayDataset(load_arrays(name, "train", train_limit), batch_size=batch_size, seed=seed)
    test = ArrayDataset(
        load_arrays(name, "test", test_limit),
        batch_size=eval_batch_size,
        shuffle=False,
        restart_each_iter=True,
    )
    return train, test
