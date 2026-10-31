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
    x = np.asarray(images) * np.asarray(info.std, np.float32) + np.asarray(
        info.mean, np.float32
    )
    return np.clip(x, 0.0, 1.0)


def random_flip_and_crop(
    batch: dict[str, np.ndarray], rng: np.random.Generator, pad: int = 4
) -> dict[str, np.ndarray]:
    """Standard CIFAR-10 augmentation, drawn independently for each image.

    Flips horizontally with probability 1/2, then pads by ``pad`` pixels and crops
    back to the original size at a random offset. Pixels are standardized, so the
    zero padding is the dataset's mean colour.
    """
    images = batch["image"]
    n, h, w, _ = images.shape
    flip = rng.random(n) < 0.5
    images = np.where(flip[:, None, None, None], images[:, :, ::-1], images)
    padded = np.pad(images, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    rows = rng.integers(0, 2 * pad + 1, (n, 1)) + np.arange(h)
    cols = rng.integers(0, 2 * pad + 1, (n, 1)) + np.arange(w)
    crops = padded[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]]
    return {**batch, "image": crops}


def cutout(
    batch: dict[str, np.ndarray], rng: np.random.Generator, size: int = 16
) -> dict[str, np.ndarray]:
    """Blank one random ``size`` x ``size`` square in each image (DeVries & Taylor, 2017).

    The square's centre is uniform over the image, so it can hang off an edge and
    hide less. Pixels are standardized, so zero is the dataset's mean colour.
    """
    images = batch["image"]
    n, h, w, _ = images.shape
    top = rng.integers(0, h, (n, 1)) - size // 2
    left = rng.integers(0, w, (n, 1)) - size // 2
    rows = (np.arange(h) >= top) & (np.arange(h) < top + size)
    cols = (np.arange(w) >= left) & (np.arange(w) < left + size)
    hidden = rows[:, :, None] & cols[:, None, :]
    return {**batch, "image": images * ~hidden[..., None]}


def load_arrays(
    name: str, split: str, limit: int | None = None
) -> dict[str, np.ndarray]:
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
    augment: bool = False,
    cutout_size: int = 0,
) -> tuple[ArrayDataset, ArrayDataset]:
    """Train and test ``ArrayDataset``s for ``name`` ("mnist" or "cifar10").

    Training batches are shuffled and resume mid-epoch after a checkpoint
    restore; test batches come in a fixed order and restart for every
    evaluation, so each eval sees the same examples. ``*_limit`` keeps only the
    first N examples, for quick runs. ``augment`` applies
    ``random_flip_and_crop`` to training batches, and a non-zero ``cutout_size``
    follows it with ``cutout``; test batches are never augmented.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    if cutout_size and not augment:
        raise ValueError("cutout_size needs augment=True")

    def flip_crop_and_cutout(batch, rng):
        return cutout(random_flip_and_crop(batch, rng), rng, cutout_size)

    transform = random_flip_and_crop if augment else None
    if cutout_size:
        transform = flip_crop_and_cutout
    train = ArrayDataset(
        load_arrays(name, "train", train_limit),
        batch_size=batch_size,
        seed=seed,
        transform=transform,
    )
    test = ArrayDataset(
        load_arrays(name, "test", test_limit),
        batch_size=eval_batch_size,
        shuffle=False,
        restart_each_iter=True,
    )
    return train, test
