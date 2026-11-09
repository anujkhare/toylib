"""Imagenette as harness datasets, with post 1's augmentation.

``imagenette.load`` returns uint8 photos stored at 200x200. This module crops them
to 160x160 per batch, standardizes them, and wraps them in ``ArrayDataset`` so
``Experiment`` can iterate them. Each batch is a dict::

    {"image": float32 (B, 160, 160, 3), "label": int32 (B,)}

Training batches get post 1's recipe: a random resized crop, a horizontal flip,
and optionally a cutout square. Test batches get a centre crop. The augmentation is
copied from post 1 rather than imported, because posts don't import each other.
"""

import concurrent.futures
import typing

import numpy as np
from PIL import Image

from toylib.harness.data import ArrayDataset
from toylib_projects.p03_vit import imagenette

IMAGE_SIZE = 160
CLASS_NAMES = imagenette.CLASS_NAMES

# ImageNet's per-channel statistics, after scaling to [0, 1]. Imagenette is a
# subset of ImageNet.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# split -> (uint8 images (N, H, W, 3), int32 labels (N,))
Loader = typing.Callable[[str], tuple[np.ndarray, np.ndarray]]


def preprocess(images: np.ndarray) -> np.ndarray:
    """uint8 in [0, 255] -> float32 with zero mean and unit variance per channel."""
    x = images.astype(np.float32) / 255.0
    return (x - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)


def unpreprocess(images: np.ndarray) -> np.ndarray:
    """Undo ``preprocess`` back to floats in [0, 1], for plotting."""
    x = np.asarray(images) * np.asarray(STD, np.float32) + np.asarray(MEAN, np.float32)
    return np.clip(x, 0.0, 1.0)


_RESIZE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def random_resized_crop(
    images: np.ndarray,
    rng: np.random.Generator,
    size: int,
    scale: tuple[float, float] = (0.35, 1.0),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
) -> np.ndarray:
    """Crop a random box from each uint8 image and resize it to ``size``.

    The box covers a fraction of the image drawn from ``scale``, with an aspect
    ratio drawn from ``ratio``, so the model sees objects at many sizes.
    """
    n, h, w, c = images.shape
    area = rng.uniform(*scale, n) * h * w
    aspect = np.exp(rng.uniform(*np.log(ratio), n))
    box_h = np.minimum(np.sqrt(area / aspect), h)
    box_w = np.minimum(np.sqrt(area * aspect), w)
    top = rng.uniform(0, 1, n) * (h - box_h)
    left = rng.uniform(0, 1, n) * (w - box_w)

    crops = np.empty((n, size, size, c), dtype=np.uint8)

    def resize_one(i):
        box = (left[i], top[i], left[i] + box_w[i], top[i] + box_h[i])
        crops[i] = Image.fromarray(images[i]).resize(
            (size, size), Image.Resampling.BILINEAR, box=box
        )

    list(_RESIZE_POOL.map(resize_one, range(n)))  # PIL resizes outside the GIL.
    return crops


def random_flip(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip each image left to right with probability 1/2."""
    flip = rng.random(len(images)) < 0.5
    return np.where(flip[:, None, None, None], images[:, :, ::-1], images)


def cutout(images: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """Blank one random ``size`` x ``size`` square in each image (DeVries & Taylor, 2017).

    The square's centre is uniform over the image, so it can hang off an edge and
    hide less. Images are standardized, so zero is the dataset's mean colour.
    """
    n, h, w, _ = images.shape
    top = rng.integers(0, h, (n, 1)) - size // 2
    left = rng.integers(0, w, (n, 1)) - size // 2
    rows = (np.arange(h) >= top) & (np.arange(h) < top + size)
    cols = (np.arange(w) >= left) & (np.arange(w) < left + size)
    hidden = rows[:, :, None] & cols[:, None, :]
    return images * ~hidden[..., None]


def center_crop(images: np.ndarray, size: int) -> np.ndarray:
    """The central ``size`` x ``size`` square of each image."""
    top = (images.shape[1] - size) // 2
    left = (images.shape[2] - size) // 2
    return images[:, top : top + size, left : left + size]


def _arrays(load: Loader, split: str, limit: int | None) -> dict[str, np.ndarray]:
    images, labels = load(split)
    return {"image": images[:limit], "label": labels[:limit]}


def make_datasets(
    batch_size: int,
    eval_batch_size: int = 400,
    seed: int = 0,
    crop_scale: tuple[float, float] = (0.35, 1.0),
    cutout_size: int = 0,
    limit: int | None = None,
    load: Loader | None = None,
) -> tuple[ArrayDataset, ArrayDataset]:
    """Train and test ``ArrayDataset``s for Imagenette.

    The arrays stay uint8, and every batch is cropped and standardized as it's
    served, since float32 copies would take 8 GB. Training batches are shuffled and
    resume mid-epoch after a checkpoint restore. Test batches come in a fixed order
    and restart for every evaluation, so each one sees the same images. ``limit``
    keeps only the first N images of each split, for quick runs, and ``load`` swaps
    in another source of arrays, such as synthetic ones in tests.
    """
    load = load or imagenette.load

    def augment(batch, rng):
        images = random_resized_crop(batch["image"], rng, IMAGE_SIZE, crop_scale)
        images = preprocess(random_flip(images, rng))
        if cutout_size:
            images = cutout(images, rng, cutout_size)
        return {**batch, "image": images}

    def evaluate(batch, rng):
        del rng
        return {**batch, "image": preprocess(center_crop(batch["image"], IMAGE_SIZE))}

    train = ArrayDataset(
        _arrays(load, "train", limit), batch_size=batch_size, seed=seed, transform=augment
    )
    test = ArrayDataset(
        _arrays(load, "test", limit),
        batch_size=eval_batch_size,
        shuffle=False,
        restart_each_iter=True,
        transform=evaluate,
    )
    return train, test


def evaluation_arrays(
    limit: int | None = None, load: Loader | None = None
) -> dict[str, np.ndarray]:
    """Every test image, centre-cropped and standardized, with its label.

    The harness drops a partial last batch, so its evaluation misses a few test
    images. Run the model over these with ``inference.batched_apply`` for the final
    accuracy.
    """
    arrays = _arrays(load or imagenette.load, "test", limit)
    return {**arrays, "image": preprocess(center_crop(arrays["image"], IMAGE_SIZE))}
