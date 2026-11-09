"""The Imagenette pipeline: augmentation, batching and evaluation arrays."""

import numpy as np
import pytest

from toylib_projects.p03_vit import data
from toylib_projects.p03_vit import testing


@pytest.fixture
def photos():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (6, 200, 200, 3), dtype=np.uint8)


def test_preprocess_roundtrip(photos):
    x = data.preprocess(photos)
    assert x.dtype == np.float32
    np.testing.assert_allclose(data.unpreprocess(x), photos / 255.0, atol=1e-6)


def test_random_resized_crop_shape_and_seed(photos):
    crops = data.random_resized_crop(photos, np.random.default_rng(1), 160)
    again = data.random_resized_crop(photos, np.random.default_rng(1), 160)
    assert crops.shape == (6, 160, 160, 3) and crops.dtype == np.uint8
    np.testing.assert_array_equal(crops, again)


def test_full_crop_is_a_resize(photos):
    """With the whole image as the box, the crop is just a resize."""
    crops = data.random_resized_crop(
        photos, np.random.default_rng(0), 100, scale=(1.0, 1.0), ratio=(1.0, 1.0)
    )
    smooth = np.full((1, 200, 200, 3), 77, np.uint8)
    np.testing.assert_array_equal(
        data.random_resized_crop(smooth, np.random.default_rng(0), 100, (1, 1), (1, 1)),
        np.full((1, 100, 100, 3), 77, np.uint8),
    )
    assert crops.shape == (6, 100, 100, 3)


def test_random_flip_flips_some(photos):
    flipped = data.random_flip(photos, np.random.default_rng(0))
    same = [np.array_equal(a, b) for a, b in zip(flipped, photos)]
    mirrored = [np.array_equal(a, b[:, ::-1]) for a, b in zip(flipped, photos)]
    assert all(s or m for s, m in zip(same, mirrored))
    assert any(same) and any(mirrored)


def test_cutout_blanks_at_most_one_square():
    images = np.ones((50, 32, 32, 3), np.float32)
    out = data.cutout(images, np.random.default_rng(0), size=8)
    hidden = (out[..., 0] == 0).sum(axis=(1, 2))
    assert hidden.max() == 64 and hidden.min() > 0
    assert np.isin(out, [0.0, 1.0]).all()


def test_center_crop_takes_the_middle():
    images = np.arange(10 * 10).reshape(1, 10, 10, 1)
    np.testing.assert_array_equal(data.center_crop(images, 4)[0, :, :, 0], images[0, 3:7, 3:7, 0])


def test_make_datasets_batches():
    load = testing.SyntheticImagenette(n_train=40, n_test=24)
    train, test = data.make_datasets(batch_size=8, eval_batch_size=12, cutout_size=16, load=load)

    batch = next(iter(train))
    assert batch["image"].shape == (8, 160, 160, 3) and batch["image"].dtype == np.float32
    assert batch["label"].dtype == np.int32
    assert abs(batch["image"].mean()) < 3  # Standardized, not 0-255.
    assert train.batches_per_epoch == 5 and test.batches_per_epoch == 2

    # Evaluation is the same every time; training batches differ from epoch to epoch.
    first, second = list(test), list(test)
    np.testing.assert_array_equal(first[0]["image"], second[0]["image"])
    epoch_two = list(train)[-1]
    assert not np.array_equal(epoch_two["image"], batch["image"])


def test_limit_keeps_the_first_images():
    load = testing.SyntheticImagenette(n_train=40, n_test=24)
    train, test = data.make_datasets(batch_size=4, eval_batch_size=4, limit=8, load=load)
    assert train.num_examples == 8 and test.num_examples == 8
    np.testing.assert_array_equal(test.arrays["label"], load("test")[1][:8])


def test_evaluation_arrays_cover_every_image():
    load = testing.SyntheticImagenette(n_train=8, n_test=13)
    arrays = data.evaluation_arrays(load=load)
    images, labels = load("test")
    assert arrays["image"].shape == (13, 160, 160, 3)
    np.testing.assert_array_equal(arrays["label"], labels)
    np.testing.assert_allclose(
        arrays["image"], data.preprocess(data.center_crop(images, 160)), atol=1e-6
    )
