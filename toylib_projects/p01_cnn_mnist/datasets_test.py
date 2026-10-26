"""MNIST and CIFAR-10 loaders: parsing on synthetic files, plus the real downloads."""

import gzip
import io
import pickle
import struct
import tarfile
import urllib.error

import numpy as np
import pytest

from toylib.data import download
from toylib_projects.p01_cnn_mnist import cifar10, mnist


def _write_idx(path, array: np.ndarray) -> None:
    header = b"\x00\x00\x08" + bytes([array.ndim]) + struct.pack(f">{array.ndim}i", *array.shape)
    with gzip.open(path, "wb") as f:
        f.write(header + array.astype(np.uint8).tobytes())


def _cifar_batch(images: np.ndarray, labels: list[int]) -> bytes:
    # The archive stores each image channel-first, flattened to 3072 values.
    data = images.transpose(0, 3, 1, 2).reshape(len(images), -1)
    return pickle.dumps({b"data": data, b"labels": labels})


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TOYLIB_CACHE", str(tmp_path))
    return tmp_path


def test_parse_idx_roundtrip(tmp_path):
    array = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    _write_idx(tmp_path / "x.gz", array)
    np.testing.assert_array_equal(mnist.parse_idx(tmp_path / "x.gz"), array)


def test_parse_idx_rejects_other_formats(tmp_path):
    with gzip.open(tmp_path / "bad.gz", "wb") as f:
        f.write(b"\x00\x00\x0d\x01\x00\x00\x00\x01\x00")  # dtype 0x0d = float32
    with pytest.raises(ValueError, match="not a uint8 IDX"):
        mnist.parse_idx(tmp_path / "bad.gz")


def test_mnist_load_from_cache(cache):
    """With the files already cached, load() parses them and never downloads."""
    root = cache / "mnist"
    root.mkdir()
    images = np.random.default_rng(0).integers(0, 256, (5, 28, 28), dtype=np.uint8)
    labels = np.array([3, 1, 4, 1, 5], dtype=np.uint8)
    _write_idx(root / "t10k-images-idx3-ubyte.gz", images)
    _write_idx(root / "t10k-labels-idx1-ubyte.gz", labels)

    x, y = mnist.load("test")

    assert x.shape == (5, 28, 28, 1) and x.dtype == np.uint8
    assert y.dtype == np.int32
    np.testing.assert_array_equal(x[..., 0], images)
    np.testing.assert_array_equal(y, labels)


def test_parse_cifar_batch_is_nhwc():
    images = np.random.default_rng(0).integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)
    x, y = cifar10.parse_batch(_cifar_batch(images, [7, 2]))
    np.testing.assert_array_equal(x, images)
    np.testing.assert_array_equal(y, [7, 2])
    assert y.dtype == np.int32


def test_cifar_load_from_archive(cache):
    """load() reads every train batch out of the cached archive, then caches arrays."""
    root = cache / "cifar10"
    root.mkdir()
    rng = np.random.default_rng(0)
    batches = {f"data_batch_{i}": rng.integers(0, 256, (2, 32, 32, 3), dtype=np.uint8) for i in range(1, 6)}
    with tarfile.open(root / "cifar-10-python.tar.gz", "w:gz") as tar:
        for name, imgs in batches.items():
            raw = _cifar_batch(imgs, [0, 1])
            info = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))

    x, y = cifar10.load("train")

    assert x.shape == (10, 32, 32, 3) and x.dtype == np.uint8
    np.testing.assert_array_equal(x, np.concatenate(list(batches.values())))
    assert (root / "train.npz").exists()
    # Second call uses the .npz, even with the archive gone.
    (root / "cifar-10-python.tar.gz").unlink()
    np.testing.assert_array_equal(cifar10.load("train")[0], x)


def test_bad_split_rejected():
    with pytest.raises(ValueError):
        mnist.load("val")
    with pytest.raises(ValueError):
        cifar10.load("val")


def test_real_mnist():
    """Downloads MNIST (about 11 MB) on first run; skipped when offline."""
    try:
        x_train, y_train = mnist.load("train")
        x_test, y_test = mnist.load("test")
    except urllib.error.URLError as e:
        pytest.skip(f"MNIST not cached and download failed: {e}")
    assert x_train.shape == (60_000, 28, 28, 1)
    assert x_test.shape == (10_000, 28, 28, 1)
    assert set(np.unique(y_train)) == set(range(10))
    assert y_test.shape == (10_000,)


@pytest.mark.skipif(
    not (download.cache_dir("cifar10") / "test.npz").exists(),
    reason="CIFAR-10 not cached (160 MB download); run cifar10.load('test') once",
)
def test_real_cifar10():
    x_train, y_train = cifar10.load("train")
    x_test, y_test = cifar10.load("test")
    assert x_train.shape == (50_000, 32, 32, 3)
    assert x_test.shape == (10_000, 32, 32, 3)
    assert np.bincount(y_test).tolist() == [1000] * 10
