"""The Imagenette loader: parsing a synthetic archive, plus the real data if cached."""

import io
import tarfile

import numpy as np
import pytest
from PIL import Image

from toylib.data import download
from toylib_projects.p03_vit import imagenette


def _jpeg(colour: int, size: tuple[int, int]) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (colour, colour, colour)).save(out, format="JPEG")
    return out.getvalue()


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A tiny archive in the real layout: two classes in train, one photo in val."""
    monkeypatch.setenv("TOYLIB_CACHE", str(tmp_path))
    root = tmp_path / "imagenette"
    root.mkdir()
    files = {
        "imagenette2-320/train/n01440764/a.JPEG": _jpeg(10, (320, 240)),
        "imagenette2-320/train/n03888257/b.JPEG": _jpeg(200, (240, 320)),
        "imagenette2-320/val/n02102040/c.JPEG": _jpeg(100, (320, 320)),
        "imagenette2-320/noise.csv": b"not a photo",
    }
    with tarfile.open(root / "imagenette2-320.tgz", "w:gz") as tar:
        for name, raw in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return root


def test_load_from_archive(archive):
    images, labels = imagenette.load("train")

    assert images.shape == (2, 200, 200, 3) and images.dtype == np.uint8
    assert labels.dtype == np.int32
    # Order is shuffled, so match photos to labels by their colour.
    by_label = {int(label): int(image.mean().round()) for image, label in zip(images, labels)}
    assert by_label.keys() == {0, 9}
    assert abs(by_label[0] - 10) < 5 and abs(by_label[9] - 200) < 5

    # The decoded arrays are cached, so the archive isn't needed again.
    (archive / "imagenette2-320.tgz").unlink()
    np.testing.assert_array_equal(imagenette.load("train")[0], images)


def test_test_split_reads_val(archive):
    images, labels = imagenette.load("test")
    assert images.shape == (1, 200, 200, 3)
    assert labels.tolist() == [1]


def test_bad_split_rejected():
    with pytest.raises(ValueError):
        imagenette.load("val")


@pytest.mark.skipif(
    not (download.cache_dir("imagenette") / "test_200.npz").exists(),
    reason="Imagenette not cached (340 MB download); run imagenette.load('test') once",
)
def test_real_imagenette():
    images, labels = imagenette.load("test")
    assert images.shape == (3_925, 200, 200, 3)
    assert set(np.unique(labels)) == set(range(10))
