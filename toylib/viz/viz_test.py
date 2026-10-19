"""Smoke tests for toylib.viz: every helper renders and writes files."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from toylib import viz  # noqa: E402


@pytest.fixture(autouse=True)
def _style_and_cleanup():
    viz.use_style()
    yield
    plt.close("all")


class TestToDisplay:
    def test_uint8_grayscale_with_channel(self):
        x = np.full((2, 4, 4, 1), 255, dtype=np.uint8)
        out = viz.to_display(x)
        assert out.shape == (2, 4, 4)
        assert out.max() == 1.0

    def test_signed_floats_are_shifted(self):
        x = np.array([[[-1.0, 1.0]]])
        np.testing.assert_allclose(viz.to_display(x), [[[0.0, 1.0]]])

    def test_rejects_bad_rank(self):
        with pytest.raises(ValueError):
            viz.to_display(np.zeros((4, 4)))


class TestImageGrid:
    @pytest.mark.parametrize(
        "images",
        [
            np.random.randint(0, 255, (10, 28, 28, 1), dtype=np.uint8),
            np.random.rand(10, 8, 8, 3).astype(np.float32),
            np.random.randn(10, 5, 5).astype(np.float32),
        ],
    )
    def test_draws_one_panel_per_image(self, images):
        fig = viz.image_grid(images, ncols=4, titles=[str(i) for i in range(10)])
        shown = [ax for ax in fig.axes if ax.images]
        assert len(shown) == 10

    def test_symmetric_filters(self, tmp_path):
        kernels = np.random.randn(16, 3, 3).astype(np.float32)
        fig = viz.image_grid(kernels, normalize="symmetric", ncols=8)
        (path,) = viz.save(fig, tmp_path / "filters.png")
        assert path.exists()

    def test_each_normalizes_per_image(self):
        maps = np.stack([np.zeros((4, 4)), np.full((4, 4), 5.0)]) + np.eye(4)
        fig = viz.image_grid(maps, normalize="each")
        for ax in fig.axes[:2]:
            data = ax.images[0].get_array()
            assert data.min() == 0.0 and data.max() == 1.0


class TestCurves:
    def _write_log(self, path, losses):
        with open(path, "w") as f:
            f.write("\n")  # FileLogger starts with a blank line
            for step, loss in enumerate(losses):
                record = {"train/loss": loss, "step": step, "timestamp": "t", "samples": ["x"]}
                f.write(json.dumps(record) + "\n")
            f.write(json.dumps({"val/loss": 1.5, "step": 2}) + "\n")

    def test_read_log(self, tmp_path):
        path = tmp_path / "logs_run.txt"
        self._write_log(path, [3.0, 2.0, 1.0])
        curves = viz.read_log(path)
        assert set(curves) == {"train/loss", "val/loss"}
        steps, values = curves["train/loss"]
        np.testing.assert_array_equal(steps, [0, 1, 2])
        np.testing.assert_array_equal(values, [3.0, 2.0, 1.0])
        assert viz.find_log(tmp_path) == path

    def test_plot_compares_runs(self, tmp_path):
        a, b = tmp_path / "logs_a.txt", tmp_path / "logs_b.txt"
        self._write_log(a, [3.0, 2.0, 1.0])
        self._write_log(b, [3.0, 2.5, 2.0])
        ax = viz.plot_curves({"mlp": a, "cnn": b}, smoothing=0.5)
        assert len(ax.lines) == 2
        assert ax.get_legend() is not None

    def test_ema_debiased(self):
        np.testing.assert_allclose(viz.ema(np.ones(5), 0.9), np.ones(5))


class TestHeatmaps:
    def test_attention_heatmap(self):
        weights = np.tril(np.ones((6, 6)))
        weights /= weights.sum(axis=1, keepdims=True)
        ax = viz.heatmap(weights, row_labels=list("abcdef"), col_labels=list("abcdef"))
        assert ax.images

    def test_annotated_diverging(self):
        ax = viz.heatmap(np.array([[-1.0, 0.0], [0.5, 1.0]]), diverging=True, annotate=True)
        assert len(ax.texts) == 4

    def test_token_heatmap_wraps_lines(self):
        text = "To be, or not\nto be"
        fig = viz.token_heatmap(list(text), np.linspace(0, 4, len(text)),
                                colorbar_label="loss (nats)", title="per-character loss")
        assert fig.axes[0].get_ylim()[0] == 2  # two lines of text

    def test_token_heatmap_length_mismatch(self):
        with pytest.raises(ValueError):
            viz.token_heatmap(["a", "b"], np.zeros(3))


def test_save_gif(tmp_path):
    frames = [np.full((20, 30, 3), i * 40, dtype=np.uint8) for i in range(4)]
    fig, ax = plt.subplots(figsize=(1, 1))
    frames.append(fig)
    path = viz.save_gif(frames, tmp_path / "out.gif", fps=10)
    with Image.open(path) as gif:
        assert gif.n_frames == 5
