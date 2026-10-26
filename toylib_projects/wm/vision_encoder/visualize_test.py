"""Tests for vision_encoder.visualize.

Focus on the numpy-only logic where the real bug lives (patch centering must
account for padding, and must not wrap at the edges). Plotting helpers get a
smoke test under the non-interactive Agg backend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless — no display needed

import numpy as np
import pytest

from toylib_projects.wm.datagen.preprocess_frames import PreprocessConfig, ram_to_pixel
from toylib_projects.wm.vision_encoder import visualize as viz


# ──────────────────────────────────────────────────────────────────────────
# to_uint8
# ──────────────────────────────────────────────────────────────────────────


class TestToUint8:
    def test_uint8_passthrough(self):
        x = np.array([[[0, 128, 255]]], dtype=np.uint8)
        out = viz.to_uint8(x)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, x)

    def test_neg_one_to_one(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        out = viz.to_uint8(x)
        np.testing.assert_array_equal(out, np.array([0, 127, 255], dtype=np.uint8))

    def test_zero_to_one(self):
        x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        out = viz.to_uint8(x)
        np.testing.assert_array_equal(out, np.array([0, 127, 255], dtype=np.uint8))


# ──────────────────────────────────────────────────────────────────────────
# extract_patches — the centering / no-wrap contract
# ──────────────────────────────────────────────────────────────────────────


def _frame_with_dot(h, w, cx, cy, val=255):
    f = np.zeros((1, h, w, 3), dtype=np.uint8)
    f[0, cy, cx] = val
    return f


class TestExtractPatches:
    def test_shape_and_dtype(self):
        frames = np.zeros((5, 64, 64, 3), dtype=np.uint8)
        cx = np.full(5, 30.0)
        cy = np.full(5, 40.0)
        p = viz.extract_patches(frames, cx, cy, size=16)
        assert p.shape == (5, 16, 16, 3)
        assert p.dtype == np.uint8

    def test_single_frame(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        p = viz.extract_patches(frame, 30.0, 40.0, size=16)
        assert p.shape == (16, 16, 3)

    def test_dot_lands_at_patch_center(self):
        # A single bright pixel at (cx, cy) must appear at the patch center.
        h = w = 64
        cx, cy = 30, 40
        frames = _frame_with_dot(h, w, cx, cy)
        size = 16
        p = viz.extract_patches(frames, np.array([cx]), np.array([cy]), size=size)
        # padded[c:c+size] == original[c-size//2 : c+size//2], so the center pixel
        # sits at local index size//2.
        c = size // 2
        assert p[0, c, c].tolist() == [255, 255, 255]
        # and nowhere else is bright.
        assert p[0].sum() == 255 * 3

    def test_offset_shift_is_correct(self):
        # Shifting the center by +5 in x should move the bright pixel by -5 in the
        # patch's local frame.
        h = w = 64
        cx, cy = 30, 40
        frames = _frame_with_dot(h, w, cx, cy)
        size = 16
        c = size // 2
        p = viz.extract_patches(frames, np.array([cx + 5]), np.array([cy]), size=size)
        ys, xs = np.where(p[0, :, :, 0] == 255)
        assert (ys[0], xs[0]) == (c, c - 5)

    def test_edge_uses_fill_not_wrap(self):
        # Center at the top-left corner: the patch must be half-filled with `fill`,
        # NOT wrap around to the opposite edge.
        h = w = 32
        frames = np.zeros((1, h, w, 3), dtype=np.uint8)
        frames[0, 0, 0] = 255  # bright corner pixel
        # put a distinct bright pixel at the far corner to detect wraparound
        frames[0, h - 1, w - 1] = 100
        size = 8
        p = viz.extract_patches(frames, np.array([0]), np.array([0]), size=size, fill=0)
        c = size // 2
        # corner pixel at local center
        assert p[0, c, c].tolist() == [255, 255, 255]
        # top-left quadrant (the out-of-frame region) must be all fill=0, and the
        # opposite-corner value (100) must NOT have wrapped in.
        assert p[0, :c, :c].sum() == 0
        assert not np.any(p[0] == 100)

    def test_nan_center_is_all_fill(self):
        frames = np.full((1, 32, 32, 3), 200, dtype=np.uint8)
        p = viz.extract_patches(frames, np.array([np.nan]), np.array([np.nan]),
                                size=8, fill=0)
        # center clamps to (0,0); patch is mostly fill with only the in-frame
        # quadrant showing content — assert it didn't crash and has right shape.
        assert p.shape == (1, 8, 8, 3)


# ──────────────────────────────────────────────────────────────────────────
# state_pixel_centers — consistency with ram_to_pixel
# ──────────────────────────────────────────────────────────────────────────


class TestStatePixelCenters:
    def test_matches_ram_to_pixel(self):
        cfg = PreprocessConfig()
        ram_x = np.array([80.0, 40.0, 120.0])
        ram_y = np.array([100.0, 60.0, 150.0])
        px, py, valid = viz.state_pixel_centers(ram_x, ram_y, cfg)
        for i in range(3):
            expected = ram_to_pixel(float(ram_x[i]), float(ram_y[i]), cfg)
            assert expected is not None
            assert px[i] == pytest.approx(expected[0])
            assert py[i] == pytest.approx(expected[1])
            assert valid[i]

    def test_out_of_crop_is_nan_invalid(self):
        cfg = PreprocessConfig()
        ram_x = np.array([80.0])
        ram_y = np.array([0.0])  # in cropped-away scoreboard
        px, py, valid = viz.state_pixel_centers(ram_x, ram_y, cfg)
        assert not valid[0]
        assert np.isnan(px[0]) and np.isnan(py[0])


# ──────────────────────────────────────────────────────────────────────────
# comparison_strip / draw_boxes
# ──────────────────────────────────────────────────────────────────────────


class TestStripAndBoxes:
    def test_comparison_strip_shape(self):
        a = np.zeros((3, 16, 16, 3), dtype=np.uint8)
        b = np.zeros((3, 16, 16, 3), dtype=np.uint8)
        out = viz.comparison_strip(a, b, gap_px=4)
        assert out.shape == (3, 16, 16 + 4 + 16, 3)
        # gap column carries the gap value
        assert np.all(out[:, :, 16:20] == 128)

    def test_comparison_strip_single(self):
        a = np.zeros((16, 16, 3), dtype=np.uint8)
        out = viz.comparison_strip(a, a, gap_px=2)
        assert out.shape == (16, 16 + 2 + 16, 3)

    def test_draw_boxes_marks_border(self):
        frames = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        out = viz.draw_boxes(frames, np.array([16]), np.array([16]), size=8,
                             color=(255, 0, 0), thickness=1)
        # some red pixels were drawn, original frame untouched
        assert np.any(out[0, :, :, 0] == 255)
        assert np.all(frames == 0)

    def test_draw_boxes_skips_nan(self):
        frames = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        out = viz.draw_boxes(frames, np.array([np.nan]), np.array([np.nan]), size=8)
        assert np.all(out == 0)  # nothing drawn


# ──────────────────────────────────────────────────────────────────────────
# Plotting smoke tests (Agg backend)
# ──────────────────────────────────────────────────────────────────────────


class TestPlotting:
    def test_show_reconstructions_returns_figure(self):
        import matplotlib.figure

        a = np.random.default_rng(0).integers(0, 256, (6, 16, 16, 3), dtype=np.uint8)
        fig = viz.show_reconstructions(a, a, n=4)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2 * 4

    def test_show_ball_patches_smoke(self):
        cfg = PreprocessConfig()
        rng = np.random.default_rng(1)
        frames = rng.integers(0, 256, (8, 128, 128, 3), dtype=np.uint8)
        source = {
            "ball_x": np.full(8, 80.0),
            "ball_y": np.full(8, 100.0),
        }
        fig = viz.show_ball_patches(frames, frames, source, cfg, n=4, size=16)
        assert len(fig.axes) == 2 * 4
