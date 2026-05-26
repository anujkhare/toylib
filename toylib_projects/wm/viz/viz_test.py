"""Viz smoke tests: loader roundtrip, HTML rendering produces non-empty output."""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

from .loader import list_episodes, load_episode
from .render import RenderOptions, build_episode_page, build_episode_viewer, build_shard_index_page


def _write_fake_shard(path: Path, n_episodes: int = 2, length: int = 24) -> Path:
    """Write a tiny synthetic shard matching the Stage 1 schema."""
    with h5py.File(path, "w") as f:
        for ep_idx in range(n_episodes):
            grp = f.create_group(f"episode_{ep_idx:06d}")
            grp.attrs["length"] = np.int32(length)
            frames = np.random.default_rng(ep_idx).integers(
                0, 256, size=(length, 210, 160, 3), dtype=np.uint8
            )
            grp.create_dataset("frames", data=frames, **hdf5plugin.LZ4())
            grp.create_dataset("actions", data=np.zeros(length, dtype=np.int32))
            states = grp.create_group("states")
            for k in ("paddle_x", "ball_x", "ball_y"):
                states.create_dataset(k, data=np.arange(length, dtype=np.float32))
            for k in ("score", "bricks_remaining", "lives"):
                states.create_dataset(k, data=np.zeros(length, dtype=np.int32))
    return path


def test_list_and_load(tmp_path: Path) -> None:
    shard = _write_fake_shard(tmp_path / "shard.h5", n_episodes=3, length=8)
    names = list_episodes(shard)
    assert names == ["episode_000000", "episode_000001", "episode_000002"]
    ep = load_episode(shard, "episode_000001")
    assert ep.length == 8
    assert ep.frames.shape == (8, 210, 160, 3)
    assert ep.actions.shape == (8,)
    assert set(ep.states.keys()) == {
        "paddle_x", "ball_x", "ball_y", "score", "bricks_remaining", "lives",
    }


def test_build_episode_viewer_runs(tmp_path: Path) -> None:
    shard = _write_fake_shard(tmp_path / "shard.h5", n_episodes=1, length=12)
    ep = load_episode(shard, "episode_000000")
    # Keep it small for the test: no upscale, low quality, big downsample.
    opts = RenderOptions(fps=15, downsample=1, upscale=1, webp_quality=40)
    html = build_episode_viewer(ep, opts)
    # Basic structural checks.
    assert "data:image/webp;base64," in html
    assert "data:image/png;base64," in html
    assert "episode_000000" in html
    assert "<script>" in html and "</script>" in html


def test_build_episode_page_is_full_document(tmp_path: Path) -> None:
    shard = _write_fake_shard(tmp_path / "shard.h5", n_episodes=1, length=8)
    ep = load_episode(shard, "episode_000000")
    opts = RenderOptions(fps=15, downsample=1, upscale=1, webp_quality=40)
    page = build_episode_page(ep, opts)
    assert page.startswith("<!DOCTYPE html>")
    assert "<html>" in page and "</html>" in page


def test_build_shard_index() -> None:
    rows = [
        {"name": "shard_0000 / episode_000000", "length": 1500, "score": 200,
         "lives_start": 5, "lives_end": 0, "href": "shard_0000__episode_000000.html"},
        {"name": "shard_0000 / episode_000001", "length": 2100, "score": 350,
         "lives_start": 5, "lives_end": 0, "href": "shard_0000__episode_000001.html"},
    ]
    page = build_shard_index_page(rows)
    assert "2 episodes" in page
    assert "shard_0000__episode_000000.html" in page
    assert "shard_0000__episode_000001.html" in page
