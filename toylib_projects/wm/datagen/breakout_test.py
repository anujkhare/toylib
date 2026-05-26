"""Smoke tests: env loads, RAM addresses are sane, end-to-end shard writes work."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .breakout import (
    FIRE,
    LEFT,
    NOOP,
    RIGHT,
    extract_state,
    make_env,
)
from .controller import Controller
from .generate_raw import run_episode
from .storage import ShardWriter


def test_action_meanings_match_constants() -> None:
    env = make_env(seed=0)
    assert env.unwrapped.get_action_meanings() == ["NOOP", "FIRE", "RIGHT", "LEFT"]
    assert (NOOP, FIRE, RIGHT, LEFT) == (0, 1, 2, 3)
    env.close()


def test_paddle_x_responds_to_actions() -> None:
    """RAM[paddle_x] must move when we send RIGHT, and move back with LEFT."""
    env = make_env(seed=0)
    env.reset(seed=0)
    # Launch the ball so the game is active.
    for _ in range(5):
        env.step(FIRE)

    initial = extract_state(env.unwrapped.ale.getRAM())["paddle_x"]
    for _ in range(40):
        env.step(RIGHT)
    after_right = extract_state(env.unwrapped.ale.getRAM())["paddle_x"]
    for _ in range(80):
        env.step(LEFT)
    after_left = extract_state(env.unwrapped.ale.getRAM())["paddle_x"]
    env.close()

    assert after_right > initial, f"RIGHT did not increase paddle_x ({initial} → {after_right})"
    assert after_left < after_right, f"LEFT did not decrease paddle_x ({after_right} → {after_left})"


def test_lives_starts_positive() -> None:
    env = make_env(seed=0)
    env.reset(seed=0)
    state = extract_state(env.unwrapped.ale.getRAM())
    env.close()
    assert state["lives"] >= 1, f"Unexpected initial lives: {state['lives']}"


def test_run_episode_terminates() -> None:
    """One full episode at 60Hz must terminate and produce aligned arrays."""
    env = make_env(seed=0)
    controller = Controller(rng=np.random.default_rng(0))
    frames, actions, states = run_episode(env, controller, max_steps=20_000)
    env.close()
    assert frames.shape[0] == actions.shape[0] == len(states)
    assert frames.shape[1:] == (210, 160, 3)
    assert frames.dtype == np.uint8
    assert actions.dtype == np.int32


def test_shard_writer_roundtrip(tmp_path: Path) -> None:
    """Write one tiny episode through the shard writer and read it back."""
    L = 8
    frames = np.zeros((L, 210, 160, 3), dtype=np.uint8)
    actions = np.zeros((L,), dtype=np.int32)
    states = [
        dict(paddle_x=10.0, ball_x=20.0, ball_y=30.0, score=0, bricks_remaining=108, lives=5)
        for _ in range(L)
    ]

    with ShardWriter(tmp_path, episodes_per_shard=2) as writer:
        writer.write_episode(frames, actions, states, mode=8, difficulty=1, seed=42)

    shard = tmp_path / "episodes_shard_0000.h5"
    assert shard.exists()
    with h5py.File(shard, "r") as f:
        assert "episode_000000" in f
        ep = f["episode_000000"]
        assert ep.attrs["length"] == L
        assert int(ep.attrs["mode"]) == 8
        assert int(ep.attrs["difficulty"]) == 1
        assert int(ep.attrs["seed"]) == 42
        assert ep["frames"].shape == (L, 210, 160, 3)
        assert ep["actions"].shape == (L,)
        for key in ("paddle_x", "ball_x", "ball_y", "score", "bricks_remaining", "lives"):
            assert ep[f"states/{key}"].shape == (L,)


def test_shard_writer_rolls(tmp_path: Path) -> None:
    """After episodes_per_shard episodes, writer must roll to a new file."""
    L = 4
    frames = np.zeros((L, 210, 160, 3), dtype=np.uint8)
    actions = np.zeros((L,), dtype=np.int32)
    states = [
        dict(paddle_x=0.0, ball_x=0.0, ball_y=0.0, score=0, bricks_remaining=0, lives=0)
        for _ in range(L)
    ]
    with ShardWriter(tmp_path, episodes_per_shard=1) as writer:
        writer.write_episode(frames, actions, states)
        writer.write_episode(frames, actions, states)

    assert (tmp_path / "episodes_shard_0000.h5").exists()
    assert (tmp_path / "episodes_shard_0001.h5").exists()


def test_generate_matrix_writes_per_combo_dirs(tmp_path: Path) -> None:
    """Matrix generator produces a sub-directory per (mode, difficulty) combo
    with mode/difficulty correctly stamped on each episode."""
    from .generate_matrix import _combo_dir
    from .generate_raw import run_generation

    combos = [(0, 0), (8, 1)]
    for mode, diff in combos:
        run_generation(
            num_episodes=1,
            output_dir=_combo_dir(tmp_path, mode, diff),
            episodes_per_shard=1,
            max_steps=400,  # keep it tiny — controller may die before this
            seed=mode * 7 + diff,
            mode=mode,
            difficulty=diff,
        )

    for mode, diff in combos:
        d = _combo_dir(tmp_path, mode, diff)
        assert d.exists(), f"missing {d}"
        shard = d / "episodes_shard_0000.h5"
        assert shard.exists()
        with h5py.File(shard, "r") as f:
            ep = f["episode_000000"]
            assert int(ep.attrs["mode"]) == mode
            assert int(ep.attrs["difficulty"]) == diff
