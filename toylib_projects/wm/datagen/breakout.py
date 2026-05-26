"""Atari Breakout environment + RAM state extraction.

Encapsulates env construction and state-dict extraction so the rest of the
pipeline never touches `ale-py` directly.

RAM addresses are the commonly-cited values for Atari 2600 Breakout. They
must be verified empirically against the pinned ROM (see `breakout_test.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)

ENV_ID = "ALE/Breakout-v5"

# Minimal action set for Breakout (matches env.action_space, Discrete(4)).
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
ACTION_NAMES = ("NOOP", "FIRE", "RIGHT", "LEFT")
MODES = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

# RAM addresses for Atari 2600 Breakout. Commonly cited values; verified by
# breakout_test.py against the pinned ale-py ROM.
RAM_PADDLE_X = 72
RAM_BALL_X = 99
RAM_BALL_Y = 101
RAM_SCORE_HIGH = 77  # BCD-encoded score, two bytes
RAM_SCORE_LOW = 78
RAM_BRICKS_REMAINING = 76  # not fully validated; see breakout_test.py
RAM_LIVES = 57

# Native frame dimensions returned by ALE.
FRAME_H = 210
FRAME_W = 160


class State(TypedDict):
    paddle_x: float
    ball_x: float
    ball_y: float
    score: int
    bricks_remaining: int
    lives: int


def make_env(seed: int | None = None, difficulty: int = 0, mode: int = 0) -> gym.Env:
    """Build a Breakout env at native 60Hz with deterministic dynamics.

    `frameskip=1` disables ALE's default frame-skip so we record every physics
    tick. `repeat_action_probability=0.0` disables sticky actions for clean
    action-conditioned data.
    """
    env = gym.make(
        ENV_ID,
        difficulty=difficulty,
        mode=mode,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def _bcd_to_int(byte: int) -> int:
    """Decode a binary-coded-decimal byte (e.g. 0x42 → 42)."""
    return ((byte >> 4) & 0xF) * 10 + (byte & 0xF)


def extract_state(ram: np.ndarray) -> State:
    """Pull a State dict out of a 128-byte RAM snapshot."""
    score = _bcd_to_int(int(ram[RAM_SCORE_HIGH])) * 100 + _bcd_to_int(
        int(ram[RAM_SCORE_LOW])
    )
    return State(
        paddle_x=float(ram[RAM_PADDLE_X]),
        ball_x=float(ram[RAM_BALL_X]),
        ball_y=float(ram[RAM_BALL_Y]),
        score=score,
        bricks_remaining=int(ram[RAM_BRICKS_REMAINING]),
        lives=int(ram[RAM_LIVES]),
    )


@dataclass
class Step:
    """Single env step record."""

    frame: np.ndarray  # (210, 160, 3) uint8
    action: int
    state: State
    terminated: bool
    truncated: bool
