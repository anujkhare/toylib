"""ε-greedy mixed-competency Breakout controller.

Drives data generation with a deliberately imperfect policy so the dataset
covers near-misses, deaths, and life-resets — not just clean tracking runs.

Policy per step (after FIRE-on-life-start):
  - 80% competent tracking (match paddle x to ball x)
  -  5% deliberate miss (when ball is in lower half, move *away* from ball)
  - 15% random jitter (uniform over {LEFT, RIGHT, NOOP})

FIRE is sent on the first step of every life (and at episode start).
"""

from __future__ import annotations

import numpy as np

from breakout import FIRE, LEFT, NOOP, RIGHT, FRAME_H, State


class Controller:
    def __init__(
        self,
        rng: np.random.Generator,
        eps_jitter: float = 0.15,
        eps_miss: float = 0.05,
        deadzone: float = 4.0,
    ) -> None:
        """
        Args:
            rng: numpy Generator for all stochastic decisions.
            eps_jitter: probability of a random jitter action.
            eps_miss: probability of a deliberate miss (only triggers when the
                ball is in the lower half of the frame).
            deadzone: |paddle_x - ball_x| below which the tracker emits NOOP
                instead of LEFT/RIGHT — prevents paddle jitter when aligned.
        """
        self.rng = rng
        self.eps_jitter = eps_jitter
        self.eps_miss = eps_miss
        self.deadzone = deadzone
        self._prev_lives: int | None = None

    def reset(self) -> None:
        self._prev_lives = None

    def act(self, state: State) -> int:
        # Life transition (or first step of episode): launch the ball.
        if self._prev_lives is None or state["lives"] < self._prev_lives:
            self._prev_lives = state["lives"]
            return FIRE
        self._prev_lives = state["lives"]

        r = self.rng.random()
        if r < self.eps_jitter:
            return int(self.rng.choice([LEFT, RIGHT, NOOP]))
        if r < self.eps_jitter + self.eps_miss and state["ball_y"] > FRAME_H / 2:
            return self._away_from_ball(state)
        return self._track_ball(state)

    def _track_ball(self, state: State) -> int:
        dx = state["ball_x"] - state["paddle_x"]
        if abs(dx) < self.deadzone:
            return NOOP
        return RIGHT if dx > 0 else LEFT

    def _away_from_ball(self, state: State) -> int:
        dx = state["ball_x"] - state["paddle_x"]
        if abs(dx) < self.deadzone:
            return int(self.rng.choice([LEFT, RIGHT]))
        return LEFT if dx > 0 else RIGHT
