"""Stage 1 raw episode generator.

Drives Gymnasium Atari Breakout at 60Hz with an ε-greedy mixed-competency
controller and writes complete episodes to sharded HDF5 files.

Usage (run from `toylib_projects/wm/`):
    uv run python -m datagen.generate_raw \\
        --num-episodes 1000 \\
        --output-dir data/raw/mode_00_diff_0 \\
        --episodes-per-shard 50 \\
        --mode 0 --difficulty 0 \\
        --seed 0

For a sweep over (mode, difficulty) combinations, see ``generate_matrix.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .breakout import State, extract_state, make_env
from .controller import Controller
from .storage import ShardWriter


def run_episode(
    env, controller: Controller, max_steps: int
) -> tuple[np.ndarray, np.ndarray, list[State]]:
    """Play one complete episode; return frames, actions, states."""
    controller.reset()
    obs, _info = env.reset()

    frames: list[np.ndarray] = [obs]
    actions: list[int] = []
    states: list[State] = [extract_state(env.unwrapped.ale.getRAM())]

    for _ in range(max_steps):
        action = controller.act(states[-1])
        obs, _reward, terminated, truncated, _info = env.step(action)
        actions.append(action)
        frames.append(obs)
        states.append(extract_state(env.unwrapped.ale.getRAM()))
        if terminated or truncated:
            break

    # Trim trailing frame/state so len(frames) == len(actions) == len(states).
    frames.pop()
    states.pop()

    return (
        np.stack(frames, axis=0),
        np.asarray(actions, dtype=np.int32),
        states,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument("--episodes-per-shard", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", type=int, default=0,
        help="ALE game mode. Breakout exposes [0,4,8,...,44]; modes 12/28/44 are 'Catch'.",
    )
    parser.add_argument(
        "--difficulty", type=int, default=0,
        help="ALE difficulty switch (0=easy, 1=hard / half-size paddle).",
    )
    args = parser.parse_args()

    run_generation(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        episodes_per_shard=args.episodes_per_shard,
        max_steps=args.max_steps,
        seed=args.seed,
        mode=args.mode,
        difficulty=args.difficulty,
    )


def run_generation(
    *,
    num_episodes: int,
    output_dir: Path,
    episodes_per_shard: int,
    max_steps: int,
    seed: int,
    mode: int,
    difficulty: int,
) -> None:
    """Library entry point: generate N episodes of (mode, difficulty) into output_dir.

    Used by both the CLI here and ``generate_matrix.py`` for sweeps.
    """
    rng = np.random.default_rng(seed)
    env = make_env(seed=seed, mode=mode, difficulty=difficulty)
    controller = Controller(rng=rng)

    with ShardWriter(output_dir, episodes_per_shard) as writer:
        for ep_idx in tqdm(
            range(num_episodes),
            desc=f"mode={mode} diff={difficulty}",
        ):
            frames, actions, states = run_episode(env, controller, max_steps)
            writer.write_episode(
                frames, actions, states,
                mode=mode, difficulty=difficulty, seed=seed,
            )
            tqdm.write(
                f"  ep {ep_idx:04d}: L={len(actions):>5d}  "
                f"score={states[-1]['score']:>4d}  lives_end={states[-1]['lives']}"
            )

    env.close()


if __name__ == "__main__":
    main()
