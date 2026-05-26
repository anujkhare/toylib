"""Sweep over a matrix of (mode, difficulty) combinations.

Generates one sub-dataset per combo into ``{output_root}/mode_MM_diff_D/`` so
downstream code can either consume a single combo or merge across them.

The default mode list is a curated subset of Breakout variants:

    [0, 8, 20, 40]      — standard Breakout flavors

The Atari 2600 "Catch" variants (modes 12, 28, 44) are deliberately excluded:
they have no bricks and follow different game mechanics, so mixing them in
would change the action/reward semantics rather than just the dynamics.

Usage (run from `toylib_projects/wm/`):
    uv run python -m datagen.generate_matrix \\
        --output-root data/raw \\
        --episodes-per-combo 100 \\
        --modes 0 8 20 40 \\
        --difficulties 0 1 \\
        --episodes-per-shard 25 \\
        --base-seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .generate_raw import run_generation

DEFAULT_MODES = (0, 8, 20, 40)
DEFAULT_DIFFICULTIES = (0, 1)


def _combo_dir(root: Path, mode: int, difficulty: int) -> Path:
    return root / f"mode_{mode:02d}_diff_{difficulty}"


def _combo_seed(base_seed: int, mode: int, difficulty: int) -> int:
    """Deterministic per-combo seed so each (mode, diff) gets a distinct stream."""
    return base_seed + 1_000_003 * mode + 1009 * difficulty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/raw"),
        help="Root directory; each combo writes into a `mode_MM_diff_D/` subdir.",
    )
    parser.add_argument(
        "--episodes-per-combo", type=int, default=100,
        help="Number of episodes generated per (mode, difficulty) combination.",
    )
    parser.add_argument(
        "--modes", type=int, nargs="+", default=list(DEFAULT_MODES),
        help=f"Game modes to sweep. Default {DEFAULT_MODES}. Avoid 12/28/44 (Catch).",
    )
    parser.add_argument(
        "--difficulties", type=int, nargs="+", default=list(DEFAULT_DIFFICULTIES),
        help=f"Difficulties to sweep. Default {DEFAULT_DIFFICULTIES}.",
    )
    parser.add_argument("--episodes-per-shard", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument(
        "--base-seed", type=int, default=0,
        help="Seed offset; per-combo seeds derive deterministically from this.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a combo if its output directory already contains shards.",
    )
    args = parser.parse_args()

    combos = [(m, d) for m in args.modes for d in args.difficulties]
    print(f"Sweeping {len(combos)} combos × {args.episodes_per_combo} episodes each")
    for i, (mode, diff) in enumerate(combos):
        out_dir = _combo_dir(args.output_root, mode, diff)
        if args.skip_existing and any(out_dir.glob("episodes_shard_*.h5")):
            print(f"[{i + 1}/{len(combos)}] skip mode={mode} diff={diff} (exists)")
            continue
        seed = _combo_seed(args.base_seed, mode, diff)
        print(f"\n[{i + 1}/{len(combos)}] mode={mode} diff={diff} → {out_dir} (seed={seed})")
        run_generation(
            num_episodes=args.episodes_per_combo,
            output_dir=out_dir,
            episodes_per_shard=args.episodes_per_shard,
            max_steps=args.max_steps,
            seed=seed,
            mode=mode,
            difficulty=diff,
        )


if __name__ == "__main__":
    main()
