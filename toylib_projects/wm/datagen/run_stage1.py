"""Canonical Stage 1 dataset generation.

Runs the full sweep prescribed in ``docs/dataset.md`` §2:

  - 4 Breakout modes  × 2 difficulties      = 8 combinations
  - 125 episodes per combo                  = 1,000 episodes total
  - 25 episodes per HDF5 shard              = 40 shards total

Per the dataset spec this targets ~1.5M frames at 60Hz native resolution. The
matrix layout under ``data/raw/mode_MM_diff_D/`` lets downstream code either
mix combos freely or train within a single mode.

Usage (run from `toylib_projects/wm/`):
    uv run python -m datagen.run_stage1                  # full 1000-episode run
    uv run python -m datagen.run_stage1 --smoke          # 2 ep/combo, fast sanity test
    uv run python -m datagen.run_stage1 --skip-existing  # resume after interruption

Any other flag accepted by ``generate_matrix`` may be passed through
(e.g. ``--output-root /tmp/wm_data`` for a side-by-side test).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .generate_matrix import sweep

# Curated Breakout variants (Catch modes 12/28/44 deliberately excluded).
CANONICAL_MODES = [0, 8, 20, 40]
CANONICAL_DIFFICULTIES = [0, 1]

# Aligns with docs/dataset.md §2: 1,000 episodes total, 40 shards.
CANONICAL_EPISODES_PER_COMBO = 125
CANONICAL_EPISODES_PER_SHARD = 25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/raw"),
        help="Root directory for the sweep (default: data/raw).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny sanity-test run: 2 episodes per combo with shorter rollouts.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a combo if its output directory already contains shards.",
    )
    parser.add_argument(
        "--base-seed", type=int, default=0,
        help="Seed offset; per-combo seeds derive deterministically from this.",
    )
    args = parser.parse_args()

    if args.smoke:
        episodes_per_combo = 2
        episodes_per_shard = 2
        max_steps = 1000
    else:
        episodes_per_combo = CANONICAL_EPISODES_PER_COMBO
        episodes_per_shard = CANONICAL_EPISODES_PER_SHARD
        max_steps = 20_000

    print(
        f"Stage 1 generation: {len(CANONICAL_MODES) * len(CANONICAL_DIFFICULTIES)} combos "
        f"× {episodes_per_combo} ep = "
        f"{len(CANONICAL_MODES) * len(CANONICAL_DIFFICULTIES) * episodes_per_combo} episodes total"
    )

    sweep(
        modes=CANONICAL_MODES,
        difficulties=CANONICAL_DIFFICULTIES,
        episodes_per_combo=episodes_per_combo,
        output_root=args.output_root,
        episodes_per_shard=episodes_per_shard,
        max_steps=max_steps,
        base_seed=args.base_seed,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
