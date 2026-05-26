"""Build a mode × difficulty sample matrix from a Stage 1 dataset tree.

Walks ``{data_root}/mode_MM_diff_D/episodes_shard_*.h5`` and writes:

  - ``matrix.html``       — top-level grid, one cell per combo with sample
                            animated-WebP thumbnails
  - ``mode_MM_diff_D/index.html`` — per-combo page with a few sample episodes
                                    rendered as compact cards

Designed to make the visual variation across modes/difficulties obvious at a
glance, even before any episode is opened individually.

Usage (run from `toylib_projects/wm/`):
    uv run python -m viz.matrix \\
        --input data/raw \\
        --output viz_out/matrix \\
        --samples-per-combo 3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .loader import find_shards, list_episodes, load_episode
from .render import (
    ComboSamples,
    build_combo_index_page,
    build_matrix_index_page,
)

_COMBO_DIR_RE = re.compile(r"mode_(\d+)_diff_(\d+)$")


def _discover_combos(root: Path) -> list[tuple[int, int, Path]]:
    """Find ``mode_MM_diff_D`` subdirs under ``root``. Returns (mode, diff, dir)."""
    out: list[tuple[int, int, Path]] = []
    if not root.exists():
        return out
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        m = _COMBO_DIR_RE.match(sub.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), sub))
    return out


def _sample_episodes(combo_dir: Path, k: int) -> list:
    """Pick the first ``k`` episodes across the shards in this combo dir."""
    eps = []
    for shard in find_shards(combo_dir):
        for name in list_episodes(shard):
            eps.append(load_episode(shard, name))
            if len(eps) >= k:
                return eps
    return eps


def build(
    input_root: Path,
    output_dir: Path,
    *,
    samples_per_combo: int = 3,
    thumb_downsample: int = 4,
    thumb_max_frames: int = 300,
) -> Path:
    """Build the matrix view; return the path to the top-level matrix.html."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = _discover_combos(Path(input_root))
    if not discovered:
        raise SystemExit(
            f"No mode_MM_diff_D directories found under {input_root}. "
            f"Did Stage 1 generation complete?"
        )

    cells: list[ComboSamples] = []
    for mode, diff, combo_dir in discovered:
        eps = _sample_episodes(combo_dir, samples_per_combo)
        if not eps:
            print(f"  (skip {combo_dir.name}: no episodes found)")
            continue
        combo_subdir = f"mode_{mode:02d}_diff_{diff}"
        (output_dir / combo_subdir).mkdir(exist_ok=True)
        combo_href = f"{combo_subdir}/index.html"
        cell = ComboSamples(mode=mode, difficulty=diff, episodes=eps, href=combo_href)
        cells.append(cell)

        # Per-combo page.
        combo_html = build_combo_index_page(
            cell,
            nav_html=(
                "<div class='nav'><a href='../matrix.html'>← matrix</a></div>"
            ),
        )
        (output_dir / combo_href).write_text(combo_html)
        print(f"  wrote {combo_href}  ({len(eps)} samples)")

    matrix_path = output_dir / "matrix.html"
    matrix_path.write_text(
        build_matrix_index_page(
            cells,
            thumb_downsample=thumb_downsample,
            thumb_max_frames=thumb_max_frames,
        )
    )
    print(f"\nMatrix index → {matrix_path}")
    return matrix_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Root containing mode_MM_diff_D/ subdirs.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for the generated matrix HTML tree.")
    parser.add_argument("--samples-per-combo", type=int, default=3)
    parser.add_argument("--thumb-downsample", type=int, default=4,
                        help="Take every Nth frame for thumbnail WebPs (default 4).")
    parser.add_argument("--thumb-max-frames", type=int, default=300,
                        help="Cap on frames included in each matrix thumb (default 300).")
    args = parser.parse_args()

    build(
        input_root=args.input,
        output_dir=args.output,
        samples_per_combo=args.samples_per_combo,
        thumb_downsample=args.thumb_downsample,
        thumb_max_frames=args.thumb_max_frames,
    )


if __name__ == "__main__":
    main()
