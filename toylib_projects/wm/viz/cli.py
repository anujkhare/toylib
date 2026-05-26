"""Generate standalone HTML viewers for HDF5 shard(s).

Walks a directory of `episodes_shard_*.h5` files and writes:
  - one `<shard>__<episode>.html` per episode (self-contained, opens directly)
  - one `index.html` listing every episode with summary stats

Usage (run from `toylib_projects/wm/`):
    uv run python -m viz.cli --input data/raw --output viz_out

For a single shard or single episode:
    uv run python -m viz.cli --input data/raw/episodes_shard_0000.h5 --output viz_out
    uv run python -m viz.cli --input data/raw/episodes_shard_0000.h5 \\
        --episode episode_000000 --output viz_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .loader import find_shards, list_episodes, load_episode
from .render import (
    RenderOptions,
    build_episode_page,
    build_shard_index_page,
)


def _nav_link(href: str, label: str) -> str:
    return f"<div class='nav'><a href='{href}'>{label}</a></div>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Either a directory of *.h5 shards or a single shard file.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for generated .html files.",
    )
    parser.add_argument(
        "--episode", type=str, default=None,
        help="If set (with --input pointing at a single shard), only render this episode.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--webp-quality", type=int, default=70)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    opts = RenderOptions(
        fps=args.fps,
        downsample=args.downsample,
        upscale=args.upscale,
        webp_quality=args.webp_quality,
    )

    if args.episode is not None:
        if not args.input.is_file():
            parser.error("--episode requires --input to be a single .h5 file")
        _render_one(args.input, args.episode, args.output, opts)
        return

    shards = find_shards(args.input)
    if not shards:
        parser.error(f"No episodes_shard_*.h5 found under {args.input}")

    rows: list[dict] = []
    for shard in shards:
        for name in tqdm(list_episodes(shard), desc=shard.name):
            href = _render_one(shard, name, args.output, opts)
            ep = load_episode(shard, name)
            rows.append({
                "name": f"{shard.stem} / {name}",
                "mode": ep.mode,
                "difficulty": ep.difficulty,
                "length": ep.length,
                "score": int(ep.states["score"][-1]),
                "lives_start": int(ep.states["lives"][0]),
                "lives_end": int(ep.states["lives"][-1]),
                "href": href,
            })

    index_path = args.output / "index.html"
    index_path.write_text(build_shard_index_page(rows, title=f"Episodes under {args.input}"))
    print(f"Wrote {index_path} ({len(rows)} episodes)")


def _render_one(shard: Path, name: str, out_dir: Path, opts: RenderOptions) -> str:
    """Render one episode page; return the filename (relative to out_dir)."""
    ep = load_episode(shard, name)
    filename = f"{ep.stem}.html"
    nav = _nav_link("index.html", "← index")
    (out_dir / filename).write_text(build_episode_page(ep, opts, nav_html=nav))
    return filename


if __name__ == "__main__":
    main()
