"""Download shards from the climbmix-400b-shuffle dataset."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542


def shard_filename(shard_idx: int) -> str:
    return f"shard_{shard_idx:05d}.parquet"


def download_shard(shard_idx: int, output_dir: Path, client: httpx.Client) -> Path:
    filename = shard_filename(shard_idx)
    dest = output_dir / filename
    if dest.exists():
        return dest

    url = f"{BASE_URL}/{filename}"
    with client.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        dest.write_bytes(resp.read())
    return dest


def download(
    num_shards: int, output_dir: Path, workers: int, start_offset: int = 0
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = list(range(start_offset, start_offset + num_shards))

    with httpx.Client(timeout=300) as client:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(download_shard, i, output_dir, client): i for i in indices
            }
            with tqdm(total=num_shards, unit="shard") as bar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"\nFailed shard {idx}: {exc}", file=sys.stderr)
                    bar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download climbmix dataset shards.")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=MAX_SHARD + 1,
        help=f"Number of shards to download (default: all {MAX_SHARD + 1})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/climbmix"),
        help="Directory to save shards (default: data/climbmix)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers (default: 8)",
    )
    args = parser.parse_args()

    if args.num_shards < 1 or args.num_shards > MAX_SHARD + 1:
        parser.error(f"--num-shards must be between 1 and {MAX_SHARD + 1}")

    print(
        f"Downloading {args.num_shards} shards → {args.output_dir} ({args.workers} workers)"
    )
    download(args.num_shards, args.output_dir, args.workers)
    print("Done.")


if __name__ == "__main__":
    main()
