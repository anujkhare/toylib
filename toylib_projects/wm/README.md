# wm — Text + Action-Conditioned Video World Model

Toy from-scratch world model in JAX, trained on Gymnasium Atari Breakout.

See [docs/designs/plan.md](docs/designs/plan.md) for the project plan and [docs/designs/dataset.md](docs/designs/dataset.md) for the dataset pipeline spec. Learning walkthroughs (one per track) live in [docs/walkthroughs/](docs/walkthroughs/); authoring guidelines are in [docs/walkthrough_guidelines.md](docs/walkthrough_guidelines.md).

## Project layout

```
toylib_projects/wm/
├── README.md
├── CLAUDE.md                   — AI collaboration rules for this project
├── pyproject.toml
├── docs/
│   ├── walkthrough_guidelines.md  — how to author a walkthrough doc
│   ├── designs/
│   │   ├── plan.md             — staged project plan & architecture decisions
│   │   └── dataset.md          — Stage 1 / Stage 2 dataset spec
│   └── walkthroughs/           — one learning walkthrough per implementation track
├── datagen/                    — Stage 1 raw episode generation + Stage 2 compilers
│   ├── breakout.py             — env + RAM state extraction
│   ├── controller.py           — ε-greedy mixed-competency policy
│   ├── storage.py              — HDF5 shard writer
│   ├── generate_raw.py         — single (mode, diff) generator
│   ├── generate_matrix.py      — sweep over (mode, diff) combos
│   ├── run_stage1.py           — canonical 1000-episode entry point
│   ├── preprocess_frames.py    — reusable crop + resize (VAE input pipeline)
│   ├── generate_vision_enc_data.py  — VAE dataset compiler (Stage 2)
│   ├── breakout_test.py        — Stage 1 tests
│   └── preprocess_test.py      — preprocess + VAE compiler tests
├── viz/                        — episode visualization
│   ├── loader.py               — load Episode from HDF5
│   ├── render.py               — build self-contained HTML
│   ├── cli.py                  — per-episode deep-dive HTML pages
│   ├── matrix.py               — mode × difficulty overview grid
│   └── viz_test.py             — viz tests
└── data/raw/                   — generated dataset (gitignored)
    └── mode_MM_diff_D/
        └── episodes_shard_NNNN.h5
```

## Setup

This package is a member of the toylib uv workspace. From the repo root:

```bash
uv sync --package toylib-wm
```

All `uv run` commands below assume the working directory is `toylib_projects/wm/`.

## Quickstart (end-to-end)

```bash
cd toylib_projects/wm

# 1) Verify the env + dataset code works (under a second)
uv run pytest -v datagen/ viz/

# 2) Sanity-check the generator on a tiny sweep (16 episodes, ~5s)
uv run python -m datagen.run_stage1 --smoke --output-root /tmp/wm_smoke
uv run python -m viz.matrix --input /tmp/wm_smoke --output /tmp/wm_smoke_viz
open /tmp/wm_smoke_viz/matrix.html

# 3) Run the full Stage 1 sweep (~5-10 minutes, ~3.6 GB on disk)
uv run python -m datagen.run_stage1

# 4) Build the overview viz, then drill into one combo
uv run python -m viz.matrix --input data/raw --output viz_out/matrix
open viz_out/matrix/matrix.html
```

## Stage 1: raw episode generation

Three entry points, in order of decreasing convenience.

### A. Canonical full sweep (`run_stage1.py`)

Bakes in the `docs/dataset.md` §2 plan: 4 Breakout modes × 2 difficulties × 125 episodes = **1,000 episodes total, ~3.6 GB on disk**.

```bash
uv run python -m datagen.run_stage1                  # full run
uv run python -m datagen.run_stage1 --smoke          # 16 ep sanity test (~5s)
uv run python -m datagen.run_stage1 --skip-existing  # resume after interruption
uv run python -m datagen.run_stage1 --output-root /tmp/wm_test  # side-by-side run
uv run python -m datagen.run_stage1 --base-seed 42   # different seed offset
```

### B. Custom matrix sweep (`generate_matrix.py`)

```bash
uv run python -m datagen.generate_matrix \
    --output-root data/raw \
    --episodes-per-combo 100 \
    --modes 0 8 20 40 \
    --difficulties 0 1 \
    --episodes-per-shard 25 \
    --base-seed 0
```

Notes:

- **Modes 12, 28, 44 are deliberately excluded** by default — they're the Atari "Catch" variants (no bricks; different game mechanics). See `docs/dataset.md` §2E.
- Per-combo seeds are derived deterministically from `--base-seed`.
- `--skip-existing` skips combos whose output dir already has shards (resumable).

### C. Single combo (`generate_raw.py`)

For one-off experiments with a specific mode/difficulty:

```bash
uv run python -m datagen.generate_raw \
    --num-episodes 1000 \
    --output-dir data/raw/mode_00_diff_0 \
    --episodes-per-shard 50 \
    --mode 0 --difficulty 0 \
    --max-steps 20000 \
    --seed 0
```

### Output layout

All three entry points write the same per-combo layout, so the viz tools and Stage 2 compiler can consume them interchangeably:

```
data/raw/
├── mode_00_diff_0/episodes_shard_0000.h5
├── mode_00_diff_0/episodes_shard_0001.h5
├── ...
└── mode_40_diff_1/episodes_shard_NNNN.h5
```

Each `episodes_shard_NNNN.h5` contains multiple `episode_NNNNNN/` groups. See `docs/designs/dataset.md` §2 for the full schema.

## Stage 2: vision-encoder dataset

Build a flat, pre-resized, stratified-sampled dataset for VAE training (see `datagen/generate_vision_enc_data.py` docstring for the design rationale):

```bash
# Defaults: 100k train / 10k val / 10k test at 128×128, sampled at native 60Hz
uv run python -m datagen.generate_vision_enc_data

# Diverse sampling for VAE — 6Hz effective rate avoids near-duplicates
uv run python -m datagen.generate_vision_enc_data \
    --n-train 100000 --n-val 10000 --n-test 10000 \
    --input-fps 6 \
    --output-root data/compiled

# Smoke run (few seconds, ~7 MB on disk)
uv run python -m datagen.generate_vision_enc_data \
    --n-train 2000 --n-val 200 --n-test 200 \
    --input-fps 6 \
    --output-root /tmp/wm_vae
```

Outputs `data/compiled/vae_{train,val,test}.h5`. Each file has `frames (N,H,W,3) uint8` plus a `source/` group with per-frame state and provenance. Configurable knobs:

- `--target-size N` — output frame size (default 128)
- `--input-fps N` — temporal sub-sampling rate from native 60Hz (default 60 = every frame)
- `--score-buckets a b c ...` — score-stratum boundaries
- `--crop-top --crop-bottom --crop-left --crop-right` — explicit crop bounds
- `--resize-filter {lanczos,bilinear,bicubic,nearest,box}` — default lanczos

## Visualization

Two complementary tools.

### Mode × difficulty sample matrix (`viz.matrix`)

Builds a top-level grid of auto-looping animated-WebP thumbnails — one cell per (mode, difficulty) combo — so you can see variation across the dataset at a glance. Each cell links to a per-combo page with more samples.

```bash
uv run python -m viz.matrix \
    --input data/raw \
    --output viz_out/matrix \
    --samples-per-combo 3
open viz_out/matrix/matrix.html
```

Optional flags:

- `--samples-per-combo N` — how many episodes to sample per combo (default 3)
- `--thumb-downsample K` — take every Kth frame for thumbnails (default 4)
- `--thumb-max-frames N` — cap on frames in each thumb's WebP (default 300)

### Per-episode deep dive (`viz.cli`)

Full scrubber + per-frame metadata panel + state-over-time chart. One HTML file per episode, plus an `index.html` listing them all.

```bash
# Whole dataset (one HTML per episode + index)
uv run python -m viz.cli --input data/raw --output viz_out/full
open viz_out/full/index.html

# Single shard
uv run python -m viz.cli \
    --input data/raw/mode_20_diff_1/episodes_shard_0000.h5 \
    --output viz_out/m20d1

# One specific episode
uv run python -m viz.cli \
    --input data/raw/mode_20_diff_1/episodes_shard_0000.h5 \
    --episode episode_000000 \
    --output viz_out/single
```

Optional rendering flags (defaults are usually fine):

- `--fps 30` — playback rate of the per-frame WebP carousel
- `--downsample 2` — take every Nth frame
- `--upscale 2` — pixel-double the frames for visibility
- `--webp-quality 70` — 0..100 tradeoff

### Inside Jupyter / Colab

The same renderer wraps cleanly via `IPython.display.HTML`:

```python
from viz import load_episode, show_episode, RenderOptions

ep = load_episode(
    "data/raw/mode_00_diff_0/episodes_shard_0000.h5",
    "episode_000000",
)
show_episode(ep)  # renders inline

# With custom render options
show_episode(ep, RenderOptions(downsample=4, upscale=1, webp_quality=50))
```

## Tests

```bash
uv run pytest -v datagen/ viz/   # all tests
uv run pytest -v datagen/        # datagen only
uv run pytest -v viz/            # viz only
```

## Disk usage / clean-up

```bash
du -sh data/raw                  # ~3.6 GB after a full sweep
du -sh viz_out                   # depends on samples + episode count

rm -rf data/raw                  # nuke the dataset
rm -rf viz_out                   # nuke the viz output
```

## Known open issues

See `docs/designs/dataset.md` §5. Current outstanding items:

- **`bricks_remaining` RAM address**: `RAM[76]` is the wrong byte — the brick state is a bitmap across multiple bytes, not a single int. The field is currently stuck at 0 in all episodes.
- **Brick-clear level reset**: not yet handled. Episodes that span an all-bricks-cleared reset will see a discontinuous state transition.
