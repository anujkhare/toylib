# wm — Text + Action-Conditioned Video World Model

Toy from-scratch world model in JAX, trained on Gymnasium Atari Breakout.

See [docs/plan.md](docs/plan.md) for the project plan and [docs/dataset.md](docs/dataset.md) for the dataset pipeline spec.

## Stage 1: raw episode generation

From this directory.

**Single (mode, difficulty) combination:**

```bash
uv run python -m datagen.generate_raw \
    --num-episodes 1000 \
    --output-dir data/raw/mode_00_diff_0 \
    --episodes-per-shard 50 \
    --mode 0 --difficulty 0 \
    --seed 0
```

**Sweep over a matrix of Breakout variants** (default `modes=[0,8,20,40]` × `difficulties=[0,1]`, skipping the three "Catch" modes 12/28/44):

```bash
uv run python -m datagen.generate_matrix \
    --output-root data/raw \
    --episodes-per-combo 100 \
    --episodes-per-shard 25 \
    --base-seed 0
```

This writes `data/raw/mode_MM_diff_D/episodes_shard_NNNN.h5` for each combo. The `--skip-existing` flag makes the sweep resumable.

## Visualization

Render any shard (or whole tree) into self-contained HTML files:

```bash
uv run python -m viz.cli --input data/raw --output viz_out
open viz_out/index.html
```

Or inside Jupyter / Colab:

```python
from viz import load_episode, show_episode
ep = load_episode("data/raw/mode_00_diff_0/episodes_shard_0000.h5", "episode_000000")
show_episode(ep)
```

## Tests

```bash
uv run pytest -v datagen/ viz/
```
