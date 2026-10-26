# Toylib

Small models built from scratch in JAX for learning, each written up as a blog
post with visualizations of what happens inside. `toylib/` is the shared
library; everything specific to one model or dataset lives in `toylib_projects/`.

| Layer | Where | What it holds |
| --- | --- | --- |
| Core library | `toylib/nn`, `toylib/data` | The `Module` base class, layers, attention; download-once file caching |
| Harness | `toylib/harness` | Training loop (`Experiment`), in-memory datasets, checkpoints, loggers, metrics, inference helpers |
| Visualization | `toylib/viz` | Shared figure style, image grids, training curves, heatmaps, GIFs |
| Projects | `toylib_projects/*` | Posts and longer projects: datasets, models, training scripts, figures |

`toylib_projects/` holds the blog posts (`pNN_*`, see
[`toylib_projects/README.md`](toylib_projects/README.md)) and longer projects:

- `tinystories`: a GPT-2-style decoder-only language model, following nanochat.
- `wm`: a text- and action-conditioned video world model trained on Atari Breakout.
- `dotsAndBoxes`: Monte Carlo tree search for the game Dots and Boxes.
- `bundler`: bundles a script and the local modules it imports into one file.

`archive/` keeps older notebooks that no longer run against the current library.

### Rules between layers

- `toylib` never imports `toylib_projects`. Inside it, `toylib.nn` imports
  nothing else from the repo and `toylib.viz` never imports `toylib.harness` or
  models.
- Posts import from `toylib`, never from another project. Anything two
  projects need moves into `toylib`. Dataset loaders stay with the project that
  uses them. (All enforced by `toylib/layering_test.py`.)

## Installation

```bash
uv sync --extra posts --extra dev        # library, harness, viz, data: enough for every post
uv sync --extra tinystories              # plus the language-model project
uv sync --package toylib-wm              # plus the world-model project
uv sync --all-extras --all-packages      # everything
```

The core package needs only JAX, optax, einops and jaxtyping. The `harness`,
`viz` and `data` extras add orbax/wandb/pandas, matplotlib, and Hugging Face
`datasets` respectively; `posts` installs all three.

## Tests

```bash
uv run pytest toylib                     # core library, harness, viz
uv run pytest toylib_projects/p01_cnn_mnist # one post
uv run pytest toylib toylib_projects     # everything (slow; some tinystories tests need network)
```

## Key locations for AI sessions

Read `CLAUDE.md` first: it has the collaboration rules for posts.

### wm project

All wm source and documentation lives under `toylib_projects/wm/`:

| Path | Contents |
|---|---|
| `toylib_projects/wm/CLAUDE.md` | AI collaboration rules — read this first before helping with wm |
| `toylib_projects/wm/README.md` | Project overview, quickstart, CLI reference |
| `toylib_projects/wm/docs/designs/plan.md` | Staged implementation plan and architecture decisions |
| `toylib_projects/wm/docs/designs/dataset.md` | Stage 1 / Stage 2 dataset pipeline spec |
| `toylib_projects/wm/docs/designs/vision_codec.md` | Track 1 design: KL-VAE image ↔ latent codec |
| `toylib_projects/wm/docs/walkthrough_guidelines.md` | How to author a learning walkthrough doc |
| `toylib_projects/wm/docs/walkthroughs/` | One walkthrough per implementation track (Track 1, 1b, 2, 3, 4) |
| `toylib_projects/wm/datagen/` | Stage 1 episode generation (Breakout env, controller, HDF5 storage) |
| `toylib_projects/wm/viz/` | Episode visualization tools (HTML renderer, matrix overview) |

The wm training scripts use the shared harness in `toylib/harness`.
