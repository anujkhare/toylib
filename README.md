# Toylib

Small models built from scratch in JAX for learning, each written up as a blog
post with visualizations of what happens inside. The repo has four layers;
each depends only on the ones above it.

| Layer | Where | What it holds |
| --- | --- | --- |
| Core library | `toylib/nn`, `toylib/data` | The `Module` base class, layers, attention; dataset downloads into NumPy arrays |
| Harness | `toylib/harness` | Training loop (`Experiment`), in-memory datasets, checkpoints, loggers, metrics, inference helpers |
| Visualization | `toylib/viz` | Shared figure style, image grids, training curves, heatmaps, GIFs |
| Posts | `toylib_posts/pNN_*` | One folder per post: model, training script, figures, the post itself |

`toylib_projects/` holds longer projects that use the same library:

- `tinystories`: a GPT-2-style decoder-only language model, following nanochat.
- `wm`: a text- and action-conditioned video world model trained on Atari Breakout.
- `dotsAndBoxes`: Monte Carlo tree search for the game Dots and Boxes.
- `bundler`: bundles a script and the local modules it imports into one file.

`archive/` keeps older notebooks that no longer run against the current library.

### Rules between layers

- `toylib.nn` imports nothing else from the repo. `toylib.viz` never imports
  `toylib.harness` or models. (Enforced by `toylib/layering_test.py`.)
- Posts import from `toylib`, never from each other or from `toylib_projects`.
  Anything two posts need moves into `toylib`.

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
uv run pytest toylib_posts/p01_cnn_mnist # one post
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
