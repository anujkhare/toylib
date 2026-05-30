# Toylib

This repository contains multiple sub-projects focused on building certain models from scratch in jax for educational purposes. It's broken down into the following:

- `toylib` is a small jax based ML library with implementations for basic layers (such as linear and attention).
- `toylib_projects` contains projects that use the toylib library
  - `tinystories` is a basic decoder-only LLM
  - `wm` is a text + action-conditioned video world model trained on Atari Breakout

## Key locations for AI sessions

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

## Installation

### Install just toylib (core package)

uv sync

### Install toylib + tinystories dependencies

uv sync --extra tinystories

### Install toylib + dotsandboxes dependencies  

uv sync --extra dotsandboxes

### Install a specific subproject (installs toylib + its deps)

uv sync --package toylib-tinystories

### Install everything

uv sync --all-extras --all-packages
