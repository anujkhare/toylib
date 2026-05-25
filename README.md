# Toylib

This repository contains multiple sub-projects focused on building certain models from scratch in jax for educational purposes. It's broken down into the following:

- `toylib` is a small jax based ML library with implementations for basic layers (such as linear and attention).
- `toylib_projects` contains projects that use the toylib library
  - `tinystories` is a basic decoder-only LLM
  - `wm` is a basic world model

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
