# Toylib

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
