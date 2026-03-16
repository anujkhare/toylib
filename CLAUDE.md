# Claude Code Instructions

## Python Environment

This project uses `uv` for environment management. Always use `uv run` instead of calling `python` or `pytest` directly.

```bash
# Run tests for the base library
uv run pytest -v toylib

# Run tests for all projects (only do if absoluted required)
uv run pytest -v toylib toylib_projects

# Run tests for tinystories and toylib (common)
uv run python -v toylib toylib_projects/tinystories

# Run any Python command
uv run python <args>
```

Never use bare `python`, `python3`, or `pytest` commands — they will use the wrong environment.
