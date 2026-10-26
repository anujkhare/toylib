# Claude Code Instructions

## Python Environment

This project uses `uv` for environment management. Always use `uv run` instead of calling `python` or `pytest` directly.

```bash
# Run tests for the library (nn, harness, viz)
uv run pytest -v toylib

# Run tests for one post
uv run pytest -v toylib_projects/p01_cnn_mnist

# Run tests for tinystories and toylib (common)
uv run pytest -v toylib toylib_projects/tinystories

# Run tests for all projects (only do if absolutely required)
uv run pytest -v toylib toylib_projects

# Run any Python command
uv run python <args>
```

Never use bare `python`, `python3`, or `pytest` commands — they will use the wrong environment.

## Repository layout

See `README.md`. In short: `toylib/` holds only shared code: `toylib/nn`
(layers), `toylib/data` (download-once caching), `toylib/harness` (training,
checkpoints, inference), `toylib/viz` (figures). Everything specific to a model
or dataset (loaders such as MNIST and CIFAR-10 included) lives in
`toylib_projects/`: one `pNN_*` folder per blog post, plus longer projects.
Keep the layering rules in the README; `toylib/layering_test.py` checks the
import direction.

## Working on posts

The repo owner implements each post's model and training code themselves, to
learn as they go, with help as needed.

- Don't write a post's model or training code (`model.py`, `train.py`) unless
  explicitly asked. Do review diffs, explain, write tests, and debug.
- Infrastructure that isn't the learning goal is fine to write: `toylib/harness`,
  `toylib/viz`, data plumbing, and figure helpers.
- Each post's `README.md` holds its plan: the question, milestones with
  success criteria, and commands. `toylib_projects/README.md` holds the shared
  conventions and the post template.
