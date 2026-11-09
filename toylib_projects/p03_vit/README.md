# Post 3: a Vision Transformer

A ViT trained from scratch on Imagenette at 160x160, with post 1's data and recipe, so it
compares directly with post 1's MLP (58.1% test accuracy) and ResNet (87.4%).

## Status

The data, training and Colab notebook are in place and tested with a stand-in model. The
ViT itself, `model.ViT` and `model.loss_fn`, isn't written yet, so training stops with
`NotImplementedError` until it is.

## Files

- `imagenette.py`: downloads Imagenette once and returns uint8 photos at 200x200 with
  int32 labels. It's a copy of post 1's loader, and shares its cache.
- `data.py`: crops to 160x160 and standardizes each batch as it's served, with post 1's
  augmentation for training, and wraps both splits in `ArrayDataset`.
- `model.py`: `ModelConfig`, `ViT`, `build_model`, `loss_fn` and `forward`. Its docstring
  says what the rest of the code needs from it.
- `metrics.py`: `Accuracy` for `Task(metrics=[...])`.
- `train.py`: `RUNS`, the recipe, and `make_experiment`, `time_run` and `train`.
- `notebooks/01_imagenette_vit_colab.ipynb`: trains runs on Colab, with checkpoints in
  GCS and metrics on W&B, and resumes after a lost runtime. `notebooks/COLAB.md` shows
  how to run it with the Colab CLI.
- `testing.py`: a stand-in model and synthetic Imagenette, so the tests of everything
  around the model pass before the model exists.
- Tests sit next to the code: `model_test.py` checks the ViT's interface and skips
  until it's written. The others check the loader, the data, training and resuming,
  and the notebook, which `notebooks/notebook_test.py` runs locally.

## Commands

```bash
uv sync --extra posts --extra dev
```

```bash
uv run pytest toylib_projects/p03_vit
```

Train the `smoke` run locally, on 64 images, in seconds once the ViT exists:

```bash
uv run python -m toylib_projects.p03_vit.train --run smoke
```

The real runs go through the notebook on Colab. See `notebooks/COLAB.md`.
