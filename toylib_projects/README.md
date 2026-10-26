# Projects and posts

Everything specific to one model or dataset lives here; `toylib/` keeps only
the shared library. Longer projects (`tinystories`, `wm`, `dotsAndBoxes`,
`bundler`) have their own READMEs. The rest of this page is about the posts.

One `pNN_*` folder per blog post. Each post trains a small model on a known
dataset and answers one question about its insides with a figure.

| Post | Folder | Question | Status |
| --- | --- | --- | --- |
| 0 | [`p00_module_from_scratch`](p00_module_from_scratch/) | What is a pytree, and how did "every array is a parameter" end up training our RoPE tables? | Draft notebooks |
| 1 | [`p01_cnn_mnist`](p01_cnn_mnist/) | Is a CNN translation-invariant? | Data and pipeline ready; model next |
| 2 | [`p02_char_gpt`](p02_char_gpt/) | When does a small GPT learn to copy? | Not started |

## Layout of a post

```
pNN_slug/
  README.md      the plan: question, milestones, how to run
  <dataset>.py   loaders for the post's datasets (raw NumPy arrays)
  data.py        preprocessing -> ArrayDataset
  model.py       the model (yours to write)
  train.py       data -> ArrayDataset -> Experiment -> checkpoints + logs (yours)
  figures.py     every figure in the post, rebuilt from checkpoints and logs
  *_test.py      tests next to the code they test
  POST.md        the post itself
  figures/       final figures the post embeds (committed)
  outputs/       checkpoints, logs, scratch figures (gitignored)
```

Run anything as a module from the repo root, e.g.
`uv run python -m toylib_projects.p01_cnn_mnist.train`.

## What a post builds on

Posts import from `toylib` and never from another project. If two projects
need the same thing, it moves into `toylib`.

| Need | Use |
| --- | --- |
| Layers, attention, the `Module` base class | `toylib.nn` |
| Download a file once and cache it | `toylib.data.download` |
| Arrays to batches the harness understands | `toylib.harness.data.ArrayDataset` |
| Training loop, checkpoints, logging, eval | `toylib.harness.experiment.Experiment` |
| Restore a checkpoint, list checkpoints, run a model over a dataset | `toylib.harness.inference` |
| Parameter counts and memory estimates | `toylib.harness.analyze` |
| Style, image grids, curves, heatmaps, GIFs | `toylib.viz` |

## Conventions

- **Inspection.** Models take `return_intermediates: bool = False`; when true they
  return `(output, acts)` where `acts` maps names such as `"conv1"` or
  `"blocks.0.attn_weights"` to arrays. Under `jit`, mark the flag static.
- **Checkpoints over training.** Save at fixed step intervals (not only at the
  end); "X over training" figures read them with `inference.checkpoint_steps`.
- **Predict first.** Write the prediction in `POST.md` before running the
  experiment that answers it.
- **Figures are rebuilt by code.** Every figure in `figures/` comes from
  `figures.py`, so re-running training regenerates the post.

## Post template (`POST.md`)

1. **The question**, with the figure that answers it as a teaser.
2. **The model in one screen**: a shape trace and a link to the code.
3. **Training**: the loss curve, what healthy looks like, time and hardware.
4. **The experiment**: the prediction, then the result.
5. **What surprised us**, including bugs found along the way.
6. **Try it yourself**: one command, and one thing to change.

1,500 to 2,500 words and three to five figures, each with a one-sentence takeaway as its caption.
