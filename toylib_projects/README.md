# Projects and posts

Everything specific to one model or dataset lives here; `toylib/` keeps only
the shared library. Longer projects (`tinystories`, `wm`, `dotsAndBoxes`,
`bundler`) have their own READMEs. The rest of this page is about the posts.

One `pNN_*` folder per blog post. Each post trains a small model on a known
dataset and builds intuition for one question by looking inside the model. Figures
come first: a post doesn't need a measured comparison of architectures.

| Post | Folder | Question | Status |
| --- | --- | --- | --- |
| 0 | [`p00_module_from_scratch`](p00_module_from_scratch/) | What is a pytree, and how did "every array is a parameter" end up training our RoPE tables? | Draft notebooks |
| 1 | [`p01_cnn_mnist`](p01_cnn_mnist/) | Why does a CNN work? | Published |
| 2 | [`p02_tinystories`](p02_tinystories/) | How does a small GPT keep a story straight? | Draft post |

## Layout of a post

```
pNN_slug/
  README.md      the plan: question, milestones, how to run
  <dataset>.py   loaders for the post's datasets (raw NumPy arrays)
  data.py        preprocessing -> ArrayDataset
  model.py       the model (yours to write)
  train.py       data -> ArrayDataset -> Experiment -> checkpoints + logs (yours)
  figures.py     every figure, rebuilt from checkpoints and logs
  *_test.py      tests next to the code they test
  POST.md        the post itself
  figures/       final figures (committed)
  outputs/       checkpoints, logs, scratch figures (gitignored)
  explorer/      optional interactive views, with build_data.py to export their data
```

`p01_cnn_mnist` keeps its notebooks and their Colab guide in `notebooks/`, and everything for
the post, including `figures.py` and `explorer/`, in `post/`.

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
  `figures.py`, so re-running training regenerates the post. Passing figure names
  rebuilds only those.
- **Look before you claim.** A post states what its figure shows, checked by opening the
  figure. Exploratory results get rechecked when they become figures, and if the
  prediction was wrong, the post says so. Label a suggestion as one: a few
  evidence maps can suggest a cause but don't measure one.
- **Heavy training runs remotely.** Use a Colab TPU for anything slow on a laptop, with
  metrics on W&B and checkpoints in GCS, as in `p01_cnn_mnist/notebooks/COLAB.md`. Checkpoints are
  saved every 50 epochs or so, so a lost runtime can resume. Run only one training job
  at a time on the Mac.
- **Explorers run the real model.** Export it to plain JavaScript, with no framework, and
  check the forward pass against JAX with a parity script. Save results for chosen photos
  so the page opens at once, and load the weights in a Web Worker only when a reader
  asks for a live run.

## Post template (`POST.md`)

1. **The question**, with the figure that answers it as a teaser.
2. **The model in one screen**: a shape trace and a link to the code.
3. **Training**: the loss curve, what healthy looks like, time and hardware.
4. **The experiment**: the prediction, then the result.
5. **What surprised us**, including bugs found along the way.
6. **Try it yourself**: one command, and one thing to change.

1,500 to 2,500 words and three to five figures, each with a one-sentence takeaway as its caption.

## Publishing

Posts go on [anujkhare.github.io](https://anujkhare.github.io/), a Hugo site built from a
separate repo. Add the post to `POSTS` in `publish.py`, then copy it across:

```bash
uv run python -m toylib_projects.publish p01_cnn_mnist --site ../anujkhare.github.io
```

`POST.md` stays the source of truth, so make edits here and publish again. Write it for
readers of the website: link code with full GitHub URLs, and keep private resources such
as the GCS bucket out of it. A line `<!-- explorer: stages -->` places an interactive view.

Then build from the website repo and push both its repos. The source is the repo itself,
and the live site is the separate repo in `public/`. Use a plain `hugo` build, because
`hugo server` writes localhost links into `public/`:

```bash
hugo
```
