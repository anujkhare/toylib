# Post 1: Is a CNN translation-invariant?

Most people would say yes. The flatten layer before the classifier is a reason
to doubt it: convolutions move features along with the digit, but a
flatten-then-linear head reads them from fixed positions.

**Experiment.** Shift MNIST test digits by 0 to 8 pixels and plot accuracy
against shift for three models: an MLP, a CNN with a flatten head, and a CNN
with global average pooling. Readers predict the three curves first.

**Supporting views** (pick one or two):

- First-layer filters animated over training (`viz.image_grid(..., normalize="symmetric")` per checkpoint, then `viz.save_gif`).
- A probe ladder: logistic regression on raw pixels, and on each layer of a random and of the trained network.
- Effective receptive field: `jax.grad` of one unit with respect to the input, averaged over images ([Luo et al., 2016](https://proceedings.neurips.cc/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html)).
- The 25 test digits with the highest loss.

## Milestones

Each is one session. You write the code; ask for review, tests or debugging.

1. ~~**MNIST loader**~~ Done: `mnist.py` and `cifar10.py` in this folder,
   wrapped for the harness by `data.py`. `01_data_and_pipeline.ipynb` shows a batch
   and runs the skeleton in `model.py` through `Experiment` for a few steps.
2. **MLP baseline** in `model.py`, on `toylib.nn.mlp.MLP` (fix it first: review fix 3).
   *Done when:* it overfits one batch within 50 steps.
3. **Pooling, flatten and a two-layer CNN.**
   *Done when:* a printed shape trace matches your hand calculation and the model overfits one batch.
4. **Training through the harness** in `train.py`: `ArrayDataset` for train and
   test, `Experiment` with a `model_factory` and `forward_fn`, checkpoints every
   few hundred steps. *Done when:* `viz.plot_curves` shows the loss curve,
   test accuracy is logged, and `inference.checkpoint_steps` lists the checkpoints.
5. **`return_intermediates` and the shift experiment** in `figures.py`, using
   `inference.load_model` and `inference.batched_apply`.
   *Done when:* accuracy-vs-shift curves for all three models are on one plot.
6. **One or two supporting views.** *Done when:* each figure has a one-sentence takeaway you can defend.
7. **`POST.md`.**

## Files so far

- `mnist.py`, `cifar10.py`: download once, return raw uint8 NHWC arrays and int32 labels.
- `data.py`: MNIST / CIFAR-10 -> standardized float32 NHWC -> `ArrayDataset`.
- `model.py`: `ModelConfig`, `build_model`, `forward` (harness wiring) around a
  placeholder `CNN` and `loss_fn` for you to fill in.
- `metrics.py`: `Accuracy` for `Task(metrics=[...])`.
- `01_data_and_pipeline.ipynb`: data, shape trace, dummy passes, a short
  `Experiment` run, checkpoint restore. Set `DATASET = "cifar10"` for CIFAR-10.
- `pipeline_test.py`: the same wiring on synthetic data. `datasets_test.py`: the loaders.

## Commands

```bash
uv sync --extra posts
uv run python -m toylib_projects.p01_cnn_mnist.train
uv run python -m toylib_projects.p01_cnn_mnist.figures
uv run pytest toylib_projects/p01_cnn_mnist
```
