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
- `model.py`: `ModelConfig`, `build_model`, `forward` (harness wiring) around the
  `CNN` and `loss_fn`.
- `metrics.py`: `Accuracy` for `Task(metrics=[...])`.
- `01_data_and_pipeline.ipynb`: MNIST data, shape trace, overfitting one batch,
  full training through `Experiment`, and test-set results from the checkpoint.
- `01_data_and_pipeline_cifar10.ipynb`: the same notebook for CIFAR-10.
- `resnet.py`: residual networks for CIFAR-10, kept apart from `model.py`.
- `01_resnets_colab.ipynb`: the heavier ResNets on a Colab TPU, with metrics on W&B,
  checkpoints in GCS, and resuming after a lost runtime. `COLAB.md` shows how to run
  it with the Colab CLI.
- `explorer/`: an interactive page around the trained CNN, with a real forward pass
  in the browser. `explorer/build_data.py` regenerates its `data.js`; see
  `EXPLAINER_PLAN.md` for the plan.
- `pipeline_test.py`: the same wiring on synthetic data. `model_test.py`: the
  model and loss. `resnet_test.py`: the ResNet. `datasets_test.py`: the loaders.

## Commands

```bash
uv sync --extra posts
uv run python -m toylib_projects.p01_cnn_mnist.train
uv run python -m toylib_projects.p01_cnn_mnist.figures
uv run pytest toylib_projects/p01_cnn_mnist
```
