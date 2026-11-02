# Post 1: Why does a CNN work?

Build up from the simplest image classifier to a small ResNet on one dataset, and at each
step look inside to see why the next idea is needed. The aim is intuition, mostly through
visualizing the network, not a measured comparison of architectures. Every model is
Imagenette: ten classes of real photos at 160x160, big enough for feature maps to show
shapes.

## Results

| Model | Train | Test | Source |
| --- | --- | --- | --- |
| Linear | 36.8% | 35.5% | `notebooks/04_imagenette_baselines.ipynb`, 768,000 parameters |
| MLP, 256 hidden units | 63.9% | 58.1% | `notebooks/04_imagenette_baselines.ipynb`, 19.7M parameters |
| Small CNN | 99.3% | 82.9% | `notebooks/04_imagenette_baselines.ipynb`, 368,842 parameters |
| ResNet-18 shape, width 32 | 100% | 87.4% | `notebooks/05_imagenette_resnet_colab.ipynb`, 2.8M parameters |

Every model uses the same recipe: 200 epochs, the same augmentation and the same optimizer.
Train accuracy is on clean images, and test accuracy uses all 3,925 images.

CIFAR-10 work before this: the small CNN reached 81.9% and Wide ResNet-20 93.1%.
The interactive views are in `post/explorer/`.

## Files

- `mnist.py`, `cifar10.py`, `imagenette.py`: download once, return raw uint8 NHWC arrays
  and int32 labels.
- `data.py`: MNIST / CIFAR-10 -> standardized float32 NHWC -> `ArrayDataset`.
- `model.py`: `ModelConfig`, `build_model`, `forward` (harness wiring) around the
  `CNN` and `loss_fn`.
- `metrics.py`: `Accuracy` for `Task(metrics=[...])`.
- `notebooks/01_mnist.ipynb`: MNIST data, shape trace, overfitting one batch,
  full training through `Experiment`, and test-set results from the checkpoint.
- `notebooks/02_cifar10.ipynb`: the same notebook for CIFAR-10.
- `resnet.py`: residual networks for CIFAR-10, kept apart from `model.py`.
- `notebooks/04_imagenette_baselines.ipynb`: the linear model, MLP and small CNN on Imagenette.
- `post/figures.py`: the post's figures, from checkpoints in the bucket, written to `post/figures/`.
- `post/POST.md`: the post.
- `notebooks/05_imagenette_resnet_colab.ipynb`: a ResNet-18-shaped model on 160x160 Imagenette, on a Colab TPU.
- `notebooks/03_cifar10_resnet_colab.ipynb`: the final Wide ResNet-20 on a Colab TPU, with metrics on W&B,
  checkpoints in GCS, and resuming after a lost runtime. `notebooks/COLAB.md` shows how to run
  it with the Colab CLI.
- `post/explorer/`: interactive views for the post, around the trained ResNet, with a real
  forward pass in the browser. `post/explorer/build_data.py` regenerates `data.js` and `weights.js`, and
  `node post/explorer/parity_check.js` checks `cnn.js` against JAX.
- `pipeline_test.py`: the same wiring on synthetic data. `model_test.py`: the
  model and loss. `resnet_test.py`: the ResNet. `datasets_test.py`: the loaders.

## Commands

```bash
uv sync --extra posts
uv run pytest toylib_projects/p01_cnn_mnist
GOOGLE_CLOUD_PROJECT=practicalml-db49f uv run python -m toylib_projects.p01_cnn_mnist.post.figures
GOOGLE_CLOUD_PROJECT=practicalml-db49f uv run python -m toylib_projects.p01_cnn_mnist.post.explorer.build_data
```
