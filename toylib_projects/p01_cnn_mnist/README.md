# Post 1: Why does a CNN work?

Build up from the simplest image classifier to a small ResNet on one dataset, and at each
step look inside to see why the next idea is needed. The aim is intuition, mostly through
visualizing the network, not a measured comparison of architectures. Every model is
Imagenette: ten classes of real photos at 160x160, big enough for feature maps to show
shapes.

## The story

Each section ends on a problem that the next one solves. `post/POST.md` is the written version,
and `post/figures.py` rebuilds every figure.

1. **The hook.** The trained network looks at a man holding a fish and says tench, and the
   evidence sits on the fish. What happened in between?
2. **One template per class.** A linear model learns one image per class. Its templates
   are noise, and it scores 35.5% on test. Without augmentation, on 40x40 pixels, it reaches
   73% on train and only 30% on test: it can memorize photos but not learn them. A church
   has no single average appearance, and one template tied to fixed pixel positions
   can't describe one.
3. **Hidden layers.** Hidden units let an MLP combine features, and it reaches 58.1%. Its
   hidden weights are the argument for what comes next: they are real patterns, such as
   colour blobs, stripes and bars, but each spans the whole image and is fixed to one place
   in the frame. The same stripe elsewhere needs a new unit. That took 19.7M weights, seven
   times the whole final model, and more than 50 times the small CNN, which beats it.
   Moving a photo 5 pixels barely changes its accuracy, because augmentation covered that,
   but it changes 18% of its answers, against 6% for the small CNN and 3% for the ResNet.
4. **Convolution.** Share one small filter across every position. Work one by hand, then
   show that the output keeps the input's layout: a feature map.
5. **What the layers see.** One photo through the stages, 80x80 down to 10x10. This is the
   best view in the post. Church arches, horn tubing and the fish's outline are visible in
   early layers, and later ones become coarse blobs of "what", not "where". Reach grows from
   7 pixels to 219, while the gradient of a central unit stays near the middle of it.
6. **Filters are the wrong thing to look at.** The trained 7x7 filters are smoother than
   their random start, but they still don't read as pictures, while their responses in
   section 5 are clean. What a network learns is easier to read from what it does than from
   its weights.
7. **Going deep breaks things.** Stack 56 layers and the signal dies. At initialization,
   with no training, activations without normalization or skips fade about 10 orders of
   magnitude and gradients sit around 1e-10 in every block. GroupNorm alone keeps
   activations stable, but gradients grow about 50 times on the way back to the first
   block. Skips alone keep gradients level but let activations grow 16 times. With both,
   each stays within a factor of about six. Measured backup: Plain-56 reached 34% on
   CIFAR-10 in 15 epochs against 78% for ResNet-56.
8. **Where is the evidence?** Global average pooling followed by one linear layer means each
   of the 10x10 positions has an exact score per class, with no approximation. The maps
   mostly land on the object. The parachute is an exception: its evidence sits on the
   skydiver and smoke trail, not the canopy.
9. **Where it fails.** Chain saw and gas pump are the weakest classes at 78%. Gas pumps and
   cassette players confuse each other, 24 and 22 times. Both are boxes covered in small
   panels, labels and readouts, and the evidence sits around those panels. That's a
   suggestion from a handful of maps, not a measured cause.
10. **It knows too much.** 100% on clean training images, 87.4% on test, and a median
    confidence of 99.95%, or 78.5% on its mistakes. Accuracy isn't the only thing that
    memorization damages, because the model is often sure of itself when wrong. Augmentation
    is what kept it this good.

**Left out of the main story:** occlusion maps. They were hard to read, because hiding a
40x40 patch barely changes these confident predictions, and the score drops land in odd
places. Painting over a photo stays as something readers can try, not a figure that makes
a claim. Also treat "residual blocks make small edits" with care: on the trained model,
the residual path is about as large as the skip in most blocks, and over 5 times larger in
the last.

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
The interactive page is `post/explorer/`, and `post/EXPLAINER_PLAN.md` records how it was planned.

## Milestones

All done.

1. ~~**Data and the final model.**~~ Loaders, `resnet.py`, and the trained checkpoints.
2. ~~**Linear model and MLP.**~~ `model.py`, with figures from `post/figures.py`.
3. ~~**A small CNN on Imagenette.**~~ Your CNN with pooling added for 160x160.
4. ~~**`return_intermediates`.**~~ On the small CNN and the ResNet.
5. ~~**`post/figures.py`.**~~ Every figure, rebuilt from checkpoints.
6. ~~**The explorer.**~~ The Imagenette ResNet, with a live forward pass in the browser.
7. ~~**`post/POST.md`.**~~ Published on the website.

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
- `post/explorer/`: an interactive page around the trained ResNet, with a real forward pass
  in the browser. `post/explorer/build_data.py` regenerates `data.js` and `weights.js`, and
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
