# Post 1: Why does a CNN work?

Build up from the simplest image classifier to a small ResNet on one dataset, and at each
step look inside to see why the next idea is needed. The aim is intuition, mostly through
visualizing the network, not a measured comparison of architectures. Every model is
Imagenette: ten classes of real photos at 160x160, big enough for feature maps to show
shapes.

## The story

Each section ends on a problem that the next one solves. Results marked "tried" come from
exploratory runs on the trained model, and those figures still need writing up properly.

1. **The hook.** The trained network looks at a rat holding a French horn and says French
   horn. What happened in between?
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
   Add what a small shift does to its output.
4. **Convolution.** Share one small filter across every position. Work one by hand, then
   show that the output keeps the input's layout: a feature map.
5. **What the layers see.** One photo through the stages, 80x80 down to 10x10. *Tried:* this
   is the best view in the post. Church arches, horn tubing and the fish's outline are
   visible in early layers, and later ones become coarse blobs of "what", not "where".
   Add the receptive field, to show why deeper units can respond to whole objects.
6. **Filters are the wrong thing to look at.** *Tried:* the trained 7x7 filters still look
   noisy, while their responses in section 5 are clean. What a network learns is easier to
   read from what it does than from its weights.
7. **Going deep breaks things.** Stack 56 layers and the signal dies. *Tried, with no
   training needed:* at initialization, without normalization or skips, activations fade
   about 9 orders of magnitude and gradients are around 1e-10. GroupNorm alone keeps
   activations stable, but gradients still fade about 100 times on the way back. Skips keep
   gradients strong. Measured backup: Plain-56 reached 34% on CIFAR-10 in 15 epochs against
   78% for ResNet-56.
8. **Where is the evidence?** Global average pooling followed by one linear layer means each
   of the 10x10 positions has an exact score per class, with no approximation. *Tried:*
   the maps land on the object, and they explain mistakes.
9. **Where it fails.** *Tried:* chain saw and gas pump are the weakest classes at 78%. Gas
   pumps and cassette players confuse each other, 24 and 22 times, and the evidence maps
   suggest why: both have digital number displays.
10. **It knows too much.** 100% on clean training images, 87.4% on test, and a median
    confidence of 99.95%. Accuracy isn't the only thing that memorization damages, because
    the model is sure of itself even when wrong. Augmentation is what kept it this good.

**Left out of the main story:** occlusion maps. *Tried:* they were hard to read, because
hiding a 40x40 patch barely changes these confident predictions, and the score drops land
in odd places. Keep painting over a photo as something readers can try, not as a figure
that makes a claim. Also treat "residual blocks make small edits" with care: *tried* on
the trained model, the residual path is about as large as the skip in most blocks, and
over 5 times larger in the last.

## Results so far

| Model | Train | Test | Source |
| --- | --- | --- | --- |
| Linear | 36.8% | 35.5% | `01_imagenette_baselines.ipynb`, 768,000 parameters |
| MLP, 256 hidden units | 63.9% | 58.1% | `01_imagenette_baselines.ipynb`, 19.7M parameters |
| Small CNN | 99.3% | 82.9% | `01_imagenette_baselines.ipynb`, 368,842 parameters |
| ResNet-18 shape, width 32 | 100% | 87.4% | `01_imagenette_colab.ipynb`, 2.8M parameters |

Every model uses the same recipe: 200 epochs, the same augmentation and the same optimizer.
Train accuracy is on clean images, and test accuracy uses all 3,925 images.

CIFAR-10 work before this: the small CNN reached 81.9% and Wide ResNet-20 93.1%.
`EXPLAINER_PLAN.md` covers the interactive page.

## Milestones

Each is one session. You write the models and training code; ask for review, tests or
debugging. Figure code is fair game for help.

1. ~~**Data and the final model.**~~ Done: loaders, `resnet.py`, and the trained checkpoints.
2. ~~**Linear model and MLP**~~ Done: `model.py`, with figures from `figures.py`.
3. ~~**A small CNN on Imagenette**~~ Done: your CNN with pooling added for 160x160.
4. ~~**`return_intermediates`**~~ Done for the small CNN.
5. **`figures.py`** with the views in sections 5, 7, 8 and 9. *Done when:* each figure has a
   one-sentence takeaway you can defend.
6. **The explorer** on the Imagenette model, per `EXPLAINER_PLAN.md`.
7. **`POST.md`.**

## Files so far

- `mnist.py`, `cifar10.py`, `imagenette.py`: download once, return raw uint8 NHWC arrays
  and int32 labels.
- `data.py`: MNIST / CIFAR-10 -> standardized float32 NHWC -> `ArrayDataset`.
- `model.py`: `ModelConfig`, `build_model`, `forward` (harness wiring) around the
  `CNN` and `loss_fn`.
- `metrics.py`: `Accuracy` for `Task(metrics=[...])`.
- `01_data_and_pipeline.ipynb`: MNIST data, shape trace, overfitting one batch,
  full training through `Experiment`, and test-set results from the checkpoint.
- `01_data_and_pipeline_cifar10.ipynb`: the same notebook for CIFAR-10.
- `resnet.py`: residual networks for CIFAR-10, kept apart from `model.py`.
- `01_imagenette_baselines.ipynb`: the linear model, MLP and small CNN on Imagenette.
- `figures.py`: the post's figures, from checkpoints in the bucket.
- `01_imagenette_colab.ipynb`: a ResNet-18-shaped model on 160x160 Imagenette, on a Colab TPU.
- `01_resnets_colab.ipynb`: the final Wide ResNet-20 on a Colab TPU, with metrics on W&B,
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
uv run pytest toylib_projects/p01_cnn_mnist
```
