# Explainer plan: inside a small CNN

The goal is a page where a reader starts with one photo and follows it through the
network, ending with an understanding of what each component does and where the
model fails. Everything shown should be the real trained model, and wherever it
is cheap, a real forward pass running in the reader's browser.

## What people have built

| Work | Technique | What to take from it |
| --- | --- | --- |
| [CNN Explainer](https://poloclub.github.io/cnn-explainer) (Wang et al., 2020) | Moves between the whole network and the arithmetic of one operation, in the browser | The core interaction: click a unit, see the numbers that made it. A user study found it helps people understand CNNs. |
| [Deep Visualization Toolbox](https://arxiv.org/abs/1506.06579) (Yosinski et al., 2015) | Live activations for every layer as the input changes | Change the input and watch the network respond. Their claim is that this builds intuition directly. |
| [Visualizing and Understanding ConvNets](https://arxiv.org/abs/1311.2901) (Zeiler & Fergus, 2013) | Top-activating patches; occlusion sensitivity | Both are cheap and honest: real data, no optimization artifacts. Occlusion is just repeated forward passes. |
| [Feature Visualization](https://research.google/pubs/feature-visualization/) (Olah et al., 2017) | Optimize an input to excite one unit | Striking, but needs regularization to work. [A later study](https://catalyzex.com/paper/how-well-do-feature-visualizations-support) questioned how much these images help people predict a unit's behaviour, so pair them with real patches. |
| [Activation Atlas](https://distill.pub/activation-atlas/) (Carter et al., 2019) | A map of the feature space, built from millions of activations | The small version is a 2D map of test images in the model's last hidden layer. |
| [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292) (Adebayo et al., 2018) | Compare an explanation on trained and random weights | Gradients and Grad-CAM pass; Guided Backprop does not. An explanation must change when the weights do, so show that. |
| [Effective Receptive Field](https://arxiv.org/abs/1701.04128) (Luo et al., 2016) | Gradient of one unit with respect to the input | The theoretical field overstates what a unit actually uses, and the effective field is roughly Gaussian. |
| [Visualizing the Loss Landscape](https://arxiv.org/abs/1712.09913) (Li et al., 2018) | 2D slices of the loss around a solution | The standard picture for why skip connections help. Costly, so a stretch goal. |
| [TensorSpace](https://github.com/BoTime/tensorspace), [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/) | 3D layer views; training in the browser | 3D looks good but hides the numbers. Skipping it: flat small multiples read better. |

## The story

The narrative is now the post's: see "The story" in `README.md`. The published page still
follows the earlier CIFAR-10 story, and it needs moving to the Imagenette model.

## How it is built

The Imagenette model is about 100 times more work per photo than the small CNN: about 1
billion multiply-adds against 11 million. Scaling from today's 4 ms suggests about 0.4 s per
photo in plain JavaScript, which is an estimate, not a measurement. That is too slow for
sliders, so the page splits into two kinds of view.

- **Cached, instant:** activations for about 30 chosen photos across the stages, weights at
  a few checkpoints, and the depth curves. The evidence map stays live for these photos,
  because the head is one linear layer over cached 10x10 activations.
- **Live, on demand:** one forward pass in a Web Worker when a reader uploads or edits a
  photo, with a progress state. It needs GroupNorm, strided convolutions and skips added to
  `cnn.js`, and it keeps the parity check against JAX.
- **Size:** one set of weights is 11 MB, or 15 MB as base64, which is close to the page's
  16 MB limit per file. Ship one snapshot as a binary file.
- **Measure first:** time the JavaScript pass before building around it. If it's much
  slower than 0.4 s, drop live uploads rather than adding a framework.

## Order of work

1. Port the forward pass and pass the parity check, with timing.
2. Export the cached photos and rebuild the activation view.
3. Add the evidence map, the depth curves and the mistakes view.
4. Add upload and painting if timing allows.
5. Check it at phone width and in both themes.

**Not doing:** 3D views, adversarial examples, loss landscapes, and backpropagation
in the browser. Each is a stretch goal that doesn't serve the core story.
