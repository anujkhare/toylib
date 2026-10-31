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

One photo, carried through the whole page. Each section answers the question the
last one raised.

1. **The prediction.** Pick a test photo, or upload your own. The network says
   "cat, 61%". What happened in between? Show the architecture with real shapes:
   32x32x3 → 16x16x32 → 8x8x64 → 8x8x64 → 4,096 → 10.
2. **An image is numbers.** The red, green and blue grids, and the standardization
   the model sees.
3. **One convolution by hand.** Hover a pixel to see the 3x3 window, the filter and
   the sum that gives one output number. Try edge, blur and sharpen filters.
4. **The filters it learned.** All 32 first-layer filters, with a slider through
   training from noise to edges and colour blobs.
5. **Activations.** Every channel at every layer for your photo. ReLU and pooling
   each get a toggle: see what they remove and why pooling tolerates small shifts.
6. **What one unit can see.** Click a unit and highlight its receptive field,
   growing 3 → 4 → 8 → 10 → 18 pixels. Compare it with the effective field.
7. **What a channel wants.** The test patches that excite it most, beside its
   feature visualization, with a note on how far to trust the latter.
8. **The decision.** The flatten head is linear, so each position's contribution
   to a class score can be computed exactly and summed. That gives an evidence map
   with no approximation. Toggle weights from an early checkpoint as the sanity
   check. Paint an occluder on the photo and watch the scores move.
9. **Shift the photo.** The post's question: a slider moves the image and the
   prediction updates live, next to accuracy against shift for the whole test set.
10. **Where it fails.** The confusion matrix, the most confident mistakes, and a
    2D map of test images where clicking a dot loads that photo.
11. **What depth buys.** Augmentation, weight decay, then residual blocks: the
    measured gains, and plain against residual networks of the same depth.

## How it is built

- **Live forward passes.** The CNN is conv, ReLU, pool and a linear layer, so about
  100 lines of plain JavaScript can run it on `Float32Array`s with no ML library.
  Its 97,290 parameters are under 400 KB. This also avoids a JAX → TensorFlow.js
  conversion step, which [the route that exists](https://blog.tensorflow.org/2022/08/jax-on-web-with-tensorflowjs.html)
  would need, and exposes every intermediate tensor for free.
- **Live:** prediction, activations, occlusion, evidence maps, shifts, uploads.
- **Precomputed in Python, embedded as data:** test photos, weights at several
  checkpoints, top patches, feature visualizations, effective receptive fields,
  shift curves, the confusion matrix, the 2D map, and ResNet results.
- **Parity check.** The build script stores JAX's logits for the embedded photos,
  and the page's forward pass must reproduce them. A JS bug would otherwise
  silently misrepresent the model.
- **Files.** `explorer/build_data.py` exports the data; `explorer/index.html` is
  the page.

## Order of work

1. Export script and JS forward pass with the parity check. Nothing else is
   trustworthy until this passes.
2. Sections 1, 3 and 5, the core loop of input, operation and activations.
3. Sections 4, 6 and 8, then 7, 9 and 10.
4. Section 11 once the ResNet runs are finished.
5. Check it at phone width and in both themes.

**Not doing:** 3D views, adversarial examples, loss landscapes, and backpropagation
in the browser. Each is a stretch goal that doesn't serve the core story.
