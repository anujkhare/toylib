# Track A1: Vision Codec

## Overview

The vision codec is the first component we build, and everything downstream depends on it. Its job is to compress a 64×64 Breakout frame into a compact 8×8×4 latent vector and reconstruct the frame from that latent. Both directions are first-class: encoding is used at training time to cache latents and at inference time to condition on a first frame; decoding is used at inference time to render the world model's predictions into pixels.

Every downstream track — unconditional image generation (Track 1b), video diffusion (Track 2), the hybrid action model (Track 3), the full any-conditional model (Track 4) — operates entirely in this learned latent space. The codec is trained once and frozen. Its spatial grid size directly determines the transformer sequence length for all downstream tracks: 8×8 = 64 tokens per frame, so a 16-frame clip contributes 1,024 frame tokens before actions and captions are added.

<!-- VIZ: Full pipeline diagram. Left path (encoding): Stage 2 frame (B, 64, 64, 3) → Encoder → μ (B, 8, 8, 4) and log σ² (B, 8, 8, 4) → reparameterize → z (B, 8, 8, 4). Right path (decoding): z (B, 8, 8, 4) → Decoder → reconstructed frame (B, 64, 64, 3). Bottom section: show z flattened to 64 tokens feeding into the downstream world model transformer. Label all tensor shapes on every arrow. -->

## Prerequisites

**Knows:**

- Python, NumPy; basic JAX/Flax — can write a forward pass and a training step with `optax`
- Backpropagation, gradient descent, convolutional networks
- What an autoencoder is: encoder maps input to latent, decoder maps latent back to input

**Does not know:**

- What a VAE is, what KL divergence does to a latent space, or what the reparameterization trick is
- The Breakout domain in any detail

If you've completed the Stage 1 datagen work, you have the HDF5 frame data this track reads. If not, the sampler milestone (Milestone 1) describes exactly what format is needed.

---

## The Problem We're Solving

Imagine training a plain autoencoder on Breakout frames. The encoder learns to map each frame to a point in latent space, and the decoder reconstructs the frame from that point. With enough capacity, this works perfectly — you get near-pixel-perfect reconstructions.

Now try to *generate* a new frame by sampling a random latent vector `z ~ N(0, I)` and decoding it. What you get is garbage. Here is why.

A plain autoencoder has no incentive to fill the latent space smoothly. The encoder can pack all training frames into isolated islands, leaving vast voids in between:

```
Latent space (2D sketch of what a plain autoencoder learns):

         ● frame 312          ← "islands" of training data —
   ●  ●                         arbitrary isolated clusters
        ●  ●
             ●
  ← void →              ●
                    ●  frame 997
```

Any sampled `z` that lands in a void between islands decodes to noise, because the decoder has never seen those coordinates and has no constraint to handle them gracefully. The autoencoder is an excellent *compressor* but a useless *generator*.

There is a second problem specific to this project. The Breakout ball is approximately 4 pixels wide in a 64×64 frame. At 8× spatial compression, the ball maps to a region smaller than one cell in the 8×8 latent grid. A standard autoencoder trained with per-pixel L1 loss will blur the ball away — it contributes too little to the total pixel error for the encoder to be forced to preserve its position precisely.

We need a codec that is (1) a good compressor with faithful reconstructions, (2) has a smooth, samplable latent space, and (3) reliably encodes small objects like the ball. A KL-regularized VAE with auxiliary supervision gives us all three.

---

## Core Concept: KL-Regularized Variational Autoencoders

We build intuition on a 1D toy example before touching image data.

Suppose we have scalar observations `x` drawn from some distribution — say, a mixture of two Gaussians. We want a model that can encode `x` to a latent `z` and decode `z` back to `x`, *and* we want to be able to generate new samples by drawing `z ~ N(0, 1)` and decoding. How do we accomplish both goals at once?

The key insight: instead of encoding `x` to a single point `z`, encode `x` to a *distribution* over `z`. Concretely, we parameterize the encoder's output as a Gaussian: the encoder outputs a mean `μ(x)` and a standard deviation `σ(x)`, and the latent is a *sample* from `N(μ(x), σ(x)²)`. The decoder then maps this sample back to a reconstruction of `x`.

For the Breakout codec, every spatial position in the 8×8 latent grid gets its own `μ` and `σ` per latent channel. The encoder outputs two tensors, each `(B, 8, 8, 4)`.

### The reparameterization trick

To train with stochastic sampling in the forward pass, we need gradients to flow back through the sample into `μ` and `σ`. Sampling `z ~ N(μ, σ²)` is not differentiable as written — you can't backpropagate through a random draw. The reparameterization trick solves this by externalizing the randomness:

```
ε ~ N(0, 1)         ← sampled fresh each forward pass; no gradient attached
z = μ + σ · ε       ← deterministic function of μ, σ, and external ε
```

`z` is now a differentiable function of `μ` and `σ`. Gradients from the decoder loss flow through the subtraction and multiplication back into both encoder outputs.

> **Note:** The encoder outputs `log σ²`, not `σ` directly, for numerical stability. To compute `σ`, use `σ = exp(0.5 · log σ²)`. This lets the encoder emit any real number — negative outputs give `σ < 1`, large outputs give `σ > 1` — without constraining the output range.

### KL regularization

Without additional pressure, the encoder will learn to produce very tight posteriors (`σ → 0`, `z ≈ μ`) — exactly the isolated-island behavior of a plain autoencoder. The KL term is the pressure that prevents this.

We measure how far the encoder's posterior `q(z|x) = N(μ, σ²)` is from the standard Gaussian prior `p(z) = N(0, I)` using KL divergence. Because both are Gaussians, this has a closed form:

```
L_KL = KL( N(μ, σ²) || N(0, I) )
     = 0.5 · Σ_d ( μ_d² + σ_d² - log σ_d² - 1 )
```

The sum is over all latent dimensions `d` — for our codec, that is `8 × 8 × 4 = 256` per image. This term is minimized when `μ = 0` and `σ = 1` everywhere: the encoder's posterior is identical to the prior and carries no information about `x`. In practice we never train to this extreme — we use it as a *regularizer*, not a bottleneck.

The total loss balances reconstruction and regularization:

```
L_base = L_rec + β · L_KL
```

The weight `β = 1e-6` (the Stable Diffusion default). This is four to six orders of magnitude smaller than the β-VAE regime (β = 1–100) where KL is the primary constraint. Here, KL is just a gentle nudge toward `N(0, I)` that prevents gross discontinuities.

This is easy to confuse with β-VAE disentanglement. The difference: β-VAE uses large β to force a strongly bottlenecked, disentangled representation; we use tiny β to get just enough regularization for downstream diffusion to sample from `N(0, I)` and get sensible latents. The goal is not disentanglement, it is smoothness.

> **Watch out:** The leading failure mode for VAE training is **posterior collapse** — the encoder learns to ignore the input by setting `μ → 0` and `σ → 1` everywhere. The KL term then equals zero and the decoder receives pure noise. This happens when KL dominates reconstruction early in training, before the encoder has learned useful representations. We prevent it with **KL warmup**: set `β = 0` for the first 10,000 steps and ramp linearly to `β = 1e-6`. The encoder learns to reconstruct first, then is gently regularized.

---

## Architecture

We define these dimension symbols once here and use them consistently throughout:

| Symbol | Value | Meaning |
|---|---|---|
| B | varies | batch size |
| H, W | 64 | input frame height and width (px) |
| h, w | 8 | latent grid height and width |
| C | 4 | latent channels |
| ch | 64 | base channel multiplier |

### ResBlock

**Purpose.** A residual block applies two convolutions with a skip connection. It is the workhorse unit of both the encoder and decoder.

**Intuition.** Without skip connections, information from the input must survive every nonlinearity in the stack. Skip connections make the optimal "do nothing" transformation easy — if no transformation is needed, the conv weights converge to zero and the block passes the input through unchanged. This is why very deep conv stacks converge faster and more stably with residual connections.

**Math.** Given input `x (B, H', W', c)`:

```
h = GroupNorm(x)              → (B, H', W', c)
h = SiLU(h)                   → (B, H', W', c)
h = Conv2D(c → c, k=3, pad=1) → (B, H', W', c)
h = GroupNorm(h)              → (B, H', W', c)
h = SiLU(h)                   → (B, H', W', c)
h = Conv2D(c → c, k=3, pad=1) → (B, H', W', c)
out = x + h                   → (B, H', W', c)    ← skip connection
```

**Shape trace** (for c=128, spatial=32×32):

```
input:            (B, 32, 32, 128)
→ GroupNorm:      (B, 32, 32, 128)
→ SiLU:           (B, 32, 32, 128)
→ Conv2D(128→128, k=3, pad=1):  (B, 32, 32, 128)
→ GroupNorm:      (B, 32, 32, 128)
→ SiLU:           (B, 32, 32, 128)
→ Conv2D(128→128, k=3, pad=1):  (B, 32, 32, 128)
→ + skip:         (B, 32, 32, 128)    ← same spatial and channel shape
```

**Interface:**

```python
class ResBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, in_channels)
        # returns: (B, H, W, out_channels)
        ...
```

When `out_channels != x.shape[-1]`, the skip connection needs a 1×1 conv to match channels. The encoder uses this at each downsampling stage when the channel count doubles.

### AttentionBlock

**Purpose.** A single multi-head self-attention block at the encoder and decoder bottleneck. It gives every spatial position a global receptive field — each of the 64 latent positions can attend to every other.

**Intuition.** Convolutions are local: a 3×3 conv at position (i, j) only sees a 3×3 neighborhood. At 8×8 resolution this is acceptable, but the Breakout ball occupies only one or two cells. If the surrounding cells can't communicate with the ball position, the encoder may fail to propagate ball information through the full latent. One attention layer at the bottleneck costs almost nothing (64-token sequence) and gives the encoder exactly the global context it needs.

<!-- VIZ: Attention-at-bottleneck diagram. Show a 8×8 grid of cells. Highlight one cell as "ball position". Draw arrows from that cell to all 63 others, labeled "can attend to". Contrast with a 3×3 convolution overlay showing the local 9-cell neighborhood. Annotate sequence length = 64 at 8×8. -->

**Math.** The spatial feature map is flattened to a sequence before attention, then reshaped back:

```
x:   (B, 8, 8, 4ch)
→ reshape:  (B, 64, 4ch)         ← flatten spatial dims; N = h·w = 64
h = LayerNorm(x)                  (B, 64, 4ch)
Q = W_Q · h                       (B, 64, D_head · n_heads)
K = W_K · h                       (B, 64, D_head · n_heads)
V = W_V · h                       (B, 64, D_head · n_heads)
attn = softmax(Q · K^T / √D_head) (B, n_heads, 64, 64)
out = attn · V                    (B, n_heads, 64, D_head)
out = W_O · reshape(out)          (B, 64, 4ch)
→ + skip:  (B, 64, 4ch)           ← residual over sequence dim
→ reshape: (B, 8, 8, 4ch)         ← restore spatial layout
```

**Interface:**

```python
class AttentionBlock(nn.Module):
    num_heads: int = 1

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C) — spatial feature map
        # returns: (B, H, W, C)
        ...
```

> **Watch out:** The attention block receives `(B, H, W, C)` but internally reshapes to `(B, H*W, C)`. Do not forget to reshape back to `(B, H, W, C)` before returning — downstream convolutions expect the spatial layout.

### Encoder

**Purpose.** The encoder maps a normalized input frame `(B, 64, 64, 3)` to the Gaussian parameters `μ (B, 8, 8, 4)` and `log σ² (B, 8, 8, 4)` of the latent distribution.

**Intuition.** We progressively downsample spatial resolution while expanding the channel dimension, building a progressively more abstract description of the frame. At the 8×8 bottleneck we insert one attention block. A final convolution projects to `2C = 8` output channels, which we split equally into `μ` and `log σ²`.

<!-- VIZ: Encoder architecture diagram. Show a downward funnel: boxes at each stage labeled with their output shape. Spatial dims shrink left-to-right (64→32→16→8), channel count grows (3→64→128→256→256→8). Label the attention block at the bottleneck. Label the final split into μ and log σ². Color downsampling stages differently from ResBlocks. -->

**Shape trace:**

```
input:                    (B, 64, 64,   3)   ← normalized to [-1, 1]
→ Conv2D(3→ch, k=3):     (B, 64, 64,  64)   ch = 64
→ ResBlock(ch):           (B, 64, 64,  64)
→ Downsample(ch→2ch):    (B, 32, 32, 128)   ← stride-2 conv or avgpool+conv
→ ResBlock(2ch):          (B, 32, 32, 128)
→ Downsample(2ch→4ch):   (B, 16, 16, 256)
→ ResBlock(4ch):          (B, 16, 16, 256)
→ Downsample(4ch→4ch):   (B,  8,  8, 256)   ← spatial bottleneck reached
→ ResBlock(4ch):          (B,  8,  8, 256)
→ AttnBlock:              (B,  8,  8, 256)   ← global receptive field here
→ ResBlock(4ch):          (B,  8,  8, 256)
→ GroupNorm + SiLU:       (B,  8,  8, 256)
→ Conv2D(4ch→2C, k=3):   (B,  8,  8,   8)   2C = 8 (4 μ + 4 log σ²)
→ split on axis=-1:
    μ:       (B,  8,  8,   4)
    log_σ²:  (B,  8,  8,   4)
```

**Interface:**

```python
class Encoder(nn.Module):
    base_ch: int = 64
    latent_channels: int = 4

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, 3) float32 in [-1, 1]
        # returns: mu (B, h, w, C), log_sigma_sq (B, h, w, C)
        ...
```

> **Watch out:** Clamp `log_sigma_sq` to `[-30, 20]` before any `exp` operations. Unclamped early-training outputs can overflow to `NaN`, which silently propagates through the entire training step and is hard to debug after the fact. A simple `jnp.clip(log_sigma_sq, -30, 20)` before computing `σ = exp(0.5 · log σ²)` costs nothing.

### Decoder

**Purpose.** The decoder maps a latent sample `z (B, 8, 8, 4)` to a reconstructed frame `(B, 64, 64, 3)` in `[-1, 1]`.

**Intuition.** The decoder mirrors the encoder: we expand spatial resolution from 8×8 back to 64×64 while reducing the channel count. We upsample then convolve instead of using transposed convolutions. Transposed convolutions produce checkerboard artifacts — the stride pattern creates uneven pixel contributions that are nearly impossible to train away. Nearest-neighbor or bilinear upsampling followed by a regular convolution avoids this entirely.

**Shape trace:**

```
z:                        (B,  8,  8,   4)
→ Conv2D(C→4ch, k=3):    (B,  8,  8, 256)   ch = 64, 4ch = 256
→ ResBlock(4ch):          (B,  8,  8, 256)
→ AttnBlock:              (B,  8,  8, 256)   ← mirror the encoder bottleneck
→ ResBlock(4ch):          (B,  8,  8, 256)
→ Upsample(4ch→4ch):     (B, 16, 16, 256)   ← 8 → 16, bilinear + conv
→ ResBlock(4ch→2ch):      (B, 16, 16, 128)
→ Upsample(2ch→2ch):     (B, 32, 32, 128)   ← 16 → 32
→ ResBlock(2ch→ch):       (B, 32, 32,  64)
→ Upsample(ch→ch):       (B, 64, 64,  64)   ← 32 → 64
→ ResBlock(ch):           (B, 64, 64,  64)
→ GroupNorm + SiLU:       (B, 64, 64,  64)
→ Conv2D(ch→3, k=3):     (B, 64, 64,   3)
→ Tanh:                  (B, 64, 64,   3)   ← output in (-1, 1)
```

**Interface:**

```python
class Decoder(nn.Module):
    base_ch: int = 64
    latent_channels: int = 4

    @nn.compact
    def __call__(self, z):
        # z: (B, h, w, C) float32
        # returns: (B, H, W, 3) float32 in (-1, 1)
        ...
```

> **Watch out:** The `Tanh` output is in the *open* interval `(-1, 1)` — values of exactly -1 and 1 are never produced. Make sure your input frames are normalized to `[-1, 1]` using `x = (frame_uint8 / 127.5) - 1.0` (not `/ 255.0`). If the target is in `[0, 1]` and the prediction is in `(-1, 1)`, the loss gradients will push the decoder toward the wrong range and training will diverge.

### Reparameterization

This is not a trainable module — it is a pure function called between the encoder and decoder:

```python
def reparameterize(mu, log_sigma_sq, rng_key):
    # mu:           (B, h, w, C)
    # log_sigma_sq: (B, h, w, C)
    # rng_key:      JAX PRNG key
    # returns z:    (B, h, w, C)
    sigma = jnp.exp(0.5 * log_sigma_sq)        # (B, h, w, C)
    eps = jax.random.normal(rng_key, mu.shape)  # (B, h, w, C)
    return mu + sigma * eps
```

At inference time — when encoding a frame to cache its latent for the world model — use `z = mu` directly (deterministic, no noise). The stochastic version is only needed during VAE training.

---

## Training

### Loss components

We train with four loss terms. Start with the first two; add the perceptual and auxiliary losses after the base autoencoder is stable.

**Reconstruction loss** (always on):

```
L_rec = mean( |x̂ - x|₁ )
```

where `x̂ = decode(reparameterize(encode(x)))` and both `x̂` and `x` are in `[-1, 1]`. We use L1 over MSE because L1 penalizes all pixels equally — MSE's squared penalty downweights small errors, causing it to ignore small objects like the ball in favor of large uniform regions like the brick background.

**KL regularization** (always on):

```
L_KL = 0.5 · mean_B( sum_{h,w,C}( μ² + σ² - log σ² - 1 ) )
```

Shape note: `μ` and `log_σ²` are `(B, 8, 8, 4)`. Sum over the `(8, 8, 4) = 256` latent dimensions *first*, then mean over `B`.

> **Watch out:** Whether you `sum` or `mean` over the latent dimensions changes the effective scale of `L_KL` by a factor of 256. If `L_KL` is unexpectedly dominating `L_rec`, check this first. The convention we follow (matching Stable Diffusion): sum over spatial and channel dims, mean over batch.

**Perceptual loss** (add after base training is stable, around step 5k):

```
L_perc = Σ_l || φ_l(x̂) - φ_l(x) ||₂²
```

where `φ_l` extracts intermediate features from a frozen VGG-16 at layer `l`. We use only the early layers — `relu1_2` and `relu2_2` — which respond to edges and low-level textures. These features transfer well from ImageNet to Atari frames. VGG's later layers respond to semantic categories that don't exist in Atari, so using `relu4_3` or `relu5_3` would penalize visually correct reconstructions that happen to not look like dogs or cars.

**Auxiliary ball supervision** (add at the same time as perceptual loss):

```
L_aux = λ_aux · MSE( MLP( mean_pool(z) ), [ball_x_norm, ball_y_norm, paddle_x_norm] )
```

We spatially mean-pool the `(B, 8, 8, 4)` latent to `(B, 4)`, then pass it through a tiny 2-layer MLP (~500 parameters total) that predicts the three normalized coordinates. Ground truth comes from the Stage 1 HDF5 RAM state arrays — no additional annotation needed. Coordinates are normalized to `[0, 1]` by dividing by frame dimensions.

This loss is the key mitigation for the small-object problem. Without it, the encoder has no direct gradient signal for ball position — the ball is too small to meaningfully affect `L_rec`. With it, the encoder is explicitly penalized for any latent that doesn't encode where the ball is.

> **Alternative:** A spatially-weighted reconstruction loss (`L_rec_weighted = mask · |x̂ - x|₁`, where `mask` is 5.0 in the ball bounding box and 1.0 elsewhere) can be added on top of `L_aux`. It provides a complementary gradient signal — forcing the decoder to be more accurate in the ball region, rather than just forcing the encoder to encode ball position. Worth adding if the ball detection rate is above 80% but below 95% after training with `L_aux` alone.

**Total loss:**

```
L = L_rec + β · L_KL + λ_perc · L_perc + λ_aux · L_aux
```

Starting hyperparameters: `β = 1e-6` (after warmup), `λ_perc = 0.1`, `λ_aux = 0.1`. Tune on the validation set before committing to a full run.

### Optimizer

We use Adam with a fixed learning rate of `1e-4` and no weight decay. No learning rate schedule is needed for a VAE at this scale — the loss surfaces are smooth enough that a fixed lr works well. Gradient clipping to global norm 1.0 is a good safeguard against early-training instability.

### β warmup schedule

For steps 0 to `warmup_steps = 10_000`: `β(t) = (t / warmup_steps) · 1e-6`.
After step 10,000: `β = 1e-6`.

Implement this as a function of the current training step, called at the start of each training step to compute the current effective `β`.

### What to log

Every 100 steps: `L_rec`, `L_KL` (unweighted and weighted by current β), `L_perc`, `L_aux`, current β, gradient norm.

Every 500 steps on the validation set: SSIM, LPIPS, ball detection rate (run the classical detector on reconstructed frames and count the fraction where the ball is found within 5px of ground truth), a grid of 8 input/reconstruction pairs.

### What healthy training looks like

- **Steps 0–500:** `L_rec` is high (> 0.4). Expected — the network is randomly initialized.
- **Steps 500–5k:** `L_rec` drops sharply toward 0.05–0.1. `L_KL` is near zero (β = 0 during warmup). Reconstructions start to show brick structure.
- **Steps 5k–15k:** `L_rec` stabilizes. `L_KL` begins to increase as β ramps up, then plateaus. Ball detection rate climbs above 50%.
- **Steps 15k–50k:** Both losses stable. Ball detection rate should cross 80% with `L_aux` active and reach 95%+ by the end.

If `L_KL` grows to be much larger than `L_rec` at any point during warmup, the β ramp is too aggressive. Halve the ramp rate and restart from the last checkpoint.

If the ball detection rate is stuck below 50% after 30k steps despite `L_aux` being active, check that the auxiliary head's gradient is actually non-zero and that ball coordinates in the HDF5 files are sensible (a common data loading bug is reading the wrong state array column).

---

## Implementation Milestones

### Milestone 1: Stratified Frame Sampler

**What to build:** A dataset class that reads from Stage 1 HDF5 shards and yields batches of `(frame, ball_x, ball_y, paddle_x)` tuples, stratified across game state.

**Success criteria:** The sampler yields frame batches of shape `(B, 64, 64, 3)` float32 in `[-1, 1]`. Visualizing 64 sampled frames from a single training epoch shows a visible mix of early-game (full brick wall), mid-game, and late-game (sparse or empty bricks) states — no single state dominates. Plotting histograms of `ball_x` and `ball_y` across 1,000 sampled batches shows roughly uniform coverage across both axes (not clustered near the paddle or walls from a single game phase).

---

### Milestone 2: Encoder

**What to build:** The `Encoder` module: initial convolution, three downsampling stages with ResBlocks, bottleneck attention, final projection to `(μ, log σ²)`.

**Success criteria:** Given random input `(2, 64, 64, 3)`, the encoder outputs `mu` and `log_sigma_sq` each of shape `(2, 8, 8, 4)`. Passing the same input twice with the same model parameters produces identical outputs (the encoder is deterministic — no randomness here). Total encoder parameter count is approximately 3M (verify with a parameter count before proceeding to avoid building the wrong model size).

---

### Milestone 3: Decoder

**What to build:** The `Decoder` module: initial conv, bottleneck attention, three upsampling stages with ResBlocks, final conv + Tanh.

**Success criteria:** Given random input `(2, 8, 8, 4)`, the decoder outputs shape `(2, 64, 64, 3)` with all values in `(-1, 1)`. End-to-end shape test: encode a real frame to get `μ`, then pass `μ` (no reparameterization) through the decoder and confirm the output shape is `(1, 64, 64, 3)`. The pixel values will look like noise at initialization — the point is confirming that the entire shape chain works before writing the training loop. Decoder parameter count approximately 3M.

---

### Milestone 4: Reparameterize + Base Loss (Overfit Test)

**What to build:** The full VAE forward pass (encode → reparameterize → decode), the `L_rec + β · L_KL` loss, the β warmup schedule, and a training loop. Run a 300-step overfit on a single fixed batch of 8 frames.

**Success criteria:** `L_rec` decreases monotonically over the 300 steps and reaches below 0.05 on the held-in batch. At step 300, reconstructions of those 8 frames are visually recognizable as Breakout frames — brick rows and the paddle are visible, background color is correct. The ball may still be blurry or missing. Inspect the `μ` tensor after 300 steps: most values should be non-zero (if `μ ≈ 0` everywhere, posterior collapse occurred — reduce the initial β or extend the warmup).

---

### Milestone 5: Perceptual + Auxiliary Loss

**What to build:** The VGG-16 perceptual loss using `relu1_2` and `relu2_2` feature layers, and the auxiliary ball/paddle head (2-layer MLP, ~500 parameters, operating on spatial mean of `z`).

**Success criteria:** Load the Milestone 4 checkpoint and add `L_perc` (λ=0.1) and `L_aux` (λ=0.1) to the training loss. Train for 1,000 more steps. The reconstruction sharpness should improve visually relative to M4 — brick edges more defined, paddle less blurry. The auxiliary loss `L_aux` decreases over the 1,000 steps, confirming the MLP is learning. On a held-out validation batch of 100 frames, the ball zone classification accuracy (6 spatial buckets) using the auxiliary head's predictions exceeds 50% (random baseline is ~17%).

---

### Milestone 6: Full Training Run + Stage-Gate Evaluation

**What to build:** A complete training run on the full stratified dataset (50k–100k steps), plus the evaluation pipeline: classical ball and paddle position detectors, SSIM and LPIPS computation, latent channel utilization analysis, ball linear probe, and a latent interpolation visualizer.

**Success criteria:** All stage-gate criteria from the design doc are met before proceeding to Track 1b:

| Criterion | Target |
|---|---|
| Ball detection rate in reconstructions | > 95% of test frames |
| Ball position MSE | < 9 px² (≤ 3 px average error on a 64 px frame) |
| Paddle position MSE | < 4 px² (≤ 2 px average error) |
| SSIM on test set | > 0.85 |
| KL per channel | No channel with KL < 0.01 nats (no dead channels) |
| Latent interpolation | Qualitatively smooth across 5 spot-checked pairs |
| Ball linear probe accuracy | > 80% zone classification on held-out test set |

The interpolation check: for 5 pairs of test frames, linearly interpolate between their `μ` values at α ∈ {0, 0.25, 0.5, 0.75, 1.0} and decode each. View as a 5-frame sequence. A well-trained codec shows the paddle sliding smoothly between positions, bricks fading in/out gradually, and no discontinuous jumps. Abrupt transitions indicate the latent manifold has holes.

---

## Further Reading

- **Kingma & Welling (2014) — Auto-Encoding Variational Bayes (arXiv:1312.6114):** The original VAE paper. Sections 2 and 3 cover the ELBO derivation, the reparameterization trick, and the KL closed form we use. The rest of the paper is optional for this project.

- **Rombach et al. (2022) — Latent Diffusion Models (arXiv:2112.10752):** The Stable Diffusion paper. Appendix A describes the KL-VAE architecture we follow. Their `base_ch=128` vs. our `base_ch=64` is the main scaling difference — Breakout needs far less capacity than natural images.

- **SD VAE reference implementation** (`CompVis/stable-diffusion`, `ldm/modules/diffusionmodules/model.py`): The `Encoder` and `Decoder` classes in PyTorch. Compare your architecture against this once Milestone 3 shapes are verified.

- **Johnson, Alahi & Fei-Fei (2016) — Perceptual Losses (arXiv:1603.08155):** The paper that established using VGG features as a training loss. Section 3.1 shows why feature-space distances outperform pixel MSE for reconstruction quality — and why early layers work better than late layers for structure preservation.

- **Zhang et al. (2018) — LPIPS (arXiv:1801.03924):** The learned perceptual image patch similarity metric. We use LPIPS as an evaluation metric (not a training loss) to measure reconstruction quality in a way that correlates with human judgment.

- **Higgins et al. (2017) — β-VAE (ICLR 2017):** Useful context for what β controls and what the latent space looks like at large β values. Our β=1e-6 is the opposite end of the spectrum from the β=10–100 range explored in this paper.

- **Vision codec design doc** (`docs/designs/vision_codec.md`): The source of record for architecture options, the small-object problem analysis, and the stage-gate criteria. This walkthrough commits to one path; the design doc explains why other paths (VQ-VAE, FSQ, explicit ball state) were set aside.
