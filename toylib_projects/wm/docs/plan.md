# Plan: Text + Action-Conditioned Video World Model in JAX (v3)

## Introduction

Build a small `world model` from scratch. The term world model can mean many things. We target a video, action, text model where we can model the following tasks such as (but not limited to):

- text -> image / video (text-conditioned generation)
- image/video -> text (classification / captioning)
- text + actions -> image / video (text and action conditioned generation)

For this project, we focus purely on synthetic 2D-physics domain where *every* component is built and trained by us without any black boxes.

## Objectives

- **Primary — learning.** Implement and understand every loss, tokenizer, and conditioning mechanism.
- **Secondary — a working artifact.** A complete action-conditioned world model in the synthetic domain.

## Dataset Selection

### Considerations

The dataset/task is chosen with the following objectives:

1. **Low modeling complexity:** easy to model without a large compute budget (e.g.: deterministic environments, low visual complexity, discrete action space, immediate action-state feedback, etc.)
2. **Data collection ease:** easy to generate a large dataset with required modalities (image, action, text, ground-truth world state) in reasonable time/resources (e.g.: reasonable generation throughput on CPU, controllable scenario generation, etc.)
3. **Non-trivial dynamics:** data should have enough complexity to demonstrate some non-trivial behavior of the world model (e.g.: non-linear dynamics, multi-agent interactions, long-term state persistence, etc.)

### Options considered

1. **Custom 2D Physics Shapes (Bouncing Circles/Squares):**
   - *Pros:* Complete control over ground-truth coordinates, simple rendering.
   - *Cons:* High development time sink. Visuals can feel too trivial.
2. **Gymnasium Atari (e.g., Breakout, Pong) [RECOMMENDED]:**
   - *Pros:* **Zero-code simulation overhead** (`pip install gymnasium`). Captures rich mechanics (ball collisions, brick destruction, UI scores). Directly matches the NeurIPS-grade **DIAMOND** world-model benchmark. Looks incredibly impressive as a final playable world model.
   - *Cons:* Captions must be procedurally generated from RAM/game-state variables (though simple).
3. **Controlled Moving MNIST:**
   - *Pros:* Excellent middle ground. Lightweight, pure Python, download via PyTorch/JAX. Physics are simple (2D particle force vectors applied per-timestep). Perfect for learning stroke/curve representations.
   - *Cons:* Lacks color diversity and advanced game state dynamics.
4. **Crafter (2D Minecraft-like Gridworld):**
   - *Pros:* Highly complex rules (inventory, health, crafting, day/night cycles).
   - *Cons:* Tile textures are complex for a toy 2D VAE; requires slightly more setup.
5. **OpenAI Procgen (e.g., CoinRun, FruitBot):**
   - *Pros:* Procedurally generated levels, excellent for testing visual generalization to unseen backgrounds/layouts. High-contrast, clean 2D graphics.
   - *Cons:* High visual variety makes VAE reconstruction significantly harder for small models. RAM states are not standardized, making state variable/coordinate extraction difficult.
6. **VizDoom (First-Person 3D Action):**
   - *Pros:* Zero-code simulator setup; highly impressive first-person 3D action rollouts.
   - *Cons:* 3D first-person projection introduces partial observability, dynamic scale changes, and high-frequency visual shifts during camera rotation. Highly likely to exceed the capacity of a ~50M parameter model on a small budget.

**Note on Real-Video Datasets (e.g., YouTube, Robotics):**
We explicitly decided against using real-video-based datasets (such as robotics manipulation logs or curated YouTube clips). Real video has infinite visual entropy (reflections, shadows, moving backgrounds, camera motion) that would require a massive, pre-trained black-box VAE to compress cleanly, violating our "built from scratch" objective. Furthermore, real videos do not come with action labels; estimating actions would require training or running a complex Inverse Dynamics Model (IDM), introducing severe engineering overhead and noise that would distract from the core flow-matching and temporal attention mechanics.

### Decision

We choose **Gymnasium Atari (Breakout)** as the primary target. Atari gives us the most interesting physics and game state transitions while requiring zero custom simulator code.

### Dataset Generation & Specifications

To train our action-conditioned world model, we generate a high-quality local dataset from **Gymnasium Atari (Breakout)** using a custom mixed-competency exploration script.

For a full specification of the emulator configuration, state RAM extraction, visual preprocessing pipelines, and linguistic templates, see the separate [dataset.md](dataset.md).

#### Core Dataset Pool Requirements

- **Total Scale (Stage 1 Master Pool):** 1,000 complete gameplay episodes (approx. 1.5 million frames total) stored as continuous raw trajectories.

- **Storage Crop:** $160\times 160$ RGB cropped play area (preserves square aspect ratio, discards scoreboards and borders).
- **Two-Stage Compiler Pipeline:** Downstream parameters—including target image resolution ($64\times 64$ or $128\times 128$), sequence horizon length ($T=16$ or $T=25$), temporal downsampling rates (e.g. 5Hz, 10Hz, 15Hz), caption rendering, and **offline VAE latent caching**—are compiled offline into highly optimized, model-coupled training shards.
- **Exploration Strategy:** A custom $\epsilon$-greedy agent ($80\%$ competent tracking, $15\%$ random jitter, $5\%$ deliberate misses) to prevent Out-of-Distribution failures during interactive rollout.

#### Saved Episode Record Format (Stage 1 Master)

The raw dataset is stored in sharded **HDF5** files (`h5py`), one episode per group. Each group contains: `frames` `(L, 210, 160, 3)` uint8 LZ4-compressed, `actions` `(L,)` int32, and a `states/` subgroup of parallel float32/int32 arrays (`paddle_x`, `ball_x`, `ball_y`, `score`, `bricks_remaining`, `lives`). Compiled Stage 2 shards extend this with a `caption` string dataset and optional `latents`. See [dataset.md](dataset.md) for the full schema.

---

## Implementation Tracks

Four stages, each isolating one new mechanism. Models are deliberately small: the domain is simple, and a larger model just memorizes.

## Track 1: Vision encoder

**Task.** `caption → single frame` (64×64 RGB). Diffusion runs directly in pixel space — no VAE — so the diffusion mechanics are validated before any tokenizer exists.

**Dataset.** ~100–200k single Breakout gameplay frames. Captions auto-generated from game state variables / RAM (e.g. "Atari Breakout: paddle at center, ball moving up-left, 18 bricks remaining"), with a **mandatory LLM-paraphrasing pass** for linguistic diversity — without it, templated captions make text conditioning a lookup table.

**Modeling choices considered.**

- Pixel vs. latent space: pixel chosen *for this stage only*, to isolate diffusion from VAE confounds.
- Backbone: DiT over UNet — matches the rest of the project and is simpler to inflate later.
- Loss: rectified-flow / flow matching (velocity field) over DDPM ε-prediction — fewer schedule hyperparameters, modern default.
- Attention: standard softmax. Linear attention (Sana) is rejected here — at 16×16–32×32 token grids it has no efficiency advantage and trains worse; kept only as an optional ablation.
- Positional encoding: 2D sinusoidal/learned; NoPE noted as an ablation.

**Implementation plan.**

1. Atari Breakout episode generator (Gymnasium) + RAM-based caption generator + paraphrasing pass.
2. Spatial DiT in Flax NNX (~10–20M params): patchify → N transformer blocks → AdaLN-zero timestep conditioning → text cross-attention.
3. Flow-matching training loop: EMA, bf16 compute / fp32 master, gradient clipping.
4. Euler sampler with classifier-free guidance on text.
5. Smoke-test on Moving-MNIST / CIFAR before the synthetic set.

**Eval & gate.** Physics-grounded: detect paddle, ball, and remaining bricks in samples, score paddle and ball positioning accuracy vs. the caption. FID reported only as a rough secondary proxy (Inception is out-of-distribution on this domain). Gate: samples clearly satisfy captions; CFG visibly improves adherence.

## A2 — Add a from-scratch VAE → latent text → image

**Task.** Same `caption → frame`, but diffusion now runs in a *learned latent space*. Introduces latent diffusion and forces us to understand the latent space the model lives in.

**Dataset.** Same Breakout gameplay frames.

**Modeling choices considered.**

- KL-VAE (continuous) vs. VQ-VAE (discrete): continuous chosen — better reconstruction, matches Theme B's continuous tokenizers; VQ noted as an ablation.
- Compression ratio: study 4× vs. 8× spatial; pick the smallest latent grid that still reconstructs ball/paddle cleanly.
- Per-frame 2D VAE this stage; 3D temporal VAE deferred to A3.

**Implementation plan.**

1. Small 2D convolutional KL-VAE; train to reconstruct Breakout frames; visualize reconstructions and latent channels.
2. Re-target the A1 DiT to operate on cached VAE latents.
3. Compare latent-space vs. A1 pixel-space samples on the physics metrics.

**Gate.** VAE reconstruction error below a set threshold; latent-space sample quality matches A1.

## A3 — Text → video

**Task.** `caption → 16-frame clip` (128×128, ~8 fps).

**Dataset.** 50–200k Breakout gameplay clips containing ball bounces, paddle movements, and brick destructions. Joint image/video batches (~30% single-frame, the Latte recipe).

**Modeling choices considered.**

- VAE: extend the A2 VAE to a **3D causal VAE** (temporal compression 4×, spatial 8×). A 16×128×128 clip → **4×16×16×C** latent (note: 16×16 spatial — not 32 — at 128px with 8× compression).
- Temporal attention: **bidirectional softmax** chosen — the whole 16-frame clip is denoised jointly. Causal attention is rejected as the default; it belongs with autoregressive rollout and is reserved for the diffusion-forcing stretch.
- Spatial/temporal structure: factorized, interleaved spatial-then-temporal blocks (Latte).
- Build temporal from scratch vs. inflate from A2: **inflate** — initialize spatial blocks from A2, zero-init temporal output projections (AnimateDiff), train spatial layers at 1/10 LR (low-LR fine-tune, not frozen).

**Implementation plan.**

1. 3D causal VAE; re-encode the clip dataset; cache latents.
2. Temporal attention blocks (temporal RoPE) interleaved into the A2 DiT.
3. Inflation: load A2 spatial weights, zero-init temporal projections.
4. Video sampler with temporal CFG.

**Eval & gate.** Primary metric: trajectory consistency — extract per-frame paddle/ball positions, compare to actual game-physics trajectory. FVD as a rough proxy only. Gate: temporally coherent clips with plausible motion that follow captions.

## A4 — Action-conditioned world model

**Task.** `(caption, first frame, action sequence) → video`. This is the world model.

**Dataset.** Breakout clips where the paddle is controlled by discrete actions `{NOOP, FIRE, RIGHT, LEFT}`; ground-truth paddle/ball positions recorded. **Held-out split on action sequences and initial configurations** — not just held-out clips — so generalization (physics vs. memorization) is genuinely tested.

**Modeling choices considered.**

- First-frame conditioning: latent concatenation along the temporal axis (CogVideoX-I2V / Cosmos-V2W).
- Action conditioning: per-frame action embedding → **per-frame AdaLN modulation**, vs. an extra cross-attention stream. The action is a *sequence* (one per frame), so conditioning must be per temporal position, not a single global vector.
- Action-CFG: drop actions during training, apply guidance at sampling.

**Implementation plan.**

1. First-frame latent-concat conditioning.
2. Action embedding + per-frame AdaLN.
3. Action-CFG dropout.
4. Controllability metric: fix caption + first frame, vary the action sequence, measure paddle movement correlation with action inputs on the held-out action split.

**Gate (and Theme A exit criterion).** Same first frame + different action sequences produce visibly different, action-consistent continuations; controlled-paddle displacement correlates with the commanded action above a set threshold **on the held-out action split**.

---

# Cross-cutting notes

## Evaluation philosophy

The synthetic domain's killer feature is ground-truth state. Primary metrics are physics-grounded — object-position MSE, collision-timing error, action-response accuracy — computed by detecting shapes in samples and comparing to ground truth. FID/FVD are reported only as rough secondary proxies; their feature extractors (Inception, I3D) are out-of-distribution on synthetic shapes and noisy even on real video.

## Prior art drawn on

Action-conditioned / world models: **DIAMOND** (50M diffusion world model on Atari — our closest analogue), **GameNGen** (real-time Doom), **Oasis** (Minecraft, diffusion forcing), **Genie** (latent actions), **Cosmos-Predict2** (our scale target). T2V DiT architecture: **Sana** (linear attention, Mix-FFN, AdaLN-zero), **CogVideoX** (3D causal VAE, expert AdaLN), **Latte** (factorized attention, joint image/video training), **AnimateDiff** (inflation recipe). Reference repos: `willisma/diffuse_nnx`, `Vchitect/Latte`, `guoyww/AnimateDiff`, `eloialonso/diamond`, `etched-ai/open-oasis`, `nvidia-cosmos/cosmos-predict2`.

## Risks

- Synthetic domain too simple → memorization. *Mitigation:* held-out action/config splits; physics metrics over FVD; deliberately small models.
- From-scratch VAE artifacts. *Mitigation:* A1 is pixel-space, so diffusion is validated before any VAE exists; visualize reconstructions in A2.
- Theme A rabbit-holes. *Mitigation:* timebox each stage; the A4 gate is the hard exit.
- TRC denied → Theme B unaffordable. *Mitigation:* drop B3b; run B2 at lower resolution.
- Cosmos weight-porting fails. *Mitigation:* B3a delivers the demo without it; B3b is optional.

## Order of operations

**Theme A** (~3–5 weeks, < $50): renderer → A1 → A2 → A3 → A4.
**Theme B** (~3–5 weeks, $150–300): B1 → B2 → B3a → B3b (optional).
Apply for TRC on day 1. Each stage is a working deliverable; Theme A end-to-end is a complete, standalone artifact before any real money is spent.
