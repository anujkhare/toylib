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

Each track isolates one new mechanism. Models are deliberately small — the domain is simple and a larger model just memorizes. Tracks 2–4 build a single joint-sequence transformer where frame latents use diffusion loss and discrete tokens (actions, captions) use AR cross-entropy loss (Transfusion-style). This enables any-conditional inference (`P(X | Y)` for any modality subset) without per-modality conditioning mechanisms.

| Track | What it builds | Loss(es) | New mechanism | Example conditionals unlocked |
|---|---|---|---|---|
| **1** | 2D KL-VAE: `image ↔ latent` | Reconstruction + KL + perceptual | Reparameterization, latent compression | — (codec only) |
| **1b** | Unconditional image generation in latent space | Diffusion (flow matching) | Latent diffusion, Euler sampler | `∅ → frame` |
| **2** | Video diffusion model in latent space | Diffusion | Temporal attention over frame latent sequence | `frames → next frame` |
| **3** | Hybrid video + action model | Diffusion (frames) + AR cross-entropy (actions) | Joint sequence, dual loss, action token AR head | `frames → actions`, `frames + actions → next frame` |
| **3b** *(optional, post e2e spike)* | 3D causal VAE: inflate 2D VAE weights into temporal codec | Reconstruction + KL + perceptual (per-clip) | 3D causal convolutions, temporal compression, weight inflation | — (codec upgrade; re-enables all Track 2–3 conditionals with 4× fewer temporal tokens) |
| **4** | Full any-conditional world model | Diffusion (frames) + AR cross-entropy (actions + captions) | Caption token AR head, modality-masked training | `captions + actions → frames`, `frames → captions`, all prior |

**Training coverage note (Track 3+):** batches must explicitly sample all conditioning patterns — not just forward dynamics. If training only covers `(frames, actions) → next frame`, the model will be poor at `frames → actions`.

**Track 3b note:** only worth attempting once the end-to-end diffusion+AR spike (Tracks 2–3) is validated. The 3D VAE replaces the per-frame encoder with a clip-level encoder (e.g. 4× temporal compression: 16 frames → 4 temporal latent positions × 8×8 spatial = 256 tokens vs 1024). The downstream transformer is re-used unchanged — only the frame latents change shape. Initialize 3D conv weights from the 2D VAE using the AnimateDiff inflation recipe [20].

## Evaluation philosophy

The synthetic domain's killer feature is ground-truth state. Primary metrics are physics-grounded — object-position MSE, collision-timing error, action-response accuracy — computed by detecting shapes in samples and comparing to ground truth. FID/FVD are reported only as rough secondary proxies; their feature extractors (Inception, I3D) are out-of-distribution on synthetic shapes and noisy even on real video.

## Prior work

### Action-conditioned world models

| Paper | Key relevance |
|---|---|
| **DIAMOND** (Alonso et al., NeurIPS 2024) | 50M diffusion world model on Atari; pixel-space; our closest benchmark analogue |
| **GameNGen** (Valevski et al., 2024) | Real-time Doom via diffusion; validates fast autoregressive rollout |
| **Oasis** (Etched, 2024) | Minecraft world model; diffusion forcing for temporal coherence |
| **Genie** (Bruce et al., 2024) | Latent action inference from video; unsupervised action space |
| **Cosmos-Predict2** (NVIDIA, 2025) | Scale target; continuous 3D tokenizer + diffusion world model |

### Video DiT architecture

| Paper | Key relevance |
|---|---|
| **Sana** (NVIDIA, 2024) | Linear attention, Mix-FFN, AdaLN-zero conditioning |
| **CogVideoX** (Yang et al., 2024) | 3D causal VAE design; expert AdaLN; reference for Track 3b |
| **Latte** (Ma et al., 2024) | Factorized spatial/temporal attention; joint image+video training recipe |
| **AnimateDiff** (Guo et al., ICLR 2024) | Weight inflation recipe for 2D → 3D temporal block initialization; used in Track 3b |

### Joint-sequence and any-conditional models

| Paper | Key relevance |
|---|---|
| **Gato** (Reed et al., DeepMind, 2022) | Tokenizes Atari frames, discrete actions, and text into one flat sequence; trains a single AR transformer over all — the most direct architectural precedent |
| **GAIA-1** (Wayve, 2023) | World model for autonomous driving; video + text + ego-actions as a joint AR sequence; structurally identical to our target in a different domain |
| **UniDiffuser** (Bao et al., 2023) | Diffusion transformer over a joint image+text sequence; randomly masks modality subsets at training time to learn `P(X|Y)` for any X, Y from one model |
| **VideoPoet** (Yu et al., Google, 2023) | Tokenizes video, audio, text, and actions into one MAGVIT-2/SentencePiece vocabulary; trains a single AR LLM; demonstrates the full conditional variety we target |
| **Transfusion** (Zhou et al., Meta, 2024) | Hybrid AR+diffusion in one transformer: discrete tokens use next-token prediction, continuous image tokens use diffusion; the natural upgrade path if frame tokens need to stay continuous at larger scale |

### Reference repositories

| Repo | Used for |
|---|---|
| `willisma/diffuse_nnx` | JAX/NNX diffusion reference |
| `Vchitect/Latte` | Factorized attention implementation |
| `guoyww/AnimateDiff` | Temporal inflation recipe |
| `eloialonso/diamond` | DIAMOND Atari world model |
| `etched-ai/open-oasis` | Open Oasis world model |
| `nvidia-cosmos/cosmos-predict2` | Cosmos scale reference |

## Risks / open questions

- Synthetic domain too simple → memorization. *Mitigation:* held-out action/config splits; physics metrics over FVD; deliberately small models.
- From-scratch VAE artifacts.
- **Hybrid Loss Balancing (Track 3+):** Joint sequences with dual losses (Diffusion and AR) are highly sensitive to gradient imbalances. The scale of the continuous flow-matching loss can easily dominate the discrete AR cross-entropy loss. We will need to implement dynamic loss weighting or separate learning rate schedules for the AR and Diffusion heads to ensure the model learns action conditioning alongside pixel generation.
- **Sequence Length Bottlenecks (Track 2+):** Flattening multi-frame spatial latents (e.g., 16 frames of $20\times 20$ latents) results in sequence lengths that quickly exceed the memory capacity of standard dense self-attention. We must explicitly define whether we are using factorized spatial-temporal attention (e.g., Latte architecture) or full 3D attention to manage OOM errors on small hardware.
- **Latent Space Regularization (Track 1):** The KL penalty must be carefully tuned for downstream diffusion. Over-regularization will blur the sharp edges of Atari sprites, while under-regularization will create a latent distribution that the Track 2 diffusion model struggles to learn.
- **Dataset State-Space Coverage:** Atari Breakout is prone to highly correlated, endless loops (e.g., the ball bouncing horizontally indefinitely). The dataset generation script must include periodic forced resets, random ball respawns, or varied initial brick layouts to prevent the model from memorizing infinite-bounce trajectories.
- **Physics Perception on Noisy Outputs:** Evaluating object-position MSE requires extracting coordinates from generated frames. Because early diffusion outputs will contain flickering edges or slight noise, standard contour detection (e.g., OpenCV) may fail. Evaluation pipelines must either use highly noise-tolerant heuristics or a lightweight CNN trained to extract coordinates from generated frames.
