# Dataset Specification: Action-Conditioned Atari Breakout (and Pong)

This document details the engineering specifications, preprocessing pipelines, state-extraction mechanisms, and linguistic templates used to generate the multi-modal dataset for the JAX world model.

---

### 1. Dataset Scale & Requirements

To decouple raw data generation from active training runs and allow rapid scaling experiments, we generate a **High-Capacity Dataset Pool** of **2,000,000 frames** (approx. 2,000 complete gameplay episodes).

- **Total Generated Pool:** ~2,000,000 frames (approx. 2,000 episodes, ~12GB raw disk storage).
- **Slicing Protocol:** Trajectories are sliced into sliding windows of **16 consecutive frames** with a stride of $S=4$ frames.
- **Total Samples in Pool:** ~480,000 distinct 16-frame clips.
- **Modality Formats per Sample:**
  - `frames`: `(16, H, W, 3)` uint8 array (RGB pixel data).
  - `actions`: `(16,)` int32 array (discrete command inputs per frame).
  - `caption`: A single string describing the starting physical state of the 16-frame window.
  - `metadata`: A dictionary of start/end RAM values (paddle position, ball coordinates, score, bricks remaining) for precise physics evaluation.

### Sizing & Scaling Laws Analysis

The dataset scale is carefully calibrated against empirical scaling laws for visual representation learning and diffusion-based world models:

#### A. The VAE Compression Regime (Reconstruction Sizing)
* **Model Scale:** 5M–10M parameters (small 2D continuous KL-VAE).
* **Entropy Bounds:** In contrast to natural images (which have near-infinite visual entropy), Breakout frames feature extremely low visual entropy (static walls, monochromatic rows of bricks, a small ball, a solid-color paddle, and a black background).
* **Data Scale:** 2,000,000 static frames represents $2,000,000 \times 64 \times 64 \approx 8.19 \times 10^9$ spatial pixel coordinates.
* **Scaling Justification:** In vision, VAEs easily overfit or experience codebook collapse (in discrete VQ setups) on small datasets. Having a 2,000,000-frame visual pool guarantees that the VAE encounters a near-infinite variety of brick configurations and ball-paddle alignments, enabling flawless continuous visual reconstruction without sequence memorization.

#### B. The DiT World Model Scaling (Denoising/Flow Sizing)
* **Model Scale:** 50M–80M parameters (spatial-temporal DiT, DIAMOND-scale).
* **The Diffusion Sizing Analogue:** Unlike autoregressive language models (where Chinchilla scaling dictates a linear $D \approx 20 \times N$ token-to-parameter ratio), diffusion models are trained on continuous denoising steps ($t \in [0, 1]$ in Flow Matching). A single 16-frame sequence acts as a source of near-infinite training variations under randomly sampled noise fields.
* **Atari 100k Benchmark Alignment:** Standard NeurIPS world-model papers (e.g., **DIAMOND** and **DreamerV3** on the Atari 100k benchmark) train on trajectories collected from 100,000 agent steps, corresponding to approximately 100,000 transition sequences.

#### C. Dynamic FLOP-to-Data Budgeting
To optimize computational training efficiency, we do not always train on the entire 2,000,000-frame pool. Instead, we dynamically subset the data source inside PyGrain to match the capacity of the active model configuration. This allows us to scale our GPU training budget from $10 to $100 without ever needing to regenerate the underlying physical dataset:

| Model Configuration | Target Parameter Scale | Active Data Slice (Frames) | Active Samples (Clips) | Targeted Epochs | Target Training FLOPs (approx) |
|---|---|---|---|---|---|
| **VAE / Stage A1 Smoke-Test** | ~5M–10M params | 100,000 | ~24,000 | 10 epochs | $O(10^{13})$ FLOPs |
| **Theme A Primary Target (A2-A4)** | ~50M–80M params | 500,000 | ~120,000 | 20 epochs | $O(10^{14})$ FLOPs |
| **Theme B Multi-Host Scale Stretch** | ~150M+ params | 2,000,000 | ~480,000 | 15 epochs | $O(10^{15})$ FLOPs |

---

## 2. Environment Setup & Mixed-Competency Agent

We wrap `gymnasium.make("BreakoutNoFrameskip-v4")` to disable default frame-skipping and action-repeating, ensuring 1-to-1 temporal fidelity between action inputs and visual updates.

### The $\epsilon$-Greedy Play Agent
To avoid out-of-distribution (OOD) visual collapse when a human interactively plays the world model and misses the ball, the data generation script uses a mixed-competency heuristic controller:

1. **Competent Tracking (80% probability):**
   The agent matches the paddle’s horizontal center to the ball’s horizontal position:
   $$a_t = \begin{cases} 2 (\text{RIGHT}) & \text{if } \text{paddle}_x < \text{ball}_x - \delta \\ 3 (\text{LEFT}) & \text{if } \text{paddle}_x > \text{ball}_x + \delta \\ 0 (\text{NOOP}) & \text{otherwise} \end{cases}$$
   *This ensures deep gameplay runs, high scores, ceiling breakthroughs, and diverse bounce angles.*

2. **Exploratory Noise (15% probability):**
   At each step, with a probability of $\epsilon=0.15$, the agent selects a random action $a_t \in \{0, 2, 3\}$. *This generates near-misses, recovery angles, and realistic paddle jitter.*

3. **Forced Failure (5% probability):**
   When the ball is moving downwards ($v_y > 0$) and is in the lower half of the screen ($Y > 120$), the agent is forced with a $5\%$ chance to move *away* from the ball. *This guarantees the world model trains on paddle misses, life loss transitions, and screen resets.*

---

## 3. Visual Preprocessing Pipeline

Raw Atari frames are rendered at $210 \times 160$ RGB pixels. To maximize visual quality and prevent VAE capacity waste, we apply the following sequential pipeline:

```mermaid
graph TD
    A[Raw Frame: 210x160x3] --> B[Crop Play Area: 160x160x3]
    B --> C[Downsample: 64x64x3 or 128x128x3]
    C --> D[Store on Disk: uint8]
    D --> E[JAX DataLoader: Cast & Normalize to [-1, 1]]
```

### A. Spatial Cropping & Aspect Ratio Preservation
Atari Breakout contains static border walls and a top score bar. We crop the frame to keep only the active play area (removing scores and static side walls):
* **Height Crop:** Lines $32$ to $195$ (keeps the bricks, ball, and paddle; discards the score counter).
* **Width Crop:** Pixels $8$ to $152$ (keeps the inner play bounds; discards the thick side walls).
* This yields a square $160 \times 160$ RGB frame, avoiding aspect-ratio stretching during downsampling.

### B. Downsampling
Using bilinear interpolation (via `cv2.resize` or `pillow`), we downsample the cropped square to our target modeling resolution:
- **Standard Resolution (Theme A default):** $64 \times 64 \times 3$ RGB.
- **High Resolution (A3/A4 Stretch target):** $128 \times 128 \times 3$ RGB.

### C. Normalization & JAX Precision
- **Disk Storage:** To save storage space (compressing the 500k dataset to ~3GB), frames are saved as raw `uint8` arrays $[0, 255]$.
- **Dataloader Pipeline:** During JAX batch ingestion, the array is converted to `jax.numpy.bfloat16` or `float32` and normalized to flow-matching scale:
  $$x_{\text{norm}} = \frac{x}{127.5} - 1.0 \quad \in [-1.0, 1.0]$$

---

## 4. Ground-Truth State Extraction from Emulator RAM

Instead of expensive computer vision models or noisy OCR, we query the emulator's memory space directly at the start frame $t_0$ of each 16-frame window using `env.unwrapped.ale.getRAM()`.

### Breakout RAM Address Layout
We map the following indices (specifically for standard Atari ROMs):
* **Ball X-Coordinate:** `ram[99]` (Integer values from 0 to 160).
* **Ball Y-Coordinate:** `ram[101]` (Integer values representing vertical position).
* **Paddle X-Coordinate:** `ram[72]` (Horizontal center of the controlled paddle).
* **Score Counter:** Binary-coded decimals across `ram[76:78]`.
* **Remaining Bricks:** Evaluated by summing the active bits in the block grid memory segment `ram[0:36]`.

> [!TIP]
> **Verification Fallback (Color-based Masking):** 
> To prevent ROM or emulator version desynchronization, the data generator runs a self-validation check on start. It isolates the paddle (pure red pixels, HSV range `[0, 100, 100]`) and the ball (pure white pixels) from the downsampled image and validates their bounding box centers against the RAM values. If RAM variables drift, it falls back to the color-mask coordinates.

---

## 5. Captions & Linguistic Diversity Strategy

To prevent the spatial DiT (Stage A1) from memorizing a single structured prompt structure (which turns text conditioning into a simple lookup table search), we use a dynamic template interpolation engine.

### A. Game State Variables
At timestep $t_0$, the following variables are parsed:
- `paddle_pos`: Classified as `"left third"`, `"center"`, or `"right third"` of the screen.
- `ball_x_dir`: Classified as `"leftward"` or `"rightward"` based on $v_x = \text{ball}_x(t_0) - \text{ball}_x(t_0 - 1)$.
- `ball_y_dir`: Classified as `"falling"` or `"rising"` based on $v_y$.
- `bricks_remaining`: Counted as an integer.

### B. Template Library
The caption generator randomly selects one of 8 distinct grammatical templates to represent the exact same state:

| Template Style | Example Generated String |
|---|---|
| **Direct Declarative** | `"Atari Breakout: ball is at (45, 98) moving falling-rightward, paddle is centered, 18 bricks left."` |
| **Action-Focused** | `"The player controls the paddle at center-right. The ball travels down-leftward towards the paddle. Grid has 32 bricks remaining."` |
| **Spatial Description** | `"A game of Breakout. 12 bricks remain in the overhead wall. The ball is high, rising-leftward at coordinates (72, 40). Paddle is at the left boundary."` |
| **Sparse Technical** | `"Breakout state: paddle=72, ball=(45, 98), trajectory=down-right, blocks=18."` |

### C. Paraphrasing Regularization
In addition to structural template shuffling, we apply light offline token swapping using a pre-defined dictionary:
* `paddle centered` $\leftrightarrow$ `paddle at center` $\leftrightarrow$ `paddle in the middle`
* `bricks remain` $\leftrightarrow$ `blocks left` $\leftrightarrow$ `bricks left in the wall`

This ensures that the DiT text cross-attention layers learn a robust cross-modal semantic mapping rather than an exact word-frequency lookup.

---

## 6. Storage Format & PyGrain Integration

To maximize data loading performance on TPU/GPU clusters, avoid OS-level file descriptor constraints, and maintain 100% deterministic sharding across multiple JAX hosts, the dataset is serialized into Google's **ArrayRecord** format and parsed natively using **PyGrain**.

### A. Data Serialization Protocol
Each 16-frame window sample is packaged as a Python dictionary and serialized using `msgpack` for compact storage. We write shards of the dataset sequentially using `array_record.python.array_record_module.ArrayRecordWriter`:

```python
import array_record.python.array_record_module as ar
import msgpack
import numpy as np

def write_dataset_shard(samples, output_path):
    """Writes a list of samples into a single ArrayRecord file shard."""
    writer = ar.ArrayRecordWriter(output_path)
    for sample in samples:
        # Flatten raw frames to save memory and serialize cleanly
        payload = {
            "frames": sample["frames"].tobytes(),  # (16, H, W, 3) uint8 as bytes
            "actions": sample["actions"].tolist(), # (16,) list of discrete integers
            "caption": sample["caption"],          # Interpolated state string
            "metadata": sample["metadata"]         # Raw dictionary of RAM states
        }
        serialized = msgpack.packb(payload, use_bin_type=True)
        writer.write(serialized)
    writer.close()
```

### B. PyGrain Data Source & Transform Pipeline
During training, PyGrain interacts with the dataset as a random-access `MapDataset`. We construct a high-throughput, multi-threaded `grain.DataLoader` in JAX:

```python
import grain.python as grain
import jax
import msgpack
import numpy as np
import array_record.python.array_record_module as ar

# 1. Define custom PyGrain ArrayRecord Data Source
class ArrayRecordDataSource(grain.RandomAccessDataSource):
    def __init__(self, file_pattern: str):
        self._reader = ar.ArrayRecordReader(file_pattern)
        self._length = self._reader.num_records()

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict:
        serialized = self._reader.read(index)
        return msgpack.unpackb(serialized, raw=False)

# 2. Define deterministic transform to decode and normalize binary data
class ParseAndNormalizeTransform(grain.MapTransform):
    def __init__(self, image_resolution: int = 64):
        self.res = image_resolution

    def map(self, element: dict) -> dict:
        # Reconstruct uint8 frames from binary msgpack payload
        flat_frames = np.frombuffer(element["frames"], dtype=np.uint8)
        frames = flat_frames.reshape((16, self.res, self.res, 3))
        
        # Cast and scale to flow-matching standard range [-1.0, 1.0]
        element["frames"] = (frames.astype(np.float32) / 127.5) - 1.0
        element["actions"] = np.array(element["actions"], dtype=np.int32)
        return element

# 3. Create Multi-Host JAX DataLoader
def make_jax_data_loader(file_pattern: str, batch_size: int, seed: int, num_workers: int = 4):
    data_source = ArrayRecordDataSource(file_pattern)
    
    # Initialize index sampler with deterministic distributed sharding
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=None,  # Infinite training stream
        shuffle=True,
        seed=seed,
        shard_options=grain.ShardOptions(
            shard_index=jax.process_index(),  # multi-host process index
            num_shards=jax.process_count()    # total accelerator hosts
        )
    )
    
    # Assemble the PyGrain Loader with transformations
    loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=[
            ParseAndNormalizeTransform(image_resolution=64),
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
        worker_count=num_workers,
    )
    return loader
```

