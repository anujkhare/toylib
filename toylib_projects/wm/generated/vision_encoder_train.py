from __future__ import annotations

# ============================================================
# External Imports
# ============================================================

from PIL import Image as PILImage
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from pathlib import Path
import abc
import argparse
import dataclasses
import datetime
import einops
import grain.python as grain
import h5py
import hdf5plugin
import jax
import jax.numpy as jnp
import jaxtyping as jt
import json
import math
import numpy as np
import optax
import orbax.checkpoint as ocp
import os
import pandas as pd
import time
import typing
import wandb

# ============================================================
# toylib_projects.wm.vision_encoder.analyze - /Users/anuj/Desktop/code/toylib/toylib_projects/wm/vision_encoder/analyze.py
# ============================================================

"""Reporting helpers for VAE training: parameter counts, dataset sizing, and
capacity analysis.

Mirrors the shape of ``toylib_projects/tinystories/analyze.py`` so the train
script has the same ergonomics: build the experiment, init state, then call
``print_*`` helpers to dump a human-readable summary to stdout (and to the
WandB run config via the standard logger plumbing).

What's different from tinystories
---------------------------------

**No Chinchilla.** Chinchilla's "~20 tokens per parameter" rule is calibrated
to next-token cross-entropy on natural language. For an image VAE trained
with per-pixel reconstruction loss, the relevant scaling unit is
**pixel-supervisions** (``samples × H × W × C``), not tokens, and the
rule-of-thumb sweet spot is much looser (autoencoders converge well below
the Chinchilla ratio because each sample provides H*W*C parallel scalar
supervisions). So we report capacity in pixel-supervisions per parameter and
flag whether we land in the typical "well-fed" range (100–1,000 supervisions
per param across the whole run).

**Walkthrough milestones.** The Track A1 walkthrough lists three concrete
training targets (Milestones 4 / 5 / 6). We surface them next to the
current ``max_steps`` so you can see at a glance whether the configured run
matches a milestone.
"""

def get_tree_stats(model: jt.PyTree) -> pd.DataFrame:
    """One row per array leaf in the model pytree, with ``params``, ``n_bytes``,
    ``dtype``, ``path``, and ``level_<i>`` columns for grouping at any depth."""
    results = []
    leaf_stats = [(k, v.shape, v.dtype) for k, v in jax.tree_util.tree_leaves_with_path(model)]
    for path, shape, dtype in leaf_stats:
        path = [str(p) for p in path]
        count = math.prod(shape)
        nbytes = count * dtype.itemsize
        row = {'params': count, 'n_bytes': nbytes, 'dtype': str(dtype), 'path': '/'.join(path)}
        for i, p in enumerate(path):
            row[f'level_{i}'] = p
        results.append(row)
    return pd.DataFrame(results)

def print_param_sizes(model: jt.PyTree, depth: int=1, size_denom: int=1) -> tuple[pd.DataFrame, int, int]:
    """Print and return a grouped param-count / byte table.

    Same surface as ``tinystories.analyze.print_param_sizes`` so train scripts
    can use it interchangeably. ``depth`` controls how many path components
    to group by (depth=1 = encoder vs decoder; depth=2 = per sub-block).
    """
    df_stats = get_tree_stats(model)
    if len(df_stats) == 0:
        print('Model has no parameters.')
        return (pd.DataFrame(), 0, 0)
    df_stats.loc[:, 'n_bytes_divided'] = df_stats['n_bytes'] / size_denom
    total_params = int(df_stats['params'].sum())
    total_bytes = float(df_stats['n_bytes_divided'].sum())
    print(f'Total Parameters: {total_params:,}. Bytes: ({total_bytes:,.2f})')
    level_cols = [f'level_{i}' for i in range(depth)]
    grouped = df_stats.fillna('').groupby(level_cols + ['dtype']).sum()[['params', 'n_bytes_divided']].reset_index()
    return (grouped, total_params, total_bytes)

def print_dataset_stats(dataset, name: str='train') -> dict[str, int]:
    """Print shape, count, and pixel volume for a ``Hdf5FramesDataset``."""
    H, W, C = dataset.frame_shape
    n_frames = dataset.num_frames
    pixels_per_frame = H * W * C
    total_pixels = n_frames * pixels_per_frame
    print('------------------------------')
    print(f'{name.capitalize()} dataset:')
    print('------------------------------')
    print(f'  Frame shape:        {H}x{W}x{C} ({pixels_per_frame:,} pixels/frame)')
    print(f'  Frames:             {n_frames:,}')
    print(f'  Batch size:         {dataset.batch_size:,}')
    print(f'  Batches per epoch:  {len(dataset):,}')
    print(f'  Total pixels:       {total_pixels:,}')
    print(f'  Raw uint8 bytes:    {total_pixels:,} ({total_pixels / 1000000000.0:.2f} GB)')
    return {'n_frames': n_frames, 'pixels_per_frame': pixels_per_frame, 'total_pixels': total_pixels, 'batches_per_epoch': len(dataset)}

def print_training_estimate(exp) -> dict[str, float]:
    """Total samples / epochs / pixel-supervisions for the configured run."""
    ds = exp.train_task.dataset
    max_steps = exp.training_config.max_steps
    batch_size = ds.batch_size
    pixels_per_frame = math.prod(ds.frame_shape)
    n_frames = ds.num_frames
    samples_seen = max_steps * batch_size
    epochs = samples_seen / max(n_frames, 1)
    pixel_supervisions = samples_seen * pixels_per_frame
    print('------------------------------')
    print('Training estimate:')
    print('------------------------------')
    print(f'  Max steps:                 {max_steps:,}')
    print(f'  Microbatches per step:     {exp.training_config.num_microbatches:,}')
    print(f'  Samples per step:          {batch_size:,}')
    print(f'  Total samples seen:        {samples_seen:,}')
    print(f'  Equivalent epochs:         {epochs:.2f}')
    print(f'  Total pixel-supervisions:  {pixel_supervisions:,.0f}')
    return {'max_steps': max_steps, 'samples_seen': samples_seen, 'epochs': epochs, 'pixel_supervisions': pixel_supervisions}

def print_capacity_analysis(model: jt.PyTree, exp) -> dict[str, float]:
    """Pixel-supervisions per parameter — the VAE analogue of the Chinchilla ratio.

    For an autoencoder, each training sample provides ``H*W*C`` independent
    scalar supervisions (one per pixel), not a single token. So the
    cross-entropy-trained Chinchilla rule (~20 tokens/param) doesn't apply
    directly. Empirical sweet spot for ImageNet-scale VAEs: roughly
    ``100–1,000 pixel-supervisions per parameter`` across the full run.
    Below 100 you'll typically still be improving; above ~10k you're
    starting to memorize.
    """
    ds = exp.train_task.dataset
    n_params = sum((int(math.prod(t.shape)) for t in jax.tree.leaves(model)))
    samples_seen = exp.training_config.max_steps * ds.batch_size
    pixels_per_sample = math.prod(ds.frame_shape)
    pixel_supervisions = samples_seen * pixels_per_sample
    ratio = pixel_supervisions / max(n_params, 1)
    print('------------------------------')
    print('Capacity analysis (VAE):')
    print('------------------------------')
    print(f'  Model parameters:                {n_params:,}')
    print(f'  Pixel-supervisions over run:     {pixel_supervisions:,.0f}')
    print(f'  Pixel-supervisions per param:    {ratio:,.1f}')
    if ratio < 30:
        verdict = 'UNDER-trained — increase max_steps or data; expect poor reconstruction'
    elif ratio < 100:
        verdict = 'below typical sweet spot (100–1000); may not converge fully'
    elif ratio <= 10000:
        verdict = 'in the typical well-trained range for image VAEs'
    else:
        verdict = 'well above typical — likely memorizing; KL term should still regularize'
    print(f'  Verdict:                         {verdict}')
    return {'n_params': n_params, 'pixel_supervisions': pixel_supervisions, 'supervisions_per_param': ratio}

def print_walkthrough_milestones(exp) -> None:
    """Compare ``max_steps`` against the Track A1 walkthrough milestones."""
    max_steps = exp.training_config.max_steps
    milestones = [('M4 overfit test (300 steps)', 300, 'L_rec < 0.05 on 8 fixed frames'), ('M5 + perceptual + aux (+1k)', 1300, 'extends M4 with VGG + ball MLP'), ('M6 full A1 run (50k–100k)', 75000, 'stage-gate eval (SSIM > 0.85, ball detect > 95%)')]
    print('------------------------------')
    print(f'Walkthrough milestones (your max_steps = {max_steps:,}):')
    print('------------------------------')
    closest = min(milestones, key=lambda m: abs(m[1] - max_steps))
    for label, target, goal in milestones:
        marker = '→' if (label, target, goal) == closest else ' '
        print(f'  {marker} {label:<35} target {target:>7,} steps  ({goal})')

def print_all(exp, *, depth: int=2) -> None:
    """Convenience: param sizes + dataset stats + training estimate + capacity."""
    grouped, total_params, _ = print_param_sizes(exp.model, depth=depth)
    print()
    print(grouped.to_string(index=False))
    print()
    print_dataset_stats(exp.train_task.dataset, name='train')
    if exp.eval_task is not None:
        print()
        print_dataset_stats(exp.eval_task.dataset, name='val')
    print()
    print_training_estimate(exp)
    print()
    print_capacity_analysis(exp.model, exp)
    print()
    print_walkthrough_milestones(exp)

# ============================================================
# toylib_projects.wm.vision_encoder.dataloader - /Users/anuj/Desktop/code/toylib/toylib_projects/wm/vision_encoder/dataloader.py
# ============================================================

"""Dataloaders for the vision encoder (VAE) training pipeline.

Design notes
------------

Two patterns live in this file:

  - ``DatasetState`` — a generic dataclass-based "state" type that any
    dataset can subclass. The intent is that dataset checkpoints round-trip
    cleanly through orbax/Pickle by being plain dataclasses.

  - ``Hdf5FramesDataset`` — concrete loader over the compiled vae_*.h5 files
    produced by ``datagen/generate_vision_enc_data.py``. Uses Grain's random
    access pipeline (shuffle → batch → iter) and exposes Grain's iterator
    state via ``get_state`` / ``restore_state`` so training can checkpoint
    its position in the shuffled stream.

We deliberately do **not** inherit the frames loader from any text-streaming
base class: the tokenizer / token-buffer / seq_len machinery used by the
text loader doesn't apply to images. We only share the ``DatasetState``
checkpointing protocol.
"""

@dataclasses.dataclass
class DatasetState:
    """Serializable state for dataset checkpointing."""
    pass

@dataclasses.dataclass
class Hdf5FramesDatasetState(DatasetState):
    """Checkpointable state for ``Hdf5FramesDataset``.

    Currently just the Grain iterator's internal state (the next index in the
    shuffled stream). Kept as a dict — Grain's get/set_state already returns
    a dict — so this trivially round-trips through json/orbax.
    """
    sampler_state: dict = dataclasses.field(default_factory=dict)

class _Hdf5FramesSource:
    """Grain RandomAccessDataSource backed by one compiled vae_*.h5 file.

    Implements the protocol Grain expects of a random-access source:
    ``__len__`` + ``__getitem__``. Each ``__getitem__`` returns a single
    frame as ``(H, W, 3)`` uint8.

    The HDF5 file handle is opened **lazily** on first access. This matters
    because Grain may pickle this source to send to worker processes; an
    open ``h5py.File`` cannot be pickled. The ``__getstate__`` /
    ``__setstate__`` overrides drop the live handle on pickle so each
    worker opens its own descriptor.
    """

    def __init__(self, dataset_path: str) -> None:
        self._path = str(dataset_path)
        self._f: typing.Optional[h5py.File] = None
        with h5py.File(self._path, 'r') as f:
            self._n = int(f.attrs['n_frames'])
            self._height = int(f.attrs['height'])
            self._width = int(f.attrs['width'])

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return (self._height, self._width, 3)

    def _ensure_open(self) -> None:
        if self._f is None:
            self._f = h5py.File(self._path, 'r')

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> np.ndarray:
        self._ensure_open()
        return self._f['frames'][idx]

    def __getstate__(self) -> dict:
        return {'_path': self._path, '_f': None, '_n': self._n, '_height': self._height, '_width': self._width}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

@dataclasses.dataclass
class Hdf5FramesDataset:
    """Stream batches of frames from a compiled vae_*.h5 file.

    Yields ``jnp.ndarray`` batches of shape ``(batch_size, H, W, 3)`` with
    dtype ``uint8``. Normalization (uint8 → float32 / [-1, 1]) is left to
    the training loop so this loader stays pure and easy to inspect in tests.

    Checkpointing follows the same ``DatasetState`` / ``get_state`` /
    ``restore_state`` protocol used elsewhere in the project. The state
    captures the Grain iterator's position so resuming after a crash gives
    the same per-epoch shuffle order from the next index forward.

    Args
    ----
    dataset_path :
        Path to ``data/compiled/vae_train.h5`` (or val/test).
    batch_size :
        Number of frames per batch.
    seed :
        Shuffle seed. Together with the iterator state, fully determines the
        shuffled order for reproducible resumes.
    shuffle :
        If False, frames are read in storage order. The compiled file is
        already pre-shuffled at write time, so a sequential read is also a
        valid (and faster) "random-but-fixed" order.
    drop_remainder :
        Drop the last partial batch (so every batch has exactly
        ``batch_size`` frames).
    """
    dataset_path: str
    batch_size: int = 64
    seed: int = 0
    shuffle: bool = True
    drop_remainder: bool = True
    repeat: bool = False

    def __post_init__(self) -> None:
        self._state = Hdf5FramesDatasetState()
        self._source = _Hdf5FramesSource(self.dataset_path)
        self._grain_iterator: typing.Optional[typing.Any] = None
        self.dataset_iter = self._make_iterator()

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """``(H, W, 3)`` per-frame shape, read from the file attrs."""
        return self._source.frame_shape

    @property
    def num_frames(self) -> int:
        """Total frames in the underlying file (before batching)."""
        return len(self._source)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self._source) // self.batch_size
        if not self.drop_remainder and len(self._source) % self.batch_size:
            n += 1
        return n

    def _make_iterator(self) -> typing.Iterator[jnp.ndarray]:
        """Build the Grain pipeline and yield (batch_size, H, W, 3) batches.

        When ``repeat=True`` the pipeline loops indefinitely; each epoch uses
        ``seed + epoch`` so the shuffle order differs across epochs.
        """
        epoch = 0
        while True:
            dataset = grain.MapDataset.source(self._source)
            if self.shuffle:
                dataset = dataset.shuffle(seed=self.seed + epoch)
            dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
            iterator = iter(dataset)
            if epoch == 0 and self._state.sampler_state:
                iterator.set_state(self._state.sampler_state)
            self._grain_iterator = iterator
            for batch in iterator:
                yield jnp.asarray(batch)
            if not self.repeat:
                break
            epoch += 1

    def __iter__(self) -> 'Hdf5FramesDataset':
        return self

    def __next__(self) -> jnp.ndarray:
        return next(self.dataset_iter)

    def get_state(self) -> dict[str, typing.Any]:
        """Capture the current shuffled-iterator position for checkpointing."""
        if self._grain_iterator is None:
            raise RuntimeError('Iterator not initialized; call __post_init__ first.')
        self._state.sampler_state = self._grain_iterator.get_state()
        return dataclasses.asdict(self._state)

    def restore_state(self, state: dict[str, typing.Any]) -> None:
        """Restore from a previously captured ``get_state()`` dict.

        Rebuilds the iterator and fast-forwards to the saved position. After
        this call, ``__next__`` will return the same batch sequence as if the
        original run hadn't been interrupted (assuming same ``seed``,
        ``batch_size``, and underlying file).
        """
        self._state = Hdf5FramesDatasetState(**state)
        self.dataset_iter = self._make_iterator()

# ============================================================
# toylib_projects.wm.vision_encoder.logger - /Users/anuj/Desktop/code/toylib/toylib_projects/wm/vision_encoder/logger.py
# ============================================================

"""Metric loggers for the vision-encoder training loop.

Forked verbatim from `toylib_projects/tinystories/logger.py`. Pure
infrastructure — no model dependencies.

Three sinks are provided:

  - ``StdoutLogger`` — prints to stdout (great for local development)
  - ``FileLogger``   — appends JSON-lines to a file under ``output_path``
  - ``WandBLogger``  — lazily imports wandb; for cloud runs

All three share the ``Logger`` ABC so the experiment can swap them via
config without touching the training loop.
"""

class Logger(abc.ABC):
    """Interface for logging training metrics."""

    def __init__(self, config_dict: dict, *args, **kwargs) -> None:
        self.config_dict = config_dict

    @abc.abstractmethod
    def log(self, step: int, metrics: dict) -> None:
        """Log the given metrics at the specified step."""
        pass

    def log_images(self, step: int, key: str, images: np.ndarray) -> None:
        """Log a uint8 (N, H, W, 3) image batch. No-op if not overridden."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Close any resources held by the logger."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class WandBLogger(Logger):
    """Logger implementation using Weights and Biases (wandb)."""

    def __init__(self, config_dict: dict, project_name: str, user_name: str, run_id: str | None=None, *args, **kwargs) -> None:
        self.config_dict = config_dict
        self.run = wandb.init(entity=user_name, project=project_name, config=self.config_dict, id=run_id, resume='allow')
        self.run.define_metric('*', step_metric='global_step')

    def log(self, step: int, metrics: dict) -> None:
        metrics['global_step'] = step
        self.run.log(metrics)

    def log_images(self, step: int, key: str, images: np.ndarray) -> None:
        self.run.log({key: [wandb.Image(img) for img in images], 'global_step': step})

    def close(self) -> None:
        self.run.finish()

class FileLogger(Logger):
    """Logger implementation that logs metrics to a local file."""

    def __init__(self, config_dict: dict, output_path: str, run_id: str | None=None, *args, **kwargs) -> None:
        self.config_dict = config_dict
        label = run_id or datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        os.makedirs(output_path, exist_ok=True)
        self.file_ptr = open(os.path.join(output_path, f'logs_{label}.txt'), 'w')
        self.file_ptr.write('\n')

    def log(self, step: int, metrics: dict) -> None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics['timestamp'] = timestamp
        metrics['step'] = step
        self.file_ptr.write(json.dumps(metrics, default=str) + '\n')
        self.file_ptr.flush()

    def log_images(self, step: int, key: str, images: np.ndarray) -> None:
        try:
            n, h, w, c = images.shape
            ncols = min(n, 4)
            nrows = math.ceil(n / ncols)
            grid = np.zeros((nrows * h, ncols * w, c), dtype=np.uint8)
            for i, img in enumerate(images):
                r, col = divmod(i, ncols)
                grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = img
            out_dir = os.path.dirname(self.file_ptr.name)
            fname = f"step{step:07d}_{key.replace('/', '_')}.png"
            PILImage.fromarray(grid).save(os.path.join(out_dir, fname))
        except ImportError:
            pass

    def close(self) -> None:
        self.file_ptr.close()

class StdoutLogger(Logger):
    """Logger implementation that logs metrics to standard output."""

    def __init__(self, config_dict: dict, *args, **kwargs) -> None:
        self.config_dict = config_dict

    def log(self, step: int, metrics: dict) -> None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{timestamp}] Step {step}: {metrics}')

    def log_images(self, step: int, key: str, images: np.ndarray) -> None:
        print(f'[Step {step}] {key}: {images.shape} uint8 images')

    def close(self) -> None:
        pass

# ============================================================
# toylib_projects.wm.vision_encoder.metrics - /Users/anuj/Desktop/code/toylib/toylib_projects/wm/vision_encoder/metrics.py
# ============================================================

"""Generic metric scaffolding for the vision-encoder training loop.

Forked from `toylib_projects/tinystories/metrics.py`, keeping only the
domain-agnostic pieces:

  - ``Metric`` Protocol — interface every metric implements.
  - ``Loss``            — pass-through metric that surfaces the forward-fn loss.

The text-specific ``BitsPerByte`` is intentionally not forked. Image-specific
metrics (PSNR, SSIM, reconstruction MSE per pixel, per-mode FID, etc.) are
deferred to the model-author — they're not infrastructure.
"""

class Metric(typing.Protocol):
    """Protocol for computing and accumulating metrics."""

    def __call__(self, loss: float, aux: jt.PyTree, batch: jt.PyTree) -> dict[str, jt.Array]:
        """Compute final metric value(s) for the given inputs.

        Args:
            loss: The loss value returned by forward_fn.
            aux: The auxiliary jt.PyTree returned by forward_fn.
            batch: The input batch.
        """
        pass

@dataclasses.dataclass
class Loss:
    """Pass-through metric that returns the loss value."""

    def __call__(self, loss: float, aux: jt.PyTree, batch: jt.PyTree) -> dict[str, jt.Array]:
        del aux, batch
        return {'loss': loss}

class VisualizationMetric(typing.Protocol):
    """Protocol for generative image metrics that run outside JIT.

    Unlike ``Metric``, these don't consume an input batch — they generate images
    directly from the model (e.g. decoding random latents). Each call returns a
    dict of ``{name: (N, H, W, 3) uint8 ndarray}`` logged via ``logger.log_images``.

    For reconstruction visualization (encoding then decoding real frames), use
    ``ReconstructionVisualization`` which implements the standard ``Metric``
    protocol and runs inside the JIT-compiled eval step via ``aux``.
    """

    def __call__(self, model: typing.Any) -> dict[str, np.ndarray]:
        ...

@dataclasses.dataclass
class ReconstructionVisualization:
    """Return input frames and their VAE reconstructions from the eval forward pass.

    Implements the standard ``Metric`` protocol so it runs inside the JIT-compiled
    eval step. The eval ``forward_fn`` must include ``"recon"`` in the returned
    ``aux`` dict (float32, ``[-1, 1]``). Both outputs are converted to uint8
    ``[0, 255]`` before being returned for logging.

    Args:
        recon_aux_key: Key in ``aux`` where the eval forward_fn stores reconstructed
            frames as float32 in ``[-1, 1]``.
        num_images: How many images from the batch to return.
    """
    recon_aux_key: str = 'recon'
    num_images: int = 8
    gap_px: int = 4

    def __call__(self, loss: float, aux: jt.PyTree, batch: jt.PyTree) -> dict[str, jt.Array]:
        del loss
        inputs = batch[:self.num_images]
        recon_f32 = aux[self.recon_aux_key][:self.num_images]
        recons = ((recon_f32 + 1.0) * 127.5).clip(0, 255).astype(jnp.uint8)
        n, h = inputs.shape[:2]
        gap = jnp.full((n, h, self.gap_px, 3), 128, dtype=jnp.uint8)
        comparison = jnp.concatenate([inputs, gap, recons], axis=2)
        return {'recon_comparison': comparison}

@dataclasses.dataclass
class PriorSamplingVisualization:
    """Log images decoded from randomly sampled latents.

    The PRNG key is derived from ``seed`` and held fixed across evals so
    outputs are directly comparable over the course of training.

    Args:
        sample_fn: ``(model, key, n) -> images`` where ``images`` is
            ``(n, H, W, 3)`` uint8.
        num_samples: Number of images to generate.
        seed: Fixed seed for the sampling key.
    """
    sample_fn: typing.Callable[..., np.ndarray]
    num_samples: int = 16
    seed: int = 42

    def __call__(self, model: typing.Any) -> dict[str, np.ndarray]:
        key = jax.random.key(self.seed)
        samples = self.sample_fn(model, key, self.num_samples)
        return {'prior_samples': np.asarray(samples)}

# ============================================================
# toylib_projects.wm.vision_encoder.experiment - /Users/anuj/Desktop/code/toylib/toylib_projects/wm/vision_encoder/experiment.py
# ============================================================

"""Experiment harness for vision-encoder training.

Forked and adapted from `toylib_projects/tinystories/experiment.py`. The bulk
of the orchestration (sharded `Mesh`, microbatching scan, optimizer chain,
checkpoint manager, logger plumbing) is unchanged — only the text-specific
pieces have been ripped out and a couple of generic seams added so this
harness works with the vision-encoder dataset and any user-supplied model.


What changed vs. the tinystories template
-----------------------------------------

  - **No model imports.** Per the project's learning rules, this file knows
    nothing about VAEs / DiTs / attention. The user supplies the model via
    ``model_factory(model_config, key) -> model`` and the loss via
    ``forward_fn(model, batch) -> (loss, aux_pytree)``.
  - **No tokenizer.** The text-only ``sampling_evaluation`` and the
    ``DEFAULT_PROMPTS`` constant are gone. ``eval()`` runs validation only.
    Subclasses can override ``sampling_evaluation`` if they want to log VAE
    reconstructions or similar — it's a no-op by default.
  - **Generic batch shape.** ``_train_step`` no longer assumes
    ``batch["inputs"]``; it pulls the batch size from
    ``jax.tree.leaves(batch)[0]`` so dicts *and* raw tensors both work.
  - **Generic forward-fn contract.** ``forward_fn(model, batch)`` always
    returns ``(loss, aux)``. The old ``return_aux=True`` kwarg has been
    dropped — one consistent signature for train and eval.
  - **Dataset type.** ``Task.dataset`` is typed as ``Hdf5FramesDataset``
    (forked HDF5-backed Grain loader with checkpointable iterator state).
    Any object with the same protocol (``__iter__``, ``batch_size``,
    ``get_state``, ``restore_state``) works at runtime.
  - **Required model_factory + forward_fn.** No tinystories defaults.


Intended usage (sketch)
-----------------------

::

    from vision_encoder.dataloader import Hdf5FramesDataset
    from vision_encoder.experiment import (
        Experiment, Task, TrainingConfig, EvalConfig,
        CheckpointConfig, LoggerConfig,
    )

    train_ds = Hdf5FramesDataset("data/compiled/vae_train.h5", batch_size=64, seed=0)
    val_ds   = Hdf5FramesDataset("data/compiled/vae_val.h5",   batch_size=64,
                                 seed=0, shuffle=False)

    # User code (the bits this harness deliberately doesn't touch):
    #   class MyVAEConfig: ...
    #   class MyVAE: ...
    #   def vae_factory(cfg, key): return MyVAE(cfg, key=key)
    #   def vae_train_step(model, batch): return loss, {"recon_mse": ...}

    exp = Experiment(
        train_task=Task("train", train_ds),
        eval_task=Task("val", val_ds),
        model_config=MyVAEConfig(...),
        model_factory=vae_factory,
        forward_fn=vae_train_step,
    )
    exp.init_state()
    exp.outer_loop()
    exp.cleanup()
"""

@dataclasses.dataclass
class CheckpointConfig:
    save_interval_steps: int = 5000
    max_to_keep: typing.Optional[int] = 10
    checkpoint_dir: str = '/tmp/checkpoints'
    checkpoint_dataset_iterator: bool = False

@dataclasses.dataclass
class OptimizerConfig:
    """Configuration for a single optimizer."""
    name: str
    optimizer: optax.GradientTransformation

@dataclasses.dataclass
class MultiOptimizerConfig:
    """Configuration for multi-optimizer training (e.g. different lr for VAE encoder vs decoder)."""
    optimizer_configs: list[OptimizerConfig]
    optimizer_for_param: typing.Callable[[tuple], str]

    def build_optimizer_map(self) -> dict[str, optax.GradientTransformation]:
        if not self.optimizer_configs:
            raise ValueError('multi_optimizer_config.optimizer_configs cannot be empty')
        optimizer_map = {config.name: config.optimizer for config in self.optimizer_configs}
        assert len(optimizer_map) == len(self.optimizer_configs)
        return optimizer_map

@dataclasses.dataclass
class TrainingConfig:
    optimizer_config: MultiOptimizerConfig | None = None
    max_steps: int = 100000
    num_microbatches: int = 1
    max_grad_norm: float = 0.0

@dataclasses.dataclass
class EvalConfig:
    eval_interval_steps: int = 500
    num_eval_steps: int = 1

@dataclasses.dataclass
class Task:
    name: str
    dataset: Hdf5FramesDataset
    metrics: list[Metric] = dataclasses.field(default_factory=lambda: [Loss()])
    visualization_metrics: list[VisualizationMetric] = dataclasses.field(default_factory=list)

@dataclasses.dataclass(kw_only=True)
class LoggerConfig:
    logger_cls: type[Logger] = FileLogger
    log_dir: str = '/tmp/'
    run_id: str | None = None
    train_log_interval_steps: int = 1

    def build_logger(self, config_dict: dict) -> Logger:
        return self.logger_cls(config_dict=config_dict, output_path=self.log_dir, run_id=self.run_id)

@dataclasses.dataclass
class WandBLoggerConfig(LoggerConfig):
    logger_cls: type[Logger] = WandBLogger
    project_name: str = ''
    user_name: str = ''

    def build_logger(self, config_dict: dict) -> Logger:
        return self.logger_cls(config_dict=config_dict, output_path=self.log_dir, project_name=self.project_name, user_name=self.user_name, run_id=self.run_id)

def _serialize_dataclass_config(config) -> dict:
    """Recursively convert dataclasses to dicts, leaving non-dataclass values alone.

    Used to capture the experiment configuration for the logger. Non-dataclass
    values (callables, user model configs, etc.) are stringified so json/wandb
    can serialize them without crashing.
    """
    if dataclasses.is_dataclass(config) and (not isinstance(config, type)):
        result = {}
        for f in dataclasses.fields(config):
            v = getattr(config, f.name)
            try:
                result[f.name] = _serialize_dataclass_config(v)
            except (TypeError, ValueError):
                result[f.name] = repr(v)
        return result
    if isinstance(config, (list, tuple)):
        return [_serialize_dataclass_config(x) for x in config]
    if isinstance(config, dict):
        return {k: _serialize_dataclass_config(v) for k, v in config.items()}
    if isinstance(config, (str, int, float, bool, type(None))):
        return config
    return repr(config)

@dataclasses.dataclass
class Experiment:
    """Generic training harness for the vision-encoder pipeline.

    The user supplies the model + loss; this class owns sharding, microbatching,
    optimizer chain, checkpointing, validation loop, and metric logging.

    Required fields:
        train_task:     ``Task`` with a ``Hdf5FramesDataset``.
        forward_fn:     ``(model, batch) -> (loss, aux_pytree)``.
        model_factory:  ``(model_config, jax.random.key) -> model`` callable
                        invoked at ``init_state()`` time.

    Optional fields default to sensible scaffolding values.
    """
    train_task: Task
    forward_fn: typing.Callable[..., tuple[jt.Array, jt.PyTree]]
    model_factory: typing.Callable[..., typing.Any]
    eval_task: Task | None = None
    eval_forward_fn: typing.Callable[..., tuple[jt.Array, jt.PyTree]] | None = None
    model_config: typing.Any = None
    training_config: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    eval_config: EvalConfig = dataclasses.field(default_factory=EvalConfig)
    checkpoint_config: CheckpointConfig = dataclasses.field(default_factory=CheckpointConfig)
    logger_config: LoggerConfig = dataclasses.field(default_factory=LoggerConfig)
    seed: int = 0
    jit_computations: bool = True

    def _validate_configs(self) -> None:
        train_batch_size = self.train_task.dataset.batch_size
        eval_batch_size = self.eval_task.dataset.batch_size if self.eval_task is not None else 0
        if train_batch_size % self.num_devices != 0:
            raise ValueError(f'Train batch size {train_batch_size} not divisible by number of devices {self.num_devices}')
        if eval_batch_size % self.num_devices != 0 and eval_batch_size != 0:
            raise ValueError(f'Eval batch size {eval_batch_size} not divisible by number of devices {self.num_devices}')
        if train_batch_size // self.num_devices % self.training_config.num_microbatches != 0:
            raise ValueError(f'Number of microbatches {self.training_config.num_microbatches} does not evenly divide per-device batch size {train_batch_size // self.num_devices}')

    def _setup_sharding(self) -> None:
        self.num_devices = jax.local_device_count()
        devices = np.array(jax.local_devices())
        self.mesh = Mesh(devices, axis_names=('data',))
        self.replicated_sharding = NamedSharding(self.mesh, P())
        self.data_sharding = NamedSharding(self.mesh, P('data'))
        print(f'Initialized mesh {self.mesh} with {self.num_devices} devices: {devices}')

    def _compute_metrics(self, task: Task, loss: float, aux: jt.PyTree, batch: jt.PyTree) -> dict[str, jt.Array]:
        all_metrics: dict[str, jt.Array] = {}
        for metric in task.metrics:
            metric_results = metric(loss=loss, aux=aux, batch=batch)
            all_metrics.update(metric_results)
        return all_metrics

    def _create_optimizer(self) -> optax.GradientTransformation:
        """Build the optimizer chain: optional grad-clip then a single or multi-optimizer."""
        optimizer_chain: list[optax.GradientTransformation] = []
        if self.training_config.max_grad_norm > 0.0:
            optimizer_chain.append(optax.clip_by_global_norm(self.training_config.max_grad_norm))
        if self.training_config.optimizer_config is None:
            print('Using default optimizer: Adam, lr=1e-3')
            optimizer_chain.append(optax.adam(learning_rate=0.001))
        else:
            optimizer_map = self.training_config.optimizer_config.build_optimizer_map()
            optimizer_for_param = self.training_config.optimizer_config.optimizer_for_param

            def label_fn(params):
                return jax.tree_util.tree_map_with_path(lambda path, _: optimizer_for_param(path), params)
            optimizer_chain.append(optax.multi_transform(transforms=optimizer_map, param_labels=label_fn))
        if len(optimizer_chain) == 1:
            return optimizer_chain[0]
        return optax.chain(*optimizer_chain)

    def _train_step(self, model, opt_state, batch):
        """One sharded training step with microbatching."""
        sharded_batch_size = jax.tree.leaves(batch)[0].shape[0]
        num_microbatches = self.training_config.num_microbatches
        microbatch_size = sharded_batch_size // num_microbatches
        microbatches = jax.tree.map(lambda x: x.reshape(num_microbatches, microbatch_size, *x.shape[1:]), batch)

        def scan_fn(carry_grads, microbatch):
            (loss_val, aux), grads = jax.value_and_grad(self.forward_fn, has_aux=True)(model, microbatch)
            new_carry_grads = jax.tree.map(lambda c, g: c + g, carry_grads, grads)
            microbatch_metrics = self._compute_metrics(task=self.train_task, loss=loss_val, aux=aux, batch=microbatch)
            return (new_carry_grads, microbatch_metrics)
        init_carry_grads = jax.tree.map(jnp.zeros_like, model)
        with jax.profiler.TraceAnnotation('microbatch_loop'):
            total_grads, all_metrics = jax.lax.scan(scan_fn, init_carry_grads, microbatches)
        total_grads = jax.tree.map(lambda g: g / self.num_devices / num_microbatches, total_grads)
        averaged_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), all_metrics)
        with jax.profiler.TraceAnnotation('optimizer_update'):
            updates, opt_state = self.optimizer.update(total_grads, opt_state, model)
            model = optax.apply_updates(model, updates)
        return (model, opt_state, averaged_metrics)

    def _eval_step(self, model, batch):
        """One sharded eval step. Uses eval_forward_fn if set, else forward_fn."""
        fwd = self.eval_forward_fn if self.eval_forward_fn is not None else self.forward_fn
        with jax.profiler.TraceAnnotation('eval_forward'):
            loss_val, aux = fwd(model, batch)
        return self._compute_metrics(task=self.eval_task, loss=loss_val, aux=aux, batch=batch)

    def __post_init__(self):
        self._setup_sharding()
        self._validate_configs()
        self.logger_obj = self.logger_config.build_logger(config_dict=_serialize_dataclass_config(self))
        self.optimizer: optax.GradientTransformation | None = None
        self.opt_state = None
        self.model = None
        self.ckpt_manager = ocp.CheckpointManager(self.checkpoint_config.checkpoint_dir, options=ocp.CheckpointManagerOptions(max_to_keep=self.checkpoint_config.max_to_keep))
        if self.jit_computations:
            self.train_step_fn = jax.jit(self._train_step, in_shardings=(self.replicated_sharding, self.replicated_sharding, self.data_sharding), out_shardings=(self.replicated_sharding, self.replicated_sharding, self.replicated_sharding))
            self.eval_step_fn = jax.jit(self._eval_step, in_shardings=(self.replicated_sharding, self.data_sharding), out_shardings=self.replicated_sharding)
        else:
            self.train_step_fn = self._train_step
            self.eval_step_fn = self._eval_step

    def init_state(self):
        """Construct the model + optimizer state. Must be called before ``outer_loop``."""
        with jax.set_mesh(self.mesh):
            self.model = self.model_factory(self.model_config, jax.random.key(self.seed))
        self.optimizer = self._create_optimizer()
        with jax.set_mesh(self.mesh):
            self.opt_state = self.optimizer.init(self.model)
        self.step = 0
        self._train_start_time = time.monotonic()
        print(f'Model initialized and replicated across {self.num_devices} devices')

    def _assert_initialized(self) -> None:
        assert self.model is not None and self.opt_state is not None, 'Experiment state not initialized. Call init_state() first.'

    def _unreplicate_for_checkpoint(self, pytree):
        """Pull a single host-side copy of replicated state for serialization."""
        return jax.tree.map(lambda x: np.asarray(x), pytree)

    def save_checkpoint(self):
        self._assert_initialized()
        model_to_save = self._unreplicate_for_checkpoint(self.model)
        opt_state_to_save = self._unreplicate_for_checkpoint(self.opt_state)
        args = {'model': ocp.args.StandardSave(model_to_save), 'opt_state': ocp.args.StandardSave(opt_state_to_save)}
        if self.checkpoint_config.checkpoint_dataset_iterator:
            args['dataset_iterator'] = ocp.args.StandardSave(self.train_task.dataset.get_state())
        self.ckpt_manager.save(self.step, args=ocp.args.Composite(**args))
        self.ckpt_manager.wait_until_finished()

    def _resolve_latest_saved_checkpoint_step(self) -> int:
        raise NotImplementedError('provide a step explicitly!')

    def restore_checkpoint(self, step: int | None=None):
        self._assert_initialized()
        if step is None:
            step = self._resolve_latest_saved_checkpoint_step()
        model_template = self._unreplicate_for_checkpoint(self.model)
        opt_state_template = self._unreplicate_for_checkpoint(self.opt_state)
        args = {'model': ocp.args.StandardRestore(model_template), 'opt_state': ocp.args.StandardRestore(opt_state_template)}
        if self.checkpoint_config.checkpoint_dataset_iterator:
            args['dataset_iterator'] = ocp.args.StandardRestore(self.train_task.dataset.get_state())
        restored = self.ckpt_manager.restore(step, args=ocp.args.Composite(**args))
        self.model = jax.device_put(restored['model'], self.replicated_sharding)
        self.opt_state = jax.device_put(restored['opt_state'], self.replicated_sharding)
        if self.checkpoint_config.checkpoint_dataset_iterator:
            self.train_task.dataset.restore_state(restored['dataset_iterator'])
        self.step = step

    def run_validation(self) -> dict[str, float]:
        """Run validation and return averaged scalar metrics (with ``val/`` prefix).

        Metric values with ndim == 0 are accumulated and averaged across batches.
        Values with ndim > 0 are treated as image batches: those from the first
        eval step are logged once via ``log_images`` and not averaged.
        """
        self._assert_initialized()
        if self.eval_task is None:
            print('No eval task defined, skipping validation.')
            return {}
        accumulated_scalars: dict | None = None
        first_batch_images: dict | None = None
        num_batches = 0
        for ix, batch in enumerate(self.eval_task.dataset):
            batch_metrics = self.eval_step_fn(self.model, batch)
            scalars = {k: v for k, v in batch_metrics.items() if v.ndim == 0}
            images = {k: v for k, v in batch_metrics.items() if v.ndim > 0}
            if accumulated_scalars is None:
                accumulated_scalars = scalars
            else:
                accumulated_scalars = jax.tree.map(lambda x, y: x + y, accumulated_scalars, scalars)
            if first_batch_images is None and images:
                first_batch_images = images
            num_batches += 1
            if ix >= self.eval_config.num_eval_steps:
                break
        if num_batches == 0:
            print('Eval dataset yielded no batches, skipping validation.')
            return {}
        avg_metrics = jax.tree.map(lambda x: float(x) / num_batches, accumulated_scalars)
        avg_metrics = {f'val/{k}': v for k, v in avg_metrics.items()}
        self.logger_obj.log(self.step, metrics=avg_metrics)
        if first_batch_images:
            for k, imgs in first_batch_images.items():
                self.logger_obj.log_images(self.step, f'val/{k}', np.asarray(imgs))
        return avg_metrics

    def sampling_evaluation(self) -> None:
        """Run generative VisualizationMetrics (outside JIT) and log resulting images.

        These metrics don't consume an input batch — they generate images from the
        model directly (e.g. decoding random prior latents). Reconstruction metrics
        that read from ``aux`` belong in the regular ``metrics`` list on the task and
        run inside the JIT-compiled eval step instead.
        """
        if self.eval_task is None or not self.eval_task.visualization_metrics:
            return
        for metric in self.eval_task.visualization_metrics:
            images = metric(self.model)
            for key, imgs in images.items():
                self.logger_obj.log_images(self.step, f'val/{key}', np.asarray(imgs))

    def eval(self):
        self._assert_initialized()
        self.run_validation()
        self.sampling_evaluation()

    def inner_loop(self, batch: jt.PyTree):
        self._assert_initialized()
        self.model, self.opt_state, train_metrics = self.train_step_fn(self.model, self.opt_state, batch)
        if self.step % self.logger_config.train_log_interval_steps == 0:
            train_metrics_with_prefix = {f'train/{k}': float(v) for k, v in train_metrics.items()}
            elapsed = time.monotonic() - self._train_start_time
            if elapsed > 0 and self.step > 0:
                train_metrics_with_prefix['train/steps_per_sec'] = self.step / elapsed
            self.logger_obj.log(self.step, metrics=train_metrics_with_prefix)

    def outer_loop(self):
        finished = self.step >= self.training_config.max_steps
        while True:
            epoch_start_step = self.step
            for batch in self.train_task.dataset:
                with jax.profiler.StepTraceAnnotation('inner_loop', step_num=self.step):
                    self.inner_loop(batch)
                if self.step % self.checkpoint_config.save_interval_steps == 0:
                    self.save_checkpoint()
                if self.step % self.eval_config.eval_interval_steps == 0:
                    self.eval()
                self.step += 1
                if self.step >= self.training_config.max_steps:
                    finished = True
                    break
            if finished:
                break
            if self.step == epoch_start_step:
                raise ValueError(f'Dataset for task {self.train_task.name} is empty.')

    def cleanup(self):
        """Release logger + checkpoint manager. Call once after training."""
        self.logger_obj.close()
        self.ckpt_manager.close()

# ============================================================
# toylib.nn.module - /Users/anuj/Desktop/code/toylib/toylib/nn/module.py
# ============================================================

def _is_array(x: typing.Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic)) or hasattr(x, '__jax_array__')

def _is_random_key(x: str) -> bool:
    return x == 'key'

def _is_supported_container(x: typing.Any) -> bool:
    return isinstance(x, (list, tuple))

def _wrap_init(orig: typing.Callable) -> typing.Callable:

    def wrapped(self) -> None:
        orig(self)
        for v in self.__dict__.values():
            if isinstance(v, Module) and (not hasattr(v, '_trainable_param_keys')):
                v.init()
            elif _is_supported_container(v):
                for elem in v:
                    if isinstance(elem, Module) and (not hasattr(elem, '_trainable_param_keys')):
                        elem.init()
        self._trainable_param_keys = self._get_trainable_param_keys()
        if hasattr(self, 'key'):
            self.key = None
    return wrapped

@dataclasses.dataclass
class Module(abc.ABC):
    """
    Defines a base class to use for the neural network modules in toylib.

    Assumes that all jax arrays are leaf nodes that are trainable and
    everything else is a static param. Defines the flatten and unflatten methods
    to make the modules compatible with jax `jit` and `grad` functions.

    Refer https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees

    Inspired by equinox and the Custom PyTres and Initialization section in jax docs.

    Every subclass automatically receives two dtype fields inherited from this base:

        param_dtype: storage dtype for trainable parameters (default float32).
        dtype: compute dtype for forward-pass operations (default float32).
    """
    param_dtype: np.dtype | type = jnp.float32
    dtype: np.dtype | type = jnp.float32

    def __init_subclass__(cls, **kwargs: typing.Any) -> None:
        """Initialize subclass as a dataclass and register as a pytree node.

        Sub-classes of dataclasses are not automatically dataclasses, so we need to explicitly convert them.
        We also register the class as a pytree with jax so that it can be used with jax transformations like jit and grad.

        Also wraps the subclass's init() to recursively initialize any sub-Module instances
        created during init(), then compute _trainable_param_keys. This means calling init()
        on the top-level module is sufficient to initialize the entire module tree.
        """
        super().__init_subclass__(**kwargs)
        cls = dataclasses.dataclass(cls, kw_only=True)
        cls = jax.tree_util.register_pytree_with_keys_class(cls)
        if 'init' in cls.__dict__:
            original_init = cls.__dict__['init']
            cls.init = _wrap_init(original_init)

    @abc.abstractmethod
    def init(self) -> None:
        """Initialize all the trainable parameters in the module."""
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> typing.Any:
        """Run a forward pass of the module."""
        pass

    def _get_trainable_param_keys(self) -> list[str]:
        """Get the list of attribute names that are trainable parameters."""
        param_keys = []
        for k, v in self.__dict__.items():
            if _is_array(v) and (not _is_random_key(k)) or isinstance(v, Module) or (_is_supported_container(v) and all((isinstance(elem, Module) for elem in v))):
                param_keys.append(k)
        return param_keys

    def tree_flatten_with_keys(self) -> tuple:
        params_with_keys = []
        aux_data = dict()
        for k, v in self.__dict__.items():
            if k not in self._trainable_param_keys:
                aux_data[k] = v
        for k in self._trainable_param_keys:
            v = self.__dict__[k]
            params_with_keys.append((jax.tree_util.GetAttrKey(k), v))
        return (params_with_keys, aux_data)

    @classmethod
    def tree_unflatten(cls, static, dynamic) -> 'Module':
        obj = object.__new__(cls)
        param_keys = static['_trainable_param_keys']
        for k, v in zip(param_keys, dynamic):
            obj.__setattr__(k, v)
        for k, v in static.items():
            obj.__setattr__(k, v)
        return obj

# ============================================================
# toylib.nn.layers - /Users/anuj/Desktop/code/toylib/toylib/nn/layers.py
# ============================================================

class Linear(Module):
    """Defines a simple feedforward layer: which is a linear transformation."""
    in_features: int
    out_features: int
    key: jt.PRNGKeyArray
    use_bias: bool = False
    init_std: typing.Optional[float] = None
    weights: typing.Optional[jt.Float[jt.Array, 'in_features out_features']] = None
    bias: typing.Optional[jt.Float[jt.Array, ' out_features']] = None

    def init(self) -> None:
        w_key = self.key
        in_features = self.in_features
        out_features = self.out_features
        if self.init_std is not None:
            std = self.init_std
            s = std * math.sqrt(3)
            self.weights = jax.random.uniform(key=w_key, shape=(in_features, out_features), minval=-s, maxval=s).astype(self.param_dtype)
        else:
            std = min(1.0, math.sqrt(out_features / in_features)) / math.sqrt(in_features)
            self.weights = (jax.random.normal(key=w_key, shape=(in_features, out_features)) * std).astype(self.param_dtype)
        self.bias = jax.numpy.zeros((out_features,), dtype=self.param_dtype) if self.use_bias else None

    def __call__(self, x: jt.Float[jt.Array, '... in_features']) -> jt.Float[jt.Array, '... out_features']:
        x = jax.numpy.dot(x.astype(self.dtype), self.weights.astype(self.dtype))
        if self.use_bias:
            x = x + self.bias.astype(self.dtype)
        return x

class Embedding(Module):
    """Defines an embedding layer that stores an embedding matrix for discrete tokens."""
    vocab_size: int
    embedding_dim: int
    key: jt.PRNGKeyArray
    weights: typing.Optional[jt.Float[jt.Array, 'vocab_size embedding_dim']] = None

    def init(self) -> None:
        self.weights = jax.random.normal(self.key, (self.vocab_size, self.embedding_dim)).astype(self.param_dtype)

    def __call__(self, tokens: jt.Integer[jt.Array, '... seq_len']) -> jt.Float[jt.Array, '... seq_len embedding_dim']:
        return jax.numpy.take(self.weights, tokens, axis=0).astype(self.dtype)

class Conv2D(Module):
    """2D convolution with NHWC layout, optional bias, and 'SAME' or integer padding.

    Uses ``jax.lax.conv_general_dilated`` under the hood. Kernels are stored as
    ``(kernel_size, kernel_size, in_channels, out_channels)`` (HWIO layout) to
    match the NHWC input convention.

    Weight init follows the same pattern as Linear: if ``init_std`` is set,
    weights are drawn uniformly in ``[-init_std*sqrt(3), +init_std*sqrt(3)]``;
    otherwise a fan-in-aware default ``min(1, sqrt(out/in)) / sqrt(in)`` is
    used (matching arXiv:2310.17813), where ``in/out`` here are the effective
    fan-in / fan-out of the conv (``kernel_size^2 * channels``).
    """
    in_channels: int
    out_channels: int
    key: jt.PRNGKeyArray
    kernel_size: int = 3
    stride: int = 1
    padding: typing.Union[int, str] = 'SAME'
    use_bias: bool = True
    init_std: typing.Optional[float] = None
    weights: typing.Optional[jt.Float[jt.Array, 'kh kw in_channels out_channels']] = None
    bias: typing.Optional[jt.Float[jt.Array, ' out_channels']] = None

    def init(self) -> None:
        k = self.kernel_size
        fan_in = k * k * self.in_channels
        fan_out = k * k * self.out_channels
        if self.init_std is not None:
            std = self.init_std
            s = std * math.sqrt(3)
            self.weights = jax.random.uniform(key=self.key, shape=(k, k, self.in_channels, self.out_channels), minval=-s, maxval=s).astype(self.param_dtype)
        else:
            std = min(1.0, math.sqrt(fan_out / fan_in)) / math.sqrt(fan_in)
            self.weights = (jax.random.normal(key=self.key, shape=(k, k, self.in_channels, self.out_channels)) * std).astype(self.param_dtype)
        self.bias = jnp.zeros((self.out_channels,), dtype=self.param_dtype) if self.use_bias else None

    def _resolved_padding(self) -> typing.Union[str, list[tuple[int, int]]]:
        if isinstance(self.padding, str):
            return self.padding
        p = int(self.padding)
        return [(p, p), (p, p)]

    def __call__(self, x: jt.Float[jt.Array, 'B H W in_channels']) -> jt.Float[jt.Array, 'B H_out W_out out_channels']:
        x = jax.lax.conv_general_dilated(x.astype(self.dtype), self.weights.astype(self.dtype), window_strides=(self.stride, self.stride), padding=self._resolved_padding(), dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        if self.use_bias:
            x = x + self.bias.astype(self.dtype)
        return x

class GroupNorm(Module):
    """Group Normalization over the channel dimension of an NHWC tensor.

    Splits the ``num_features`` channels into ``num_groups`` equal-sized groups
    and normalizes each group's activations (over the (H, W, C/G) volume per
    sample). Learnable per-channel scale and bias are applied after norm.

    Statistics are computed in float32 for numerical stability and cast back
    to ``self.dtype`` on the way out. Matches the convention used by
    ``rms_norm`` below.
    """
    num_features: int
    num_groups: int = 32
    eps: float = 1e-05
    scale: typing.Optional[jt.Float[jt.Array, ' num_features']] = None
    bias: typing.Optional[jt.Float[jt.Array, ' num_features']] = None

    def init(self) -> None:
        if self.num_features % self.num_groups != 0:
            raise ValueError(f'num_features ({self.num_features}) must be divisible by num_groups ({self.num_groups})')
        self.scale = jnp.ones((self.num_features,), dtype=self.param_dtype)
        self.bias = jnp.zeros((self.num_features,), dtype=self.param_dtype)

    def __call__(self, x: jt.Float[jt.Array, 'B H W num_features']) -> jt.Float[jt.Array, 'B H W num_features']:
        orig_dtype = x.dtype
        B, H, W, C = x.shape
        G = self.num_groups
        x32 = x.astype(jnp.float32).reshape(B, H, W, G, C // G)
        mean = jnp.mean(x32, axis=(1, 2, 4), keepdims=True)
        var = jnp.var(x32, axis=(1, 2, 4), keepdims=True)
        x32 = (x32 - mean) * jax.lax.rsqrt(var + self.eps)
        x32 = x32.reshape(B, H, W, C)
        scale = self.scale.astype(jnp.float32).reshape(1, 1, 1, C)
        bias = self.bias.astype(jnp.float32).reshape(1, 1, 1, C)
        return (x32 * scale + bias).astype(orig_dtype)

def upsample_nearest(x: jt.Float[jt.Array, 'B H W C'], factor: int=2) -> jt.Float[jt.Array, 'B H_out W_out C']:
    """Nearest-neighbor upsample of an NHWC image by ``factor`` along H and W.

    Pure function — no trainable parameters. Used in the VAE decoder to expand
    the spatial grid before a regular convolution. Avoids the checkerboard
    artifacts that transposed convolutions exhibit.
    """
    B, H, W, C = x.shape
    return jax.image.resize(x, (B, H * factor, W * factor, C), method='nearest')

def rms_norm(x: jt.Float[jt.Array, '... dim']) -> jt.Float[jt.Array, '... dim']:
    """Applies RMS Normalization over the last dimension of the input tensor.

    The mean-square computation is done in float32 for numerical stability,
    regardless of the input dtype. The output is cast back to the input dtype.

    Args:
        x: Input tensor

    Returns:
        The RMS normalized tensor of the same shape as input x.
    """
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-09)
    return (x / rms).astype(orig_dtype)

# ============================================================
# toylib.nn.attention - /Users/anuj/Desktop/code/toylib/toylib/nn/attention.py
# ============================================================

class RotaryPositionalEmbedding(Module):
    """Implements Rotary Positional Embeddings (RoPE) as described in https://arxiv.org/abs/2104.09864."""
    seq_len: int = 1024
    qkv_dim: int = 128
    base: int = 100000

    def init(self) -> None:
        positions = jnp.arange(0, self.seq_len)
        freqs = self.base ** (jnp.arange(0, self.qkv_dim, 2) / self.qkv_dim)
        self.gamma = einops.einsum(positions, 1.0 / freqs, 't, d -> t d')
        self.cos = jnp.cos(self.gamma).astype(self.param_dtype)
        self.sin = jnp.sin(self.gamma).astype(self.param_dtype)

    def __call__(self, x: jt.Float[jt.Array, '... seq_len qkv_dim'], t0: int=0) -> jt.Float[jt.Array, '... seq_len qkv_dim']:
        t, d = x.shape[-2:]
        if t0 + t > self.seq_len:
            raise ValueError(f'Position index out of range of RoPE cache:t0 ({t0}) + t ({t}) > seq_len ({self.seq_len})')
        sin = self.sin[t0:t0 + t, :].astype(self.dtype)
        cos = self.cos[t0:t0 + t, :].astype(self.dtype)
        x1, x2 = (x[..., :d // 2].astype(self.dtype), x[..., d // 2:].astype(self.dtype))
        es_shape = '... t d, t d -> ... t d'
        y1 = einops.einsum(x1, cos, es_shape) + einops.einsum(x2, sin, es_shape)
        y2 = -einops.einsum(x1, sin, es_shape) + einops.einsum(x2, cos, es_shape)
        return jnp.concatenate([y1, y2], axis=-1)

def scaled_dot_product_attention(q: jt.Float[jt.Array, '... seq_len qkv_dim'], k: jt.Float[jt.Array, '... seq_len qkv_dim'], v: jt.Float[jt.Array, '... seq_len qkv_dim'], mask: typing.Optional[jt.Float[jt.Array, '... seq_len seq_len']]) -> tuple[jt.Float[jt.Array, '... seq_len qkv_dim'], jt.Float[jt.Array, '... seq_len seq_len']]:
    """Compute scaled dot product attention.

    Given query (`q`), key (`k`), and value (`v`) tensors, this function first computes the
    attention weights as the softmax of the dot product of `q` and `k`, scaled by the square
    root of the dimension of the keys. If a mask is provided, it is applied to the attention
    logits before the softmax is computed.

    Finally, the attention weights are used to compute the weighted average of the given values.

    NOTE: the batch dimension is not explicitly handled in this function.

    Args:
        q: query tensor
        k: keys tensor
        v: values tensor
        mask: optional boolean mask to apply to the attention logits

    Returns:
        tuple of final values and attention weights

    """
    d_k = q.shape[-1]
    assert q.shape[-1] == k.shape[-1], 'q and k must have the same feature dimension'
    attention_logits = jnp.matmul(q, k.swapaxes(-1, -2)) / jnp.sqrt(d_k)
    if mask is not None:
        attention_logits = jnp.where(mask, attention_logits, -1000000000.0)
    attention_weights = jax.nn.softmax(attention_logits.astype(jnp.float32), axis=-1).astype(q.dtype)
    values = jnp.matmul(attention_weights, v)
    return (values, attention_weights)

class MultiHeadAttention(Module):
    """
    The MultiHeadAttention defines `num_heads` attention heads. For the given input `Q`, `K`, `V`
    tensors, `num_head` linear projections of dim `qkv_dim / num_heads` are produced.

    An attention weight is then computed using the scaled dot product attention method. The
    weighted average of the values are then concatenated from the various heads to produce a
    single output value vector. A final linear layer is applied on top of this with non-linearity.
    """
    qkv_dim: int
    num_heads: int
    key: jt.PRNGKeyArray
    use_qk_norm: bool = True

    def init(self) -> None:
        qkv_dim = self.qkv_dim
        keys = jax.random.split(self.key, 4)
        init_std = 1 / math.sqrt(qkv_dim)
        self.q_projection = Linear(in_features=qkv_dim, out_features=qkv_dim, use_bias=False, key=keys[0], init_std=init_std, param_dtype=self.param_dtype, dtype=self.dtype)
        self.k_projection = Linear(in_features=qkv_dim, out_features=qkv_dim, use_bias=False, key=keys[1], init_std=init_std, param_dtype=self.param_dtype, dtype=self.dtype)
        self.v_projection = Linear(in_features=qkv_dim, out_features=qkv_dim, use_bias=False, key=keys[2], init_std=init_std, param_dtype=self.param_dtype, dtype=self.dtype)
        self.linear = Linear(in_features=qkv_dim, out_features=qkv_dim, use_bias=False, key=keys[3], init_std=0.0, param_dtype=self.param_dtype, dtype=self.dtype)

    def __call__(self, Q: jt.Float[jt.Array, '... seq_len qkv_dim'], K: jt.Float[jt.Array, '... seq_len qkv_dim'], V: jt.Float[jt.Array, '... seq_len qkv_dim'], mask: typing.Optional[jt.Float[jt.Array, '... seq_len seq_len']]=None, *, rope: typing.Optional[RotaryPositionalEmbedding]=None, return_attention_weights: bool=False) -> typing.Union[tuple[jt.Float[jt.Array, '... seq_len qkv_dim'], jt.Float[jt.Array, '... seq_len seq_len']], jt.Float[jt.Array, '... seq_len qkv_dim']]:
        Q = self.q_projection(Q)
        K = self.k_projection(K)
        V = self.v_projection(V)
        Q = einops.rearrange(Q, '... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim', num_heads=self.num_heads)
        K = einops.rearrange(K, '... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim', num_heads=self.num_heads)
        V = einops.rearrange(V, '... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim', num_heads=self.num_heads)
        if mask is not None:
            mask = einops.rearrange(mask, '... seq_len1 seq_len2 -> ... 1 seq_len1 seq_len2')
        if rope is not None:
            Q = rope(Q)
            K = rope(K)
        if self.use_qk_norm:
            Q = rms_norm(Q)
            K = rms_norm(K)
        values, attention_weights = scaled_dot_product_attention(q=Q, k=K, v=V, mask=mask)
        values = einops.rearrange(values, '... num_heads seq_len d -> ... seq_len (num_heads d)')
        values = self.linear(values)
        if return_attention_weights:
            return (values, attention_weights)
        return values

# ============================================================
# toylib_projects.wm.vision_encoder.model - /Users/anuj/Desktop/code/toylib/toylib_projects/wm/vision_encoder/model.py
# ============================================================

"""KL-regularized VAE for the Track A1 vision codec.

Implements the architecture specified in
``docs/walkthroughs/a1_vision_codec.md`` (which is the actionable
distillation of ``docs/designs/vision_codec.md``), with one deliberate
deviation from the walkthrough: we run on the Stage 2 default
**128×128** frames rather than the walkthrough's 64×64 spec. The
architecture is unchanged (still 3 stride-2 downsample stages); this
just keeps the latent grid at ``H / 8``, so the new shapes are:

  - Input  : (B, 128, 128, 3) float32 in [-1, 1]
  - Latent : (B,  16,  16, 4) float32  — 256 tokens per frame for downstream models
  - Output : (B, 128, 128, 3) float32 in (-1, 1)   (Tanh head)

Encoder pipeline (per the walkthrough — unchanged):
    Conv → ResBlock → Down → ResBlock → Down → ResBlock → Down →
    ResBlock → AttnBlock → ResBlock → GN+SiLU → Conv → split (μ, log σ²)

Decoder mirrors it, with each up-step implemented as
``upsample_nearest → Conv2D`` (channel-preserving) followed by a
ResBlock for channel reduction. Ends with Tanh so outputs land in (-1, 1).

Built on the toylib ``Module`` base class (dataclass-style,
pytree-registered). All convolutions / GroupNorm / nearest-neighbor
upsample come from ``toylib.nn.layers``; attention is reused from
``toylib.nn.attention``.

Loss pieces live in this file as pure functions:
  - ``reparameterize``                — the differentiable ε-trick
  - ``kl_divergence``                  — closed-form KL(q(z|x) || N(0, I))
  - ``recon_loss_l1``                  — mean L1 over (-1, 1) targets
  - ``beta_warmup``                    — linear KL warmup schedule
  - ``vae_loss``                       — assembles the train-time loss

Perceptual + auxiliary losses (walkthrough Milestone 5) are deliberately
left out of this file — add them in a wrapper once base reconstruction is
stable.
"""

@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the Track A1 VAE.

    Defaults match the walkthrough: 64×64 input, 8×8×4 latent, base_ch=64
    (so 4×base_ch = 256 channels at the bottleneck).
    """
    base_ch: int = 64
    latent_channels: int = 4
    input_channels: int = 3
    num_attn_heads: int = 1
    num_norm_groups: int = 32
    log_sigma_sq_clip_min: float = -30.0
    log_sigma_sq_clip_max: float = 20.0

class ResBlock(Module):
    """Two-conv pre-activation residual block.

    GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv, plus a skip
    connection. When ``in_channels != out_channels`` the skip path goes
    through a 1×1 conv to match shapes (otherwise the skip is identity).
    """
    in_channels: int
    out_channels: int
    key: jt.PRNGKeyArray
    num_groups: int = 32

    def init(self) -> None:
        keys = jax.random.split(self.key, 3)
        self.norm1 = GroupNorm(num_features=self.in_channels, num_groups=self.num_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.conv1 = Conv2D(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding='SAME', key=keys[0], param_dtype=self.param_dtype, dtype=self.dtype)
        self.norm2 = GroupNorm(num_features=self.out_channels, num_groups=self.num_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.conv2 = Conv2D(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding='SAME', key=keys[1], param_dtype=self.param_dtype, dtype=self.dtype)
        if self.in_channels != self.out_channels:
            self.skip = Conv2D(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, padding='SAME', key=keys[2], param_dtype=self.param_dtype, dtype=self.dtype)
        else:
            self.skip = None

    def __call__(self, x: jt.Float[jt.Array, 'B H W in_channels']) -> jt.Float[jt.Array, 'B H W out_channels']:
        h = jax.nn.silu(self.norm1(x))
        h = self.conv1(h)
        h = jax.nn.silu(self.norm2(h))
        h = self.conv2(h)
        skip = x if self.skip is None else self.skip(x)
        return h + skip

class AttentionBlock(Module):
    """Single self-attention block at the spatial bottleneck.

    Pre-norm with GroupNorm, flatten the (H, W) grid into an (H*W)-long
    sequence, run multi-head self-attention, reshape back, residual add.

    Reuses the existing toylib ``MultiHeadAttention``; the output linear
    inside it is zero-initialized, so this block is the identity at init
    (helpful for training stability).
    """
    channels: int
    key: jt.PRNGKeyArray
    num_heads: int = 1
    num_groups: int = 32

    def init(self) -> None:
        self.norm = GroupNorm(num_features=self.channels, num_groups=self.num_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.attn = MultiHeadAttention(qkv_dim=self.channels, num_heads=self.num_heads, key=self.key, param_dtype=self.param_dtype, dtype=self.dtype)

    def __call__(self, x: jt.Float[jt.Array, 'B H W C']) -> jt.Float[jt.Array, 'B H W C']:
        B, H, W, C = x.shape
        h = self.norm(x)
        h_seq = h.reshape(B, H * W, C)
        h_seq = self.attn(h_seq, h_seq, h_seq)
        h = h_seq.reshape(B, H, W, C)
        return x + h

class Encoder(Module):
    """Down-3× conv encoder producing per-spatial Gaussian parameters.

    Output is a single tensor of ``(B, 8, 8, 2C)`` which is split along the
    channel axis into ``μ`` and ``log σ²``. ``log σ²`` is clipped before
    any downstream ``exp`` to avoid NaNs at init.
    """
    config: ModelConfig
    key: jt.PRNGKeyArray

    def init(self) -> None:
        cfg = self.config
        ch = cfg.base_ch
        keys = jax.random.split(self.key, 12)
        self.conv_in = Conv2D(in_channels=cfg.input_channels, out_channels=ch, kernel_size=3, padding='SAME', key=keys[0], param_dtype=self.param_dtype, dtype=self.dtype)
        self.res1 = ResBlock(in_channels=ch, out_channels=ch, key=keys[1], num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.down1 = Conv2D(in_channels=ch, out_channels=2 * ch, kernel_size=3, stride=2, padding='SAME', key=keys[2], param_dtype=self.param_dtype, dtype=self.dtype)
        self.res2 = ResBlock(in_channels=2 * ch, out_channels=2 * ch, key=keys[3], num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.down2 = Conv2D(in_channels=2 * ch, out_channels=4 * ch, kernel_size=3, stride=2, padding='SAME', key=keys[4], param_dtype=self.param_dtype, dtype=self.dtype)
        self.res3 = ResBlock(in_channels=4 * ch, out_channels=4 * ch, key=keys[5], num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.down3 = Conv2D(in_channels=4 * ch, out_channels=4 * ch, kernel_size=3, stride=2, padding='SAME', key=keys[6], param_dtype=self.param_dtype, dtype=self.dtype)
        self.res4 = ResBlock(in_channels=4 * ch, out_channels=4 * ch, key=keys[7], num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.attn = AttentionBlock(channels=4 * ch, num_heads=cfg.num_attn_heads, key=keys[8], num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.res5 = ResBlock(in_channels=4 * ch, out_channels=4 * ch, key=keys[9], num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.norm_out = GroupNorm(num_features=4 * ch, num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.conv_out = Conv2D(in_channels=4 * ch, out_channels=2 * cfg.latent_channels, kernel_size=3, padding='SAME', key=keys[10], param_dtype=self.param_dtype, dtype=self.dtype)

    def __call__(self, x: jt.Float[jt.Array, 'B 128 128 3']) -> tuple[jt.Float[jt.Array, 'B 16 16 latent_channels'], jt.Float[jt.Array, 'B 16 16 latent_channels']]:
        h = self.conv_in(x)
        h = self.res1(h)
        h = self.down1(h)
        h = self.res2(h)
        h = self.down2(h)
        h = self.res3(h)
        h = self.down3(h)
        h = self.res4(h)
        h = self.attn(h)
        h = self.res5(h)
        h = jax.nn.silu(self.norm_out(h))
        h = self.conv_out(h)
        mu, log_sigma_sq = jnp.split(h, 2, axis=-1)
        log_sigma_sq = jnp.clip(log_sigma_sq, self.config.log_sigma_sq_clip_min, self.config.log_sigma_sq_clip_max)
        return (mu, log_sigma_sq)

class Decoder(Module):
    """Mirror decoder: bottleneck attention then 3× nearest-neighbor upsample.

    Each upsample step is implemented as ``upsample_nearest → Conv2D``
    (channel-preserving smoothing conv), followed by a ResBlock that does
    the channel reduction. This avoids the checkerboard artifacts of
    transposed convolutions.
    """
    config: ModelConfig
    key: jt.PRNGKeyArray

    def init(self) -> None:
        cfg = self.config
        ch = cfg.base_ch
        keys = jax.random.split(self.key, 12)
        self.conv_in = Conv2D(in_channels=cfg.latent_channels, out_channels=4 * ch, kernel_size=3, padding='SAME', key=keys[0], param_dtype=self.param_dtype, dtype=self.dtype)
        self.res1 = ResBlock(in_channels=4 * ch, out_channels=4 * ch, key=keys[1], num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.attn = AttentionBlock(channels=4 * ch, num_heads=cfg.num_attn_heads, key=keys[2], num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.res2 = ResBlock(in_channels=4 * ch, out_channels=4 * ch, key=keys[3], num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.up1_conv = Conv2D(in_channels=4 * ch, out_channels=4 * ch, kernel_size=3, padding='SAME', key=keys[4], param_dtype=self.param_dtype, dtype=self.dtype)
        self.res3 = ResBlock(in_channels=4 * ch, out_channels=2 * ch, key=keys[5], num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.up2_conv = Conv2D(in_channels=2 * ch, out_channels=2 * ch, kernel_size=3, padding='SAME', key=keys[6], param_dtype=self.param_dtype, dtype=self.dtype)
        self.res4 = ResBlock(in_channels=2 * ch, out_channels=ch, key=keys[7], num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.up3_conv = Conv2D(in_channels=ch, out_channels=ch, kernel_size=3, padding='SAME', key=keys[8], param_dtype=self.param_dtype, dtype=self.dtype)
        self.res5 = ResBlock(in_channels=ch, out_channels=ch, key=keys[9], num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.norm_out = GroupNorm(num_features=ch, num_groups=cfg.num_norm_groups, param_dtype=self.param_dtype, dtype=self.dtype)
        self.conv_out = Conv2D(in_channels=ch, out_channels=cfg.input_channels, kernel_size=3, padding='SAME', key=keys[10], param_dtype=self.param_dtype, dtype=self.dtype)

    def __call__(self, z: jt.Float[jt.Array, 'B 16 16 latent_channels']) -> jt.Float[jt.Array, 'B 128 128 3']:
        h = self.conv_in(z)
        h = self.res1(h)
        h = self.attn(h)
        h = self.res2(h)
        h = upsample_nearest(h, factor=2)
        h = self.up1_conv(h)
        h = self.res3(h)
        h = upsample_nearest(h, factor=2)
        h = self.up2_conv(h)
        h = self.res4(h)
        h = upsample_nearest(h, factor=2)
        h = self.up3_conv(h)
        h = self.res5(h)
        h = jax.nn.silu(self.norm_out(h))
        h = self.conv_out(h)
        return jnp.tanh(h)

def reparameterize(mu: jt.Float[jt.Array, 'B h w C'], log_sigma_sq: jt.Float[jt.Array, 'B h w C'], rng_key: jt.PRNGKeyArray) -> jt.Float[jt.Array, 'B h w C']:
    """Reparameterization trick: ``z = μ + σ · ε`` with ``ε ~ N(0, I)``.

    Differentiable in both ``μ`` and ``log σ²``; the only non-differentiable
    bit (the normal sample) is held in ``ε`` and gradients flow around it.

    At **inference** time, prefer ``z = μ`` directly (no noise) — this
    function is only needed during VAE training.
    """
    sigma = jnp.exp(0.5 * log_sigma_sq)
    eps = jax.random.normal(rng_key, mu.shape, dtype=mu.dtype)
    return mu + sigma * eps

def kl_divergence(mu: jt.Float[jt.Array, 'B h w C'], log_sigma_sq: jt.Float[jt.Array, 'B h w C']) -> jt.Float[jt.Array, '']:
    """Closed-form KL( N(μ, σ²) || N(0, I) ), summed over latent dims.

    ``L_KL = 0.5 · mean_B( sum_{h,w,C} (μ² + σ² − log σ² − 1) )``.

    Sum over (h, w, C) **then** mean over the batch — matches the Stable
    Diffusion / walkthrough convention. Swapping sum/mean here scales the
    loss magnitude by ``h*w*C`` (256 for the default config).
    """
    sigma_sq = jnp.exp(log_sigma_sq)
    per_sample = 0.5 * jnp.sum(mu ** 2 + sigma_sq - log_sigma_sq - 1.0, axis=(1, 2, 3))
    return jnp.mean(per_sample)

def recon_loss_l1(recon: jt.Float[jt.Array, 'B H W C'], target: jt.Float[jt.Array, 'B H W C']) -> jt.Float[jt.Array, '']:
    """Mean L1 over pixels. Both args are assumed in ``[-1, 1]``."""
    return jnp.mean(jnp.abs(recon - target))

def beta_warmup(step: int | jt.Array, warmup_steps: int, beta_max: float) -> jt.Float[jt.Array, '']:
    """Linear KL warmup: ``β(step) = (step / warmup_steps) · β_max``, capped.

    Steps 0..warmup_steps ramp from 0 → β_max; thereafter β stays at β_max.
    The walkthrough recommends ``β_max = 1e-6`` and
    ``warmup_steps = 10_000`` to prevent posterior collapse early in
    training. Returns a jnp scalar so it can be threaded through jit.
    """
    if warmup_steps <= 0:
        return jnp.asarray(beta_max, jnp.float32)
    frac = jnp.minimum(jnp.asarray(step, jnp.float32) / warmup_steps, 1.0)
    return (frac * beta_max).astype(jnp.float32)

class VAE(Module):
    """Encoder + decoder bundled together.

    For inference, call ``encode`` and ``decode`` directly. For training,
    use ``__call__(x, rng_key)`` which returns ``(recon, aux)`` where
    ``aux`` contains ``mu``, ``log_sigma_sq``, and ``z`` for downstream
    loss computation.
    """
    config: ModelConfig
    key: jt.PRNGKeyArray

    def init(self) -> None:
        keys = jax.random.split(self.key, 2)
        self.encoder = Encoder(config=self.config, key=keys[0], param_dtype=self.param_dtype, dtype=self.dtype)
        self.decoder = Decoder(config=self.config, key=keys[1], param_dtype=self.param_dtype, dtype=self.dtype)

    def encode(self, x: jt.Float[jt.Array, 'B 128 128 3']) -> tuple[jt.Float[jt.Array, 'B 16 16 C'], jt.Float[jt.Array, 'B 16 16 C']]:
        return self.encoder(x)

    def decode(self, z: jt.Float[jt.Array, 'B 16 16 C']) -> jt.Float[jt.Array, 'B 128 128 3']:
        return self.decoder(z)

    def __call__(self, x: jt.Float[jt.Array, 'B 128 128 3'], rng_key: typing.Optional[jt.PRNGKeyArray]=None) -> tuple[jt.Float[jt.Array, 'B 128 128 3'], dict[str, jt.Array]]:
        mu, log_sigma_sq = self.encode(x)
        if rng_key is None:
            z = mu
        else:
            z = reparameterize(mu, log_sigma_sq, rng_key)
        recon = self.decode(z)
        return (recon, {'mu': mu, 'log_sigma_sq': log_sigma_sq, 'z': z})

def vae_loss(model: VAE, batch: jt.Float[jt.Array, 'B 128 128 3'], rng_key: jt.PRNGKeyArray, beta: jt.Float[jt.Array, ''] | float=1e-06) -> tuple[jt.Float[jt.Array, ''], dict[str, jt.Array]]:
    """Base VAE training loss: ``L_rec + β · L_KL``.

    Inputs are expected in ``[-1, 1]`` float32. Returns ``(loss, aux)`` where
    ``aux`` contains the individual loss components plus the model's
    intermediate tensors — suitable for plugging into the existing
    ``Experiment.forward_fn`` contract.

    Perceptual + auxiliary ball-position losses (walkthrough Milestone 5)
    are deliberately not included here; add them in a wrapper once base
    reconstruction is healthy.
    """
    recon, model_aux = model(batch, rng_key=rng_key)
    l_rec = recon_loss_l1(recon, batch)
    l_kl = kl_divergence(model_aux['mu'], model_aux['log_sigma_sq'])
    total = l_rec + beta * l_kl
    aux = {'l_rec': l_rec, 'l_kl': l_kl, 'beta': jnp.asarray(beta, jnp.float32), 'recon': recon, **model_aux}
    return (total, aux)

# ============================================================
# None - ../wm/vision_encoder/train.py
# ============================================================

"""Training entry point for the Track A1 KL-VAE vision codec.

Mirrors the shape of ``toylib_projects/tinystories/train.py``: a
``create_experiment`` factory that wires together the dataset, model,
optimizer, logger, and checkpoint manager into a ready-to-run
``Experiment``, plus a ``main`` that exposes a small CLI.

Usage (interactive / Colab)::

    from vision_encoder import train

    exp = train.create_experiment(
        train_path="data/compiled/vae_train.h5",
        val_path="data/compiled/vae_val.h5",
        max_steps=1000,
        batch_size_per_device=16,
    )
    exp.outer_loop()
    exp.cleanup()

Usage (CLI)::

    uv run python -m vision_encoder.train \\
        --train-path data/compiled/vae_train.h5 \\
        --val-path   data/compiled/vae_val.h5 \\
        --max-steps  100 \\
        --batch-size-per-device 8


Design notes
------------

**Normalization lives in ``forward_fn``.** The compiled VAE dataset stores
uint8 frames in ``[0, 255]``. The VAE consumes float32 in ``[-1, 1]``. We
convert inside the forward function (a tiny one-liner) so the loader stays
pure and the same conversion applies in eval/inference. The cost is a
single mul/sub op per batch.

**Reparameterization RNG.** The base ``Experiment._train_step`` doesn't
thread an RNG into ``forward_fn`` — the only inputs are ``(model,
opt_state, batch)``. For this v1 we therefore call ``vae_loss`` with a
**fixed** ``rng_key`` per step. Practical implications:

  - Gradients still flow through the reparameterization trick (it's
    differentiable in μ and log σ²).
  - The noise sample ε is the same on every step. The encoder is being
    asked to fit a deterministic posterior — closer to a plain AE than a
    true VAE.
  - For smoke-testing the harness end-to-end this is fine; for a real
    training run, thread a per-step key through ``Experiment`` (TODO
    below) before drawing conclusions about VAE behavior.

**β warmup is not yet plumbed through training.** ``vae_loss`` accepts a
``beta`` argument and ``beta_warmup`` computes the right schedule, but
``Experiment._train_step`` doesn't know about the current step
inside the JIT'd compute. For v1 we pass a constant ``beta``. Hooking up
the schedule cleanly requires either passing ``step`` into ``forward_fn``
or building it into a stateful model — TODO.
"""

def make_model_factory() -> typing.Callable:
    """Return a factory that builds an initialized VAE from (config, key)."""

    def factory(config: ModelConfig, key) -> VAE:
        vae = VAE(config=config, key=key)
        vae.init()
        return vae
    return factory

def make_forward_fn(beta: float, rng_seed: int=0) -> typing.Callable:
    """Build a ``(model, batch) -> (loss, aux)`` closure for training.

    Strips all large tensors from aux — only scalar losses pass through the
    microbatch scan to keep memory flat.
    """
    fixed_key = jax.random.key(rng_seed)

    def forward_fn(model: VAE, batch):
        frames = batch.astype(jnp.float32) / 127.5 - 1.0
        loss, aux = vae_loss(model, frames, rng_key=fixed_key, beta=beta)
        return (loss, {'l_rec': aux['l_rec'], 'l_kl': aux['l_kl']})
    return forward_fn

def make_eval_forward_fn(beta: float, rng_seed: int=0) -> typing.Callable:
    """Like ``make_forward_fn`` but also returns ``recon`` in aux for visualization."""
    fixed_key = jax.random.key(rng_seed)

    def eval_forward_fn(model: VAE, batch):
        frames = batch.astype(jnp.float32) / 127.5 - 1.0
        loss, aux = vae_loss(model, frames, rng_key=fixed_key, beta=beta)
        return (loss, {'l_rec': aux['l_rec'], 'l_kl': aux['l_kl'], 'recon': aux['recon']})
    return eval_forward_fn

def make_sample_fn(latent_channels: int) -> typing.Callable:
    """Return a ``(model, key, n) -> uint8 images`` function for prior sampling.

    Samples ``n`` latents from ``N(0, I)`` at the encoder's output spatial size
    (16×16 for 128×128 inputs with 8× downsampling), decodes them, and converts
    the float32 ``[-1, 1]`` output to uint8 ``[0, 255]``.
    """
    latent_spatial = 16

    def sample_fn(model: VAE, key, n: int):
        z = jax.random.normal(key, shape=(n, latent_spatial, latent_spatial, latent_channels))
        recon = model.decode(z)
        return jnp.clip((recon + 1.0) * 127.5, 0, 255).astype(jnp.uint8)
    return sample_fn

@dataclasses.dataclass
class VaeAuxMetric:
    """Surface the per-step scalar VAE losses (``l_rec``, ``l_kl``) as named metrics."""
    keys: tuple[str, ...] = ('l_rec', 'l_kl')

    def __call__(self, loss, aux, batch):
        del loss, batch
        return {k: aux[k] for k in self.keys if k in aux}

def create_experiment(train_path: str | Path, val_path: str | Path | None=None, *, base_ch: int=64, latent_channels: int=4, batch_size_per_device: int=16, num_microbatches: int=1, max_steps: int=1000, learning_rate: float=0.0001, max_grad_norm: float=1.0, beta: float=1e-06, eval_interval_steps: int=100, num_eval_steps: int=4, save_interval_steps: int=1000, checkpoint_dir: str='/tmp/wm_vae_ckpt', log_dir: str='/tmp/wm_vae_logs', run_id: str | None=None, num_recon_images: int=8, num_prior_samples: int=16, wandb_project: str | None=None, wandb_user: str | None=None, seed: int=0, jit_computations: bool=True) -> Experiment:
    """Wire datasets, model, optimizer, logger and checkpointer into an Experiment."""
    if run_id is None:
        run_id = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    n_devices = jax.local_device_count()
    batch_size = batch_size_per_device * n_devices * num_microbatches
    train_ds = Hdf5FramesDataset(dataset_path=str(train_path), batch_size=batch_size, seed=seed, shuffle=True, drop_remainder=True, repeat=True)
    train_task = Task(name='train', dataset=train_ds, metrics=[Loss(), VaeAuxMetric()])
    eval_task = None
    if val_path is not None:
        val_ds = Hdf5FramesDataset(dataset_path=str(val_path), batch_size=batch_size, seed=seed, shuffle=False, drop_remainder=False, repeat=True)
        eval_task = Task(name='val', dataset=val_ds, metrics=[Loss(), VaeAuxMetric(), ReconstructionVisualization(num_images=num_recon_images)], visualization_metrics=[PriorSamplingVisualization(sample_fn=make_sample_fn(latent_channels), num_samples=num_prior_samples)])
    optimizer_config = MultiOptimizerConfig(optimizer_configs=[OptimizerConfig(name='adam_all', optimizer=optax.adam(learning_rate=learning_rate))], optimizer_for_param=lambda path: 'adam_all')
    if wandb_project is not None:
        if wandb_user is None:
            raise ValueError('--wandb-user is required when --wandb-project is set')
        logger_config = WandBLoggerConfig(project_name=wandb_project, user_name=wandb_user, log_dir=log_dir, run_id=run_id)
    else:
        logger_config = LoggerConfig(log_dir=log_dir, run_id=run_id)
    return Experiment(train_task=train_task, eval_task=eval_task, forward_fn=make_forward_fn(beta=beta), eval_forward_fn=make_eval_forward_fn(beta=beta) if eval_task is not None else None, model_factory=make_model_factory(), model_config=ModelConfig(base_ch=base_ch, latent_channels=latent_channels), training_config=TrainingConfig(max_steps=max_steps, num_microbatches=num_microbatches, max_grad_norm=max_grad_norm, optimizer_config=optimizer_config), eval_config=EvalConfig(eval_interval_steps=eval_interval_steps, num_eval_steps=num_eval_steps), checkpoint_config=CheckpointConfig(save_interval_steps=save_interval_steps, checkpoint_dir=f'{checkpoint_dir}/{run_id}'), logger_config=logger_config, seed=seed, jit_computations=jit_computations)