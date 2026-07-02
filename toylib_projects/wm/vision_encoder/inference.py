"""Shared inference helpers for the Track A1 KL-VAE vision codec.

This is the one place that knows how to **load a trained VAE and push frames
through it**. Both the evaluation script (``vision_encoder.evaluate``) and the
Stage 2 data-generation pipeline import these helpers, so the exact same
encode/decode path — normalization, deterministic ``z = mu``, batching — is
used everywhere. That guarantees the latents cached for the world model are
bit-for-bit the ones the evaluation measured.

Everything here is **Colab-friendly**: plain functions with keyword defaults,
numpy in / numpy out, no ``Experiment`` / ``Task`` scaffolding required. A
typical interactive session is::

    from toylib_projects.wm.vision_encoder import inference

    frames, source, cfg = inference.load_frames("data/compiled/vae_test.h5", n=256)
    vae = inference.load_vae("/path/to/ckpt/<run_id>")   # latest step by default
    recons = inference.reconstruct(vae, frames)          # (256, H, W, 3) uint8

Design notes
------------

**Deterministic encoding.** At inference we use ``z = mu`` (no reparameterization
noise) — see ``model.VAE.__call__``'s inference branch. ``encode_frames`` returns
``mu`` directly.

**Normalization lives here.** The compiled datasets store uint8 ``[0, 255]``
frames; the VAE consumes float32 ``[-1, 1]``. ``encode_frames`` / ``decode_latents``
apply the ``/127.5 - 1`` map (and its inverse) so callers pass and receive plain
uint8 frames.

**Batching in host code.** The jitted encode/decode operate on a fixed device
batch; the Python wrappers chunk arbitrarily large inputs into ``batch_size``
pieces so a 20k-frame test set fits without OOM. The final short batch is padded
up to ``batch_size`` and trimmed back, so the jitted function only ever sees one
input shape (one compile).
"""

from __future__ import annotations

import functools
import json
import typing
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — registers LZ4/zstd filters before any h5py read
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from toylib_projects.wm.datagen.preprocess_frames import PreprocessConfig
from toylib_projects.wm.vision_encoder import model as model_lib


# ──────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────


def latest_step(checkpoint_dir: str | Path) -> int:
    """Return the most recent step saved under ``checkpoint_dir``.

    Raises ``ValueError`` if the directory holds no checkpoints. ``checkpoint_dir``
    may be a local path or a ``gs://bucket/...`` URI.
    """
    manager = ocp.CheckpointManager(
        str(checkpoint_dir), options=ocp.CheckpointManagerOptions()
    )
    try:
        step = manager.latest_step()
    finally:
        manager.close()
    if step is None:
        raise ValueError(f"No checkpoints found under {checkpoint_dir!r}")
    return int(step)


def load_vae(
    checkpoint_dir: str | Path,
    step: int | None = None,
    *,
    base_ch: int = 64,
    latent_channels: int = 4,
    config: model_lib.ModelConfig | None = None,
    seed: int = 0,
    key: jax.random.PRNGKey | None = None,
) -> model_lib.VAE:
    """Restore a VAE saved by ``vision_encoder.train``.

    Builds a template VAE with a matching config (so the pytree structure lines
    up with the saved arrays), then restores just the ``model`` item from the
    composite checkpoint (optimizer / dataset-iterator items are ignored).

    Parameters
    ----------
    checkpoint_dir :
        Directory holding the orbax checkpoint (the ``.../<run_id>`` folder that
        ``train`` writes to). Local path or ``gs://bucket/...`` URI.
    step :
        Which saved step to restore. ``None`` (default) restores the latest.
    base_ch, latent_channels :
        VAE shape — **must match the trained checkpoint**. Ignored if ``config``
        is given.
    config :
        Full ``ModelConfig``; overrides ``base_ch`` / ``latent_channels`` if set.
    seed :
        Seed for the template's (immediately overwritten) init weights. Ignored if `key` is set.
    key: Optional random key for weights init.

    Returns
    -------
    model_lib.VAE :
        The restored model, ready for ``encode`` / ``decode``.
    """
    if config is None:
        config = model_lib.ModelConfig(base_ch=base_ch, latent_channels=latent_channels)
    if step is None:
        step = latest_step(checkpoint_dir)
    if key is None:
        key = jax.random.key(seed)

    template_vae = model_lib.VAE(config=config, key=key)
    template_vae.init()
    template = jax.tree.map(np.asarray, template_vae)

    manager = ocp.CheckpointManager(
        str(checkpoint_dir), options=ocp.CheckpointManagerOptions()
    )
    try:
        restored = manager.restore(
            step,
            args=ocp.args.Composite(model=ocp.args.StandardRestore(template)),
        )
    finally:
        manager.close()
    # Bring host arrays onto the active device/mesh.
    return jax.tree.map(jnp.asarray, restored["model"])


# ──────────────────────────────────────────────────────────────────────────
# Data loading convenience (for Colab / notebooks)
# ──────────────────────────────────────────────────────────────────────────


def preproc_from_h5(path: str | Path) -> PreprocessConfig:
    """Reconstruct the ``PreprocessConfig`` a compiled dataset was built with.

    Reads the ``config_json`` attr written by ``generate_vision_enc_data`` and
    rebuilds the exact crop/resize config, so ``ram_to_pixel`` maps state
    coordinates onto the same pixels the VAE saw. Falls back to defaults if the
    attr is absent.
    """
    with h5py.File(str(path), "r") as f:
        raw = f.attrs.get("config_json")
    if raw is None:
        return PreprocessConfig()
    pp = json.loads(raw)["preprocess"]
    return PreprocessConfig(
        crop_top=pp["crop_top"],
        crop_bottom=pp["crop_bottom"],
        crop_left=pp["crop_left"],
        crop_right=pp["crop_right"],
        target_h=pp["target_h"],
        target_w=pp["target_w"],
        resize_filter=pp["resize_filter"],
    )


# State keys kept alongside frames in the compiled files (see
# ``generate_vision_enc_data`` output schema). Used to load ground-truth
# positions + stratification keys for the region metrics.
SOURCE_KEYS: tuple[str, ...] = (
    "ball_x",
    "ball_y",
    "paddle_x",
    "mode",
    "score",
)


def load_frames(
    path: str | Path,
    n: int | None = None,
    start: int = 0,
    *,
    source_keys: typing.Sequence[str] = SOURCE_KEYS,
) -> tuple[np.ndarray, dict[str, np.ndarray], PreprocessConfig]:
    """Load a slice of frames + per-frame source state from a compiled ``.h5``.

    Convenience for notebooks: one call gives you the uint8 frames, a dict of
    the per-frame state arrays needed for the region metrics / stratification,
    and the ``PreprocessConfig`` for coordinate mapping.

    Parameters
    ----------
    path :
        A compiled ``vae_{train,val,test}.h5``.
    n :
        Number of frames to read (``None`` = all from ``start``).
    start :
        First frame index.
    source_keys :
        Which ``source/<key>`` arrays to return.

    Returns
    -------
    (frames, source, config) :
        ``frames`` ``(n, H, W, 3)`` uint8; ``source`` maps each key to an
        ``(n,)`` array; ``config`` the dataset's ``PreprocessConfig``.
    """
    with h5py.File(str(path), "r") as f:
        total = int(f.attrs["n_frames"])
        stop = total if n is None else min(start + n, total)
        frames = f["frames"][start:stop]
        source = {}
        for k in source_keys:
            ds = f.get(f"source/{k}")
            if ds is not None:
                source[k] = np.asarray(ds[start:stop])
    return np.asarray(frames), source, preproc_from_h5(path)


# ──────────────────────────────────────────────────────────────────────────
# Encode / decode (batched host wrappers over jitted core)
# ──────────────────────────────────────────────────────────────────────────


@functools.partial(jax.jit, static_argnums=())
def _encode_batch(vae: model_lib.VAE, frames_f32: jnp.ndarray) -> jnp.ndarray:
    """Deterministic encode of one device batch: returns ``mu`` (``z = mu``)."""
    mu, _ = vae.encode(frames_f32)
    return mu


@functools.partial(jax.jit, static_argnums=())
def _decode_batch(vae: model_lib.VAE, latents: jnp.ndarray) -> jnp.ndarray:
    """Decode one device batch to float32 ``[-1, 1]`` frames."""
    return vae.decode(latents)


def _iter_padded_batches(n: int, batch_size: int) -> typing.Iterator[tuple[int, int]]:
    """Yield ``(start, count)`` spans covering ``[0, n)`` in ``batch_size`` steps."""
    for start in range(0, n, batch_size):
        yield start, min(batch_size, n - start)


def encode_frames(
    vae: model_lib.VAE,
    frames: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode uint8 frames to deterministic latents ``mu``.

    Parameters
    ----------
    vae :
        A loaded ``VAE`` (see :func:`load_vae`).
    frames :
        ``(N, H, W, 3)`` uint8 in ``[0, 255]``.
    batch_size :
        Device batch size. Every batch is padded up to this size so the jitted
        encoder compiles exactly once.

    Returns
    -------
    np.ndarray :
        ``(N, h, w, C)`` float32 latents (``h = H/8``, ``w = W/8``).
    """
    frames = np.asarray(frames)
    n = frames.shape[0]
    out: list[np.ndarray] = []
    for start, count in _iter_padded_batches(n, batch_size):
        chunk = frames[start : start + count]
        if count < batch_size:  # pad the final short batch up to batch_size
            pad = np.repeat(chunk[-1:], batch_size - count, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunk_f32 = chunk.astype(np.float32) / 127.5 - 1.0
        mu = _encode_batch(vae, jnp.asarray(chunk_f32))
        out.append(np.asarray(mu)[:count])
    return np.concatenate(out, axis=0)


def decode_latents(
    vae: model_lib.VAE,
    latents: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Decode latents to uint8 frames in ``[0, 255]``.

    Inverse of :func:`encode_frames`: float32 ``[-1, 1]`` decoder output is
    mapped back to uint8. Accepts ``(N, h, w, C)`` and returns ``(N, H, W, 3)``.
    """
    latents = np.asarray(latents)
    n = latents.shape[0]
    out: list[np.ndarray] = []
    for start, count in _iter_padded_batches(n, batch_size):
        chunk = latents[start : start + count]
        if count < batch_size:
            pad = np.repeat(chunk[-1:], batch_size - count, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        recon = _decode_batch(vae, jnp.asarray(chunk.astype(np.float32)))
        recon_u8 = np.clip((np.asarray(recon) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        out.append(recon_u8[:count])
    return np.concatenate(out, axis=0)


def reconstruct(
    vae: model_lib.VAE,
    frames: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Full round-trip ``decode(encode(frames))`` → uint8 frames.

    Convenience for the evaluation path. Equivalent to
    ``decode_latents(vae, encode_frames(vae, frames))`` but fuses the two batch
    loops so intermediate latents for the whole set are never all held at once.
    """
    frames = np.asarray(frames)
    n = frames.shape[0]
    out: list[np.ndarray] = []
    for start, count in _iter_padded_batches(n, batch_size):
        chunk = frames[start : start + count]
        if count < batch_size:
            pad = np.repeat(chunk[-1:], batch_size - count, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunk_f32 = chunk.astype(np.float32) / 127.5 - 1.0
        mu = _encode_batch(vae, jnp.asarray(chunk_f32))
        recon = _decode_batch(vae, mu)
        recon_u8 = np.clip((np.asarray(recon) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        out.append(recon_u8[:count])
    return np.concatenate(out, axis=0)


def encode_latent_stats(
    vae: model_lib.VAE,
    frames: np.ndarray,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mu, log_sigma_sq)`` for frames — used by KL-per-channel diagnostics.

    Unlike :func:`encode_frames` (which returns only ``mu``), this exposes both
    Gaussian parameters so the caller can compute the per-channel KL budget.
    """
    frames = np.asarray(frames)
    n = frames.shape[0]
    mus: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    for start, count in _iter_padded_batches(n, batch_size):
        chunk = frames[start : start + count]
        if count < batch_size:
            pad = np.repeat(chunk[-1:], batch_size - count, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunk_f32 = chunk.astype(np.float32) / 127.5 - 1.0
        mu, logvar = vae.encode(jnp.asarray(chunk_f32))
        mus.append(np.asarray(mu)[:count])
        logvars.append(np.asarray(logvar)[:count])
    return np.concatenate(mus, axis=0), np.concatenate(logvars, axis=0)
