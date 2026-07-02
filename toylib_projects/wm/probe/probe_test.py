"""Tests for the latent state probe.

Covers the three probe-specific behaviors:

  - The probe forward produces ``(B, num_targets)`` predictions.
  - Training freezes the encoder (its weights are unchanged after steps) while
    the MLP head updates — i.e. the optimizer-level freeze actually works.

Everything runs on a tiny synthetic dataset and a small VAE so the test is fast.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers LZ4 filter)
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from toylib_projects.wm.probe import model as probe_model
from toylib_projects.wm.probe import train as probe_train
from toylib_projects.wm.vision_encoder import model as vae_model

# Small but valid: input must be divisible by 8 (3 stride-2 stages) and channels
# divisible by num_norm_groups (32). 16→latent 2×2; base_ch 32 → 128 at bottleneck.
_H = _W = 16
_LATENT_SPATIAL = 2
_BASE_CH = 32
_LATENT_CH = 4


def _vae_config() -> vae_model.ModelConfig:
    return vae_model.ModelConfig(base_ch=_BASE_CH, latent_channels=_LATENT_CH)


def _make_vae(seed: int = 0) -> vae_model.VAE:
    vae = vae_model.VAE(config=_vae_config(), key=jax.random.key(seed))
    vae.init()
    return vae


def _save_vae(ckpt_dir: Path, step: int, vae: vae_model.VAE) -> None:
    """Mirror ``Experiment.save_checkpoint``: a composite with a ``model`` item."""
    np_vae = jax.tree.map(np.asarray, vae)
    manager = ocp.CheckpointManager(
        str(ckpt_dir), options=ocp.CheckpointManagerOptions()
    )
    manager.save(step, args=ocp.args.Composite(model=ocp.args.StandardSave(np_vae)))
    manager.wait_until_finished()
    manager.close()


def _write_labelled_h5(path: Path, n: int = 64) -> Path:
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(n, _H, _W, 3), dtype=np.uint8)
    with h5py.File(path, "w") as f:
        f.attrs["n_frames"] = np.int32(n)
        f.attrs["height"] = np.int32(_H)
        f.attrs["width"] = np.int32(_W)
        f.create_dataset(
            "frames", data=frames, chunks=(1, _H, _W, 3), **hdf5plugin.LZ4()
        )
        src = f.create_group("source")
        for k in probe_train.TARGET_KEYS:
            src.create_dataset(
                k, data=rng.integers(0, 256, size=(n,)).astype(np.float32)
            )
    return path


# ──────────────────────────────────────────────────────────────────────────
# Probe model forward
# ──────────────────────────────────────────────────────────────────────────


def test_probe_forward_shape() -> None:
    encoder = _make_vae(seed=2).encoder
    cfg = probe_model.ProbeConfig(
        latent_channels=_LATENT_CH,
        latent_spatial=_LATENT_SPATIAL,
        hidden_dim=16,
        num_targets=len(probe_train.TARGET_KEYS),
    )
    probe = probe_model.MLPProbe(config=cfg, encoder=encoder, key=jax.random.key(0))
    probe.init()

    frames = jnp.zeros((5, _H, _W, 3), dtype=jnp.float32)
    pred = probe(frames)
    assert pred.shape == (5, len(probe_train.TARGET_KEYS))


def test_passthrough_encoder_feeds_raw_pixels() -> None:
    """IdentityEncoder returns the image as the latent, so the flattened feature
    dim equals H*W*3 and the probe still produces (B, num_targets)."""
    encoder = probe_model.IdentityEncoder()
    encoder.init()
    cfg = probe_model.ProbeConfig(
        latent_channels=3,
        latent_spatial=_H,
        hidden_dim=16,
        num_targets=len(probe_train.TARGET_KEYS),
    )
    assert cfg.feature_dim == _H * _W * 3
    probe = probe_model.MLPProbe(config=cfg, encoder=encoder, key=jax.random.key(0))
    probe.init()
    # The pass-through encoder has no trainable parameters.
    assert jax.tree_util.tree_leaves(probe.encoder) == []

    frames = jnp.zeros((5, _H, _W, 3), dtype=jnp.float32)
    assert probe(frames).shape == (5, len(probe_train.TARGET_KEYS))


def test_probe_feature_dim_matches_pooling() -> None:
    flat = probe_model.ProbeConfig(
        latent_channels=_LATENT_CH,
        latent_spatial=_LATENT_SPATIAL,
        pooling=probe_model.Pooling.FLATTEN,
    )
    assert flat.feature_dim == _LATENT_SPATIAL * _LATENT_SPATIAL * _LATENT_CH
    mean = probe_model.ProbeConfig(
        latent_channels=_LATENT_CH, pooling=probe_model.Pooling.MEAN
    )
    assert mean.feature_dim == _LATENT_CH


# ──────────────────────────────────────────────────────────────────────────
# Encoder freeze (end-to-end via the shared Experiment)
# ──────────────────────────────────────────────────────────────────────────


def test_training_freezes_encoder_and_updates_head(tmp_path: Path) -> None:
    vae = _make_vae(seed=3)
    ckpt_dir = tmp_path / "vae_ckpt"
    _save_vae(ckpt_dir, step=0, vae=vae)
    _write_labelled_h5(tmp_path / "train.h5", n=64)

    bspd = jax.local_device_count()  # smallest batch divisible by devices
    exp = probe_train.create_experiment(
        train_path=tmp_path / "train.h5",
        val_path=None,
        vae_checkpoint_dir=str(ckpt_dir),
        vae_checkpoint_step=0,
        base_ch=_BASE_CH,
        latent_channels=_LATENT_CH,
        latent_spatial=_LATENT_SPATIAL,
        hidden_dim=16,
        batch_size_per_device=bspd,
        max_steps=3,
        learning_rate=1e-2,
        save_interval_steps=10_000,  # don't checkpoint during the test
        eval_interval_steps=10_000,
        checkpoint_dir=str(tmp_path / "probe_ckpt"),
        log_dir=str(tmp_path / "logs"),
    )
    exp.init_state()

    enc_before = [np.asarray(x) for x in jax.tree_util.tree_leaves(exp.model.encoder)]
    fc1_before = [np.asarray(x) for x in jax.tree_util.tree_leaves(exp.model.fc1)]

    exp.outer_loop()
    exp.cleanup()

    enc_after = [np.asarray(x) for x in jax.tree_util.tree_leaves(exp.model.encoder)]
    fc1_after = [np.asarray(x) for x in jax.tree_util.tree_leaves(exp.model.fc1)]

    # Encoder must be byte-for-byte frozen.
    for a, b in zip(enc_before, enc_after):
        np.testing.assert_array_equal(a, b)

    # The MLP head must have moved (training actually happened).
    assert any(not np.allclose(a, b) for a, b in zip(fc1_before, fc1_after)), (
        "MLP head did not update — optimizer freeze may be mislabeling params"
    )


def test_training_without_checkpoint_uses_random_encoder(tmp_path: Path) -> None:
    """No VAE checkpoint → encoder is randomly initialized, but the probe still
    trains end-to-end (encoder frozen, MLP head updates). This is the smoke-test
    path that lets us exercise the pipeline without a pretrained VAE."""
    _write_labelled_h5(tmp_path / "train.h5", n=64)

    bspd = jax.local_device_count()
    exp = probe_train.create_experiment(
        train_path=tmp_path / "train.h5",
        val_path=None,
        vae_checkpoint_dir=None,  # ← no checkpoint
        vae_checkpoint_step=None,
        base_ch=_BASE_CH,
        latent_channels=_LATENT_CH,
        latent_spatial=_LATENT_SPATIAL,
        hidden_dim=16,
        batch_size_per_device=bspd,
        max_steps=3,
        learning_rate=1e-2,
        save_interval_steps=10_000,
        eval_interval_steps=10_000,
        checkpoint_dir=str(tmp_path / "probe_ckpt"),
        log_dir=str(tmp_path / "logs"),
    )
    exp.init_state()

    enc_before = [np.asarray(x) for x in jax.tree_util.tree_leaves(exp.model.encoder)]
    fc1_before = [np.asarray(x) for x in jax.tree_util.tree_leaves(exp.model.fc1)]
    assert len(enc_before) > 0  # a real (random) encoder was built

    exp.outer_loop()
    exp.cleanup()

    enc_after = [np.asarray(x) for x in jax.tree_util.tree_leaves(exp.model.encoder)]
    fc1_after = [np.asarray(x) for x in jax.tree_util.tree_leaves(exp.model.fc1)]

    for a, b in zip(enc_before, enc_after):
        np.testing.assert_array_equal(a, b)  # still frozen
    assert any(not np.allclose(a, b) for a, b in zip(fc1_before, fc1_after))


# ──────────────────────────────────────────────────────────────────────────
# R² metric + pass-through baseline
# ──────────────────────────────────────────────────────────────────────────


def test_forward_fn_emits_r2_matching_definition() -> None:
    """``r2_{name}`` must equal 1 − MSE/Var for the supplied target variance."""
    var = np.array([0.04, 0.09, 0.16], dtype=np.float32)  # normalized-scale var
    forward = probe_train.make_forward_fn(target_var=var)

    class _ConstModel:
        def __call__(self, frames):
            # Predict zero for every target regardless of input.
            return jnp.zeros((frames.shape[0], len(probe_train.TARGET_KEYS)))

    n = 8
    frames = jnp.zeros((n, _H, _W, 3), dtype=jnp.float32)
    # Targets in RAM units; forward divides by TARGET_SCALE internally.
    targets = jnp.asarray(np.tile(np.array([10.0, 20.0, 30.0], np.float32), (n, 1)))
    _, aux = forward(_ConstModel(), {"frames": frames, "targets": targets})

    for i, name in enumerate(probe_train.TARGET_KEYS):
        mse = float(aux[f"mse_{name}"])
        np.testing.assert_allclose(
            float(aux[f"r2_{name}"]), 1.0 - mse / var[i], rtol=1e-5
        )


def test_target_variance_matches_numpy(tmp_path: Path) -> None:
    _write_labelled_h5(tmp_path / "train.h5", n=64)
    var = probe_train.target_variance(tmp_path / "train.h5", probe_train.TARGET_KEYS)
    assert var.shape == (len(probe_train.TARGET_KEYS),)
    assert np.all(var > 0)  # synthetic targets are non-constant


def test_passthrough_experiment_trains_and_logs_r2(tmp_path: Path) -> None:
    """End-to-end pass-through baseline: no checkpoint needed, the probe trains
    on raw pixels, and the eval/train aux exposes R² per target."""
    _write_labelled_h5(tmp_path / "train.h5", n=64)

    bspd = jax.local_device_count()
    exp = probe_train.create_experiment(
        train_path=tmp_path / "train.h5",
        val_path=None,
        encoder_type=probe_model.EncoderType.PASSTHROUGH,
        hidden_dim=16,
        batch_size_per_device=bspd,
        max_steps=3,
        learning_rate=1e-2,
        save_interval_steps=10_000,
        eval_interval_steps=10_000,
        checkpoint_dir=str(tmp_path / "probe_ckpt"),
        log_dir=str(tmp_path / "logs"),
    )
    exp.init_state()

    # The pass-through "encoder" carries no parameters, and its feature dim is
    # the flattened raw frame (H*W*3).
    assert jax.tree_util.tree_leaves(exp.model.encoder) == []
    assert exp.model.config.feature_dim == _H * _W * 3

    fc1_before = [np.asarray(x) for x in jax.tree_util.tree_leaves(exp.model.fc1)]
    exp.outer_loop()
    exp.cleanup()
    fc1_after = [np.asarray(x) for x in jax.tree_util.tree_leaves(exp.model.fc1)]

    assert any(not np.allclose(a, b) for a, b in zip(fc1_before, fc1_after))

    # The R² keys are surfaced by the metric.
    metric = probe_train.ProbeAuxMetric()
    assert all(f"r2_{k}" in metric.keys for k in probe_train.TARGET_KEYS)
