"""Linear / MLP probe on top of a frozen VAE encoder.

The probe answers a diagnostic question: *does the VAE's latent actually encode
the ball and paddle positions?* It runs the pretrained encoder, takes the
posterior mean ``mu`` (the deterministic latent), and feeds it through a small
MLP head that regresses the per-frame state targets (ball_x, ball_y, paddle_x).

The encoder is **not** trained here. Freezing is enforced at the optimizer
level (see ``probe/train.py``, which maps the ``encoder`` sub-tree to
``optax.set_to_zero``), so this module simply runs the encoder forward and lets
gradients flow; the optimizer drops the encoder updates. Keeping the freeze in
the optimizer — rather than a ``stop_gradient`` here — keeps the probe model a
plain feed-forward module and makes "what is trainable" a single, inspectable
decision in the training script.

Built on the toylib ``Module`` base (dataclass-style, pytree-registered) so it
plugs straight into the shared ``Experiment`` harness.
"""

from __future__ import annotations

import dataclasses
import enum

import jax
import jax.numpy as jnp
import jaxtyping as jt

from toylib.nn import layers, module


class Pooling(enum.Enum):
    """How the spatial latent grid is reduced to a feature vector.

    ``FLATTEN`` keeps the full spatial latent (preserves *where* the ball is);
    ``MEAN`` pools over the grid (only useful if position is encoded in channel
    statistics — usually it isn't, so flatten is the default).
    """

    FLATTEN = "flatten"
    MEAN = "mean"


class EncoderType(enum.Enum):
    """Which "encoder" feeds the probe head.

    ``VAE`` runs the pretrained (frozen) VAE encoder and probes its latent.
    ``PASSTHROUGH`` skips the VAE entirely and feeds the raw image to the head
    — a baseline / upper bound: the ball is trivially recoverable from pixels,
    so a pass-through probe tells you how much positional info the latent
    *could* carry. Compare its R² to the VAE probe's.
    """

    VAE = "vae"
    PASSTHROUGH = "passthrough"


class IdentityEncoder(module.Module):
    """Pass-through "encoder": returns the input image as the latent.

    Mirrors the VAE ``Encoder`` call signature ``(x) -> (mu, log_sigma_sq)`` so
    it drops straight into ``MLPProbe``. It has no parameters, so the
    optimizer-level encoder freeze is a harmless no-op.
    """

    def init(self) -> None:  # no parameters to build
        pass

    def __call__(
        self, x: jt.Float[jt.Array, "B H W C"]
    ) -> tuple[jt.Float[jt.Array, "B H W C"], jt.Float[jt.Array, "B H W C"]]:
        return x, x  # second value is ignored by the probe (it uses mu only)


@dataclasses.dataclass(frozen=True)
class ProbeConfig:
    """Hyperparameters for the latent probe.

    ``latent_spatial`` and ``latent_channels`` must match the encoder that
    produces the latent (16×16×4 for the default 128×128 VAE). They determine
    the flattened feature dimension fed to the MLP head.
    """

    latent_channels: int = 4
    latent_spatial: int = 16  # 128 / 8x encoder downsampling
    hidden_dim: int = 256
    num_targets: int = 3  # ball_x, ball_y, paddle_x
    pooling: Pooling = Pooling.FLATTEN

    @property
    def feature_dim(self) -> int:
        if self.pooling is Pooling.FLATTEN:
            return self.latent_spatial * self.latent_spatial * self.latent_channels
        if self.pooling is Pooling.MEAN:
            return self.latent_channels
        raise ValueError(f"unknown pooling mode: {self.pooling!r}")


class MLPProbe(module.Module):
    """Frozen VAE encoder + two-layer MLP regressing state targets.

    The ``encoder`` is supplied pre-initialized (typically restored from a VAE
    checkpoint) so this module's ``init`` only builds the trainable MLP head.
    """

    config: ProbeConfig
    # Either a VAE ``Encoder`` or an ``IdentityEncoder`` (pass-through baseline);
    # both expose ``(frames) -> (mu, _)``.
    encoder: module.Module
    key: jt.PRNGKeyArray

    def init(self) -> None:
        keys = jax.random.split(self.key, 2)
        self.fc1 = layers.Linear(
            in_features=self.config.feature_dim,
            out_features=self.config.hidden_dim,
            use_bias=True,
            key=keys[0],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.fc2 = layers.Linear(
            in_features=self.config.hidden_dim,
            out_features=self.config.num_targets,
            use_bias=True,
            key=keys[1],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def _pool(
        self, mu: jt.Float[jt.Array, "B h w C"]
    ) -> jt.Float[jt.Array, "B feature_dim"]:
        if self.config.pooling is Pooling.FLATTEN:
            return mu.reshape(mu.shape[0], -1)
        return jnp.mean(mu, axis=(1, 2))  # Pooling.MEAN

    def __call__(
        self, frames: jt.Float[jt.Array, "B 128 128 3"]
    ) -> jt.Float[jt.Array, "B num_targets"]:
        mu, _ = self.encoder(frames)  # deterministic latent (posterior mean)
        feat = self._pool(mu)
        h = jax.nn.silu(self.fc1(feat))
        return self.fc2(h)
