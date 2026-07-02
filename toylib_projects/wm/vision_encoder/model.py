"""KL-regularized VAE for the Track A1 vision codec.

Implements the architecture specified in
``docs/walkthroughs/a1_vision_codec.md`` (which is the actionable
distillation of ``docs/designs/vision_codec.md``).

3 stride-2 downsample stages give an 8× spatial reduction, so for an
input of ``(B, H, W, 3)`` the latent grid is ``(B, H/8, W/8, latent_channels)``.

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

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp
import jaxtyping as jt

from toylib.nn import attention as attn_lib
from toylib.nn import layers, module


# ──────────────────────────────────────────────────────────────────────────
# Hyperparameter container
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the Track A1 VAE.

    Defaults match the walkthrough: 64×64x3 input, 8×8×4 latent, base_ch=64
    (so 4×base_ch = 256 channels at the bottleneck).
    """

    base_ch: int = 64
    latent_channels: int = 4
    input_channels: int = 3
    num_attn_heads: int = 1
    # GroupNorm groups; SD convention. 64 % 32 = 0 and 256 % 32 = 0 so this fits.
    num_norm_groups: int = 32
    # Bounds applied to log σ² before any exp() — prevents NaN early in training.
    log_sigma_sq_clip_min: float = -30.0
    log_sigma_sq_clip_max: float = 20.0


# ──────────────────────────────────────────────────────────────────────────
# Core building blocks
# ──────────────────────────────────────────────────────────────────────────


class ResBlock(module.Module):
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
        self.norm1 = layers.GroupNorm(
            num_features=self.in_channels,
            num_groups=self.num_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.conv1 = layers.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding="SAME",
            key=keys[0],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.norm2 = layers.GroupNorm(
            num_features=self.out_channels,
            num_groups=self.num_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.conv2 = layers.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding="SAME",
            key=keys[1],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        # 1×1 channel-match conv only when needed.
        if self.in_channels != self.out_channels:
            self.skip = layers.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                padding="SAME",
                key=keys[2],
                param_dtype=self.param_dtype,
                dtype=self.dtype,
            )
        else:
            self.skip = None

    def __call__(
        self, x: jt.Float[jt.Array, "B H W in_channels"]
    ) -> jt.Float[jt.Array, "B H W out_channels"]:
        h = jax.nn.silu(self.norm1(x))
        h = self.conv1(h)
        h = jax.nn.silu(self.norm2(h))
        h = self.conv2(h)
        skip = x if self.skip is None else self.skip(x)
        return h + skip


class AttentionBlock(module.Module):
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
        self.norm = layers.GroupNorm(
            num_features=self.channels,
            num_groups=self.num_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.attn = attn_lib.MultiHeadAttention(
            qkv_dim=self.channels,
            num_heads=self.num_heads,
            key=self.key,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(
        self, x: jt.Float[jt.Array, "B H W C"]
    ) -> jt.Float[jt.Array, "B H W C"]:
        B, H, W, C = x.shape
        h = self.norm(x)
        h_seq = h.reshape(B, H * W, C)
        h_seq = self.attn(h_seq, h_seq, h_seq)
        h = h_seq.reshape(B, H, W, C)
        return x + h


# ──────────────────────────────────────────────────────────────────────────
# Encoder / Decoder
# ──────────────────────────────────────────────────────────────────────────


class Encoder(module.Module):
    """Down-3× conv encoder producing per-spatial Gaussian parameters.

    Output is split along the channel axis into ``μ`` and ``log σ²``.
    ``log σ²`` is clipped before any downstream ``exp`` to avoid NaNs at init.
    """

    config: ModelConfig
    key: jt.PRNGKeyArray

    def init(self) -> None:
        cfg = self.config
        ch = cfg.base_ch
        keys = jax.random.split(self.key, 12)

        # ── stem ────────────────────────────────────────────────────────
        self.conv_in = layers.Conv2D(
            in_channels=cfg.input_channels,
            out_channels=ch,
            kernel_size=3,
            padding="SAME",
            key=keys[0],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

        # ── stage 1, ch ─────────────────────────────────────────────────
        self.res1 = ResBlock(
            in_channels=ch,
            out_channels=ch,
            key=keys[1],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        # stride-2, ch → 2ch.
        self.down1 = layers.Conv2D(
            in_channels=ch,
            out_channels=2 * ch,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=keys[2],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

        # ── stage 2, 2ch ────────────────────────────────────────────────
        self.res2 = ResBlock(
            in_channels=2 * ch,
            out_channels=2 * ch,
            key=keys[3],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        # stride-2, 2ch → 4ch.
        self.down2 = layers.Conv2D(
            in_channels=2 * ch,
            out_channels=4 * ch,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=keys[4],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

        # ── stage 3, 4ch ────────────────────────────────────────────────
        self.res3 = ResBlock(
            in_channels=4 * ch,
            out_channels=4 * ch,
            key=keys[5],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        # stride-2, channels stay at 4ch.
        self.down3 = layers.Conv2D(
            in_channels=4 * ch,
            out_channels=4 * ch,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=keys[6],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

        # ── bottleneck ──────────────────────────────────────────────────
        self.res4 = ResBlock(
            in_channels=4 * ch,
            out_channels=4 * ch,
            key=keys[7],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.attn = AttentionBlock(
            channels=4 * ch,
            num_heads=cfg.num_attn_heads,
            key=keys[8],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res5 = ResBlock(
            in_channels=4 * ch,
            out_channels=4 * ch,
            key=keys[9],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

        # ── projection to (μ, log σ²) ───────────────────────────────────
        self.norm_out = layers.GroupNorm(
            num_features=4 * ch,
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.conv_out = layers.Conv2D(
            in_channels=4 * ch,
            out_channels=2 * cfg.latent_channels,
            kernel_size=3,
            padding="SAME",
            key=keys[10],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(
        self, x: jt.Float[jt.Array, "B H W 3"]
    ) -> tuple[
        jt.Float[jt.Array, "B h w latent_channels"],
        jt.Float[jt.Array, "B h w latent_channels"],
    ]:
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
        log_sigma_sq = jnp.clip(
            log_sigma_sq,
            self.config.log_sigma_sq_clip_min,
            self.config.log_sigma_sq_clip_max,
        )
        return mu, log_sigma_sq


class Decoder(module.Module):
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

        # ── bottleneck ──────────────────────────────────────────────────
        self.conv_in = layers.Conv2D(
            in_channels=cfg.latent_channels,
            out_channels=4 * ch,
            kernel_size=3,
            padding="SAME",
            key=keys[0],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res1 = ResBlock(
            in_channels=4 * ch,
            out_channels=4 * ch,
            key=keys[1],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.attn = AttentionBlock(
            channels=4 * ch,
            num_heads=cfg.num_attn_heads,
            key=keys[2],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res2 = ResBlock(
            in_channels=4 * ch,
            out_channels=4 * ch,
            key=keys[3],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

        # ── upsample 1, 4ch → 2ch ───────────────────────────────────────
        self.up1_conv = layers.Conv2D(
            in_channels=4 * ch,
            out_channels=4 * ch,
            kernel_size=3,
            padding="SAME",
            key=keys[4],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res3 = ResBlock(
            in_channels=4 * ch,
            out_channels=2 * ch,
            key=keys[5],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

        # ── upsample 2, 2ch → ch ────────────────────────────────────────
        self.up2_conv = layers.Conv2D(
            in_channels=2 * ch,
            out_channels=2 * ch,
            kernel_size=3,
            padding="SAME",
            key=keys[6],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res4 = ResBlock(
            in_channels=2 * ch,
            out_channels=ch,
            key=keys[7],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

        # ── upsample 3, ch → ch ─────────────────────────────────────────
        self.up3_conv = layers.Conv2D(
            in_channels=ch,
            out_channels=ch,
            kernel_size=3,
            padding="SAME",
            key=keys[8],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res5 = ResBlock(
            in_channels=ch,
            out_channels=ch,
            key=keys[9],
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

        # ── projection to pixels ────────────────────────────────────────
        self.norm_out = layers.GroupNorm(
            num_features=ch,
            num_groups=cfg.num_norm_groups,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.conv_out = layers.Conv2D(
            in_channels=ch,
            out_channels=cfg.input_channels,
            kernel_size=3,
            padding="SAME",
            key=keys[10],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(
        self, z: jt.Float[jt.Array, "B h w latent_channels"]
    ) -> jt.Float[jt.Array, "B H W 3"]:
        h = self.conv_in(z)
        h = self.res1(h)
        h = self.attn(h)
        h = self.res2(h)

        h = layers.upsample_nearest(h, factor=2)
        h = self.up1_conv(h)
        h = self.res3(h)

        h = layers.upsample_nearest(h, factor=2)
        h = self.up2_conv(h)
        h = self.res4(h)

        h = layers.upsample_nearest(h, factor=2)
        h = self.up3_conv(h)
        h = self.res5(h)

        h = jax.nn.silu(self.norm_out(h))
        h = self.conv_out(h)
        return jnp.tanh(h)


# ──────────────────────────────────────────────────────────────────────────
# Reparameterization + loss functions (pure)
# ──────────────────────────────────────────────────────────────────────────


def reparameterize(
    mu: jt.Float[jt.Array, "B h w C"],
    log_sigma_sq: jt.Float[jt.Array, "B h w C"],
    rng_key: jt.PRNGKeyArray,
) -> jt.Float[jt.Array, "B h w C"]:
    """Reparameterization trick: ``z = μ + σ · ε`` with ``ε ~ N(0, I)``.

    Differentiable in both ``μ`` and ``log σ²``; the only non-differentiable
    bit (the normal sample) is held in ``ε`` and gradients flow around it.

    At **inference** time, prefer ``z = μ`` directly (no noise) — this
    function is only needed during VAE training.
    """
    sigma = jnp.exp(0.5 * log_sigma_sq)
    eps = jax.random.normal(rng_key, mu.shape, dtype=mu.dtype)
    return mu + sigma * eps


def kl_divergence(
    mu: jt.Float[jt.Array, "B h w C"],
    log_sigma_sq: jt.Float[jt.Array, "B h w C"],
) -> jt.Float[jt.Array, ""]:
    """Closed-form KL( N(μ, σ²) || N(0, I) ), summed over latent dims.

    ``L_KL = 0.5 · mean_B( sum_{h,w,C} (μ² + σ² − log σ² − 1) )``.

    Sum over (h, w, C) **then** mean over the batch — matches the Stable
    Diffusion / walkthrough convention. Swapping sum/mean here scales the
    loss magnitude by ``h*w*C`` (256 for the default config).
    """
    sigma_sq = jnp.exp(log_sigma_sq)
    per_sample = 0.5 * jnp.sum(mu**2 + sigma_sq - log_sigma_sq - 1.0, axis=(1, 2, 3))
    return jnp.mean(per_sample)


def recon_loss_l1(
    recon: jt.Float[jt.Array, "B H W C"],
    target: jt.Float[jt.Array, "B H W C"],
) -> jt.Float[jt.Array, ""]:
    """Mean L1 over pixels. Both args are assumed in ``[-1, 1]``."""
    return jnp.mean(jnp.abs(recon - target))


def beta_warmup(
    step: int | jt.Array, warmup_steps: int, beta_max: float
) -> jt.Float[jt.Array, ""]:
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


# ──────────────────────────────────────────────────────────────────────────
# Top-level VAE module + training-loss wrapper
# ──────────────────────────────────────────────────────────────────────────


class VAE(module.Module):
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
        self.encoder = Encoder(
            config=self.config,
            key=keys[0],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.decoder = Decoder(
            config=self.config,
            key=keys[1],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def encode(
        self, x: jt.Float[jt.Array, "B H W 3"]
    ) -> tuple[jt.Float[jt.Array, "B h w C"], jt.Float[jt.Array, "B h w C"]]:
        return self.encoder(x)

    def decode(self, z: jt.Float[jt.Array, "B h w C"]) -> jt.Float[jt.Array, "B H W 3"]:
        return self.decoder(z)

    def __call__(
        self,
        x: jt.Float[jt.Array, "B H W 3"],
        rng_key: typing.Optional[jt.PRNGKeyArray] = None,
    ) -> tuple[jt.Float[jt.Array, "B H W 3"], dict[str, jt.Array]]:
        mu, log_sigma_sq = self.encode(x)
        if rng_key is None:
            # Deterministic / inference path.
            z = mu
        else:
            z = reparameterize(mu, log_sigma_sq, rng_key)
        recon = self.decode(z)
        return recon, {"mu": mu, "log_sigma_sq": log_sigma_sq, "z": z}


def vae_loss(
    model: VAE,
    batch: jt.Float[jt.Array, "B H W 3"],
    rng_key: jt.PRNGKeyArray,
    beta: jt.Float[jt.Array, ""] | float = 1e-6,
) -> tuple[jt.Float[jt.Array, ""], dict[str, jt.Array]]:
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
    l_kl = kl_divergence(model_aux["mu"], model_aux["log_sigma_sq"])
    total = l_rec + beta * l_kl
    aux = {
        "l_rec": l_rec,
        "l_kl": l_kl,
        "beta": jnp.asarray(beta, jnp.float32),
        "recon": recon,
        **model_aux,
    }
    return total, aux
