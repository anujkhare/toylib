import dataclasses
import einops
import jax
import jaxtyping as jt
import jax.numpy as jnp
import math
from toylib.nn import attention
from toylib.nn import layers
from toylib.nn import module


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Configuration for the DecoderOnlyTransformer model."""

    num_layers: int = 2
    num_heads: int = 8
    qkv_dim: int = 256

    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    seq_len: int = 512

    # logit softcap (Gemma 2: https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)
    logit_softcap: float = 15.0

    # Storage dtype for all trainable parameters (default float32).
    param_dtype: object = jnp.float32
    # Compute dtype for all forward-pass operations (default float32).
    dtype: object = jnp.float32

    # Rematerialization (gradient checkpointing) policy applied to each transformer block.
    # None disables remat entirely. Typical values from jax.checkpoint_policies:
    #   nothing_saveable                     — recompute all intermediates (max memory savings)
    #   dots_with_no_batch_dims_saveable     — save GEMM outputs, recompute activations
    #   everything_saveable                  — save all intermediates (no memory savings)
    remat_policy: object = None

    # nanochat's later additions (its gpt.py, July 2026). All are off by default, so
    # models trained before them still load; `NANOCHAT` turns them all on.
    # Sliding windows, tiled across layers: L sees the whole context, S a quarter of
    # it. The last layer always sees the whole context.
    window_pattern: str = "L"
    # A second embedding table per alternate layer, added to the values (ResFormer).
    value_embeddings: bool = False
    # Per-layer scalars: scale the residual stream, and mix the embedding back in.
    residual_scalars: bool = False
    # Mix each token's embedding with the previous one's: cheap bigram information.
    smear: bool = False
    # Subtract the halfway residual stream before the output layer.
    backout: bool = False
    # Multiplies normalized queries and keys, which sharpens attention.
    qk_scale: float = 1.0
    # Multiplies the MLP's first layer's initial weights.
    mlp_init_scale: float = 1.0
    embedding_init_std: float = 1.0
    rope_base: int = 10_000
    # None computes attention as plain masked matmuls, which can also return the
    # weights. "xla" or "cudnn" use `jax.nn.dot_product_attention`, whose cuDNN
    # kernel is a fused flash attention on NVIDIA GPUs.
    attention_impl: str | None = None

    def __post_init__(self) -> None:
        if self.qkv_dim % self.num_heads != 0:
            raise ValueError(
                f"qkv_dim ({self.qkv_dim}) must divide into {self.num_heads} heads"
            )
        if (self.qkv_dim // self.num_heads) % 2 != 0:
            raise ValueError("RoPE rotates pairs of dimensions: use an even head dim")
        if not self.window_pattern or set(self.window_pattern) - set("SL"):
            raise ValueError(f"window_pattern must be S and L only: {self.window_pattern!r}")

    def window_sizes(self) -> list[int]:
        """How far back each layer's queries can see, as in nanochat's window sizes."""
        long, short = self.seq_len, -(-self.seq_len // 4)
        pattern = self.window_pattern
        sizes = [long if pattern[i % len(pattern)] == "L" else short for i in range(self.num_layers)]
        sizes[-1] = long
        return sizes

    def has_value_embedding(self, layer: int) -> bool:
        """Alternate layers, always including the last, as in nanochat's `has_ve`."""
        return self.value_embeddings and layer % 2 == (self.num_layers - 1) % 2


NANOCHAT = dict(
    window_pattern="SSSL",
    value_embeddings=True,
    residual_scalars=True,
    smear=True,
    backout=True,
    qk_scale=1.2,
    mlp_init_scale=0.4,
    embedding_init_std=0.8,
    rope_base=100_000,
)

VALUE_GATE_CHANNELS = 12  # Inputs to each value embedding's gate, as in nanochat.
SMEAR_GATE_CHANNELS = 24


class MLP(module.Module):
    """A simple feedforward MLP with one hidden layer."""

    qkv_dim: int
    key: jt.PRNGKeyArray
    init_scale: float = 1.0

    def init(self) -> None:
        qkv_dim = self.qkv_dim
        keys = jax.random.split(self.key, 2)
        # standard transformer architecture uses a feedforward layer
        # with an inner dimension of 4 times the model dimension.
        # See "Attention is All You Need" paper for more details.
        # https://arxiv.org/abs/1706.03762
        self.fc1 = layers.Linear(
            in_features=qkv_dim,
            out_features=4 * qkv_dim,
            key=keys[0],
            init_std=self.init_scale / math.sqrt(qkv_dim),
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.fc2 = layers.Linear(
            in_features=4 * qkv_dim,
            out_features=qkv_dim,
            key=keys[1],
            init_std=0.0,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(
        self, x: jt.Float[jt.Array, "... qkv_dim"]
    ) -> jt.Float[jt.Array, "... qkv_dim"]:
        x = self.fc1(x)
        # ReLU squared (https://arxiv.org/abs/2002.05202) or SwiGLU (https://arxiv.org/pdf/2204.02311)
        x = jax.nn.relu(x) ** 2
        x = self.fc2(x)
        return x


def window_mask(seq_len: int, window: int) -> jt.Float[jt.Array, "seq_len seq_len"]:
    """Query i sees keys i - window through i, as flash attention's window does."""
    distance = jnp.arange(seq_len)[:, None] - jnp.arange(seq_len)[None, :]
    return ((distance >= 0) & (distance <= window)).astype(jnp.float32)


class CausalSelfAttention(module.Module):
    """Causal Self-Attention layer with Rotary Positional Embeddings (RoPE)."""

    qkv_dim: int
    num_heads: int
    seq_len: int
    key: jt.PRNGKeyArray
    # How far back a query can see; seq_len or more is plain causal attention.
    window: int | None = None
    # Whether a value embedding is added to this layer's values.
    value_embedding: bool = False
    qk_scale: float = 1.0
    rope_base: int = 10_000
    attention_impl: str | None = None

    def init(self) -> None:
        self.mha = attention.MultiHeadAttention(
            qkv_dim=self.qkv_dim,
            num_heads=self.num_heads,
            key=self.key,
            use_qk_norm=True,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

        self.rope = attention.RotaryPositionalEmbedding(
            qkv_dim=self.qkv_dim // self.num_heads,
            seq_len=self.seq_len,
            base=self.rope_base,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

        if self.value_embedding:
            # One gate a head, from the first few channels, starting just above
            # neutral: 3 * sigmoid(small) is about 1.5.
            channels = min(VALUE_GATE_CHANNELS, self.qkv_dim)
            self.value_gate = jax.random.uniform(
                jax.random.fold_in(self.key, 1), (channels, self.num_heads), minval=0.0, maxval=0.02
            ).astype(self.param_dtype)

    def __call__(
        self,
        x: jt.Float[jt.Array, "... seq_len qkv_dim"],
        value_embedding: jt.Float[jt.Array, "... seq_len qkv_dim"] | None = None,
        return_attention_weights: bool = False,
    ) -> (
        jt.Float[jt.Array, "... seq_len qkv_dim"]
        | tuple[jt.Array, jt.Float[jt.Array, "... num_heads seq_len seq_len"]]
    ):
        mha, seq_len = self.mha, x.shape[-2]
        split = "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim"
        q, k, v = (
            einops.rearrange(proj(x), split, num_heads=self.num_heads)
            for proj in (mha.q_projection, mha.k_projection, mha.v_projection)
        )
        if value_embedding is not None:
            # Value residual: each head mixes in the value embedding, through a gate.
            channels = self.value_gate.shape[0]
            gate = 3 * jax.nn.sigmoid(x[..., :channels] @ self.value_gate.astype(x.dtype))
            ve = einops.rearrange(value_embedding, split, num_heads=self.num_heads)
            v = v + einops.rearrange(gate, "... t h -> ... h t 1") * ve

        q, k = self.rope(q), self.rope(k)
        q, k = layers.rms_norm(q) * self.qk_scale, layers.rms_norm(k) * self.qk_scale

        window = min(self.window or seq_len, seq_len)
        if return_attention_weights or self.attention_impl is None:
            values, weights = attention.scaled_dot_product_attention(
                q=q, k=k, v=v, mask=window_mask(seq_len, window)
            )
        else:
            values, weights = self._fused_attention(q, k, v, window), None
        values = einops.rearrange(values, "... num_heads seq_len d -> ... seq_len (num_heads d)")
        values = mha.linear(values)
        if return_attention_weights:
            return values, weights
        return values

    def _fused_attention(self, q, k, v, window: int) -> jax.Array:
        """`jax.nn.dot_product_attention`, which wants [batch, seq, heads, dim]."""
        unbatched = q.ndim == 3
        q, k, v = (jnp.swapaxes(t[None] if unbatched else t, -2, -3) for t in (q, k, v))
        values = jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            local_window_size=(window, 0) if window < q.shape[-3] else None,
            implementation=self.attention_impl,
        )
        values = jnp.swapaxes(values, -2, -3)
        return values[0] if unbatched else values


class DecoderBlock(module.Module):
    qkv_dim: int
    num_heads: int
    seq_len: int
    key: jt.PRNGKeyArray
    window: int | None = None
    value_embedding: bool = False
    qk_scale: float = 1.0
    mlp_init_scale: float = 1.0
    rope_base: int = 10_000
    attention_impl: str | None = None

    def init(self) -> None:
        keys = jax.random.split(self.key, 2)
        self.causal_attn = CausalSelfAttention(
            qkv_dim=self.qkv_dim,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            key=keys[0],
            window=self.window,
            value_embedding=self.value_embedding,
            qk_scale=self.qk_scale,
            rope_base=self.rope_base,
            attention_impl=self.attention_impl,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.mlp = MLP(
            qkv_dim=self.qkv_dim,
            key=keys[1],
            init_scale=self.mlp_init_scale,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(
        self,
        x: jt.Float[jt.Array, "... seq_len qkv_dim"],
        value_embedding: jt.Float[jt.Array, "... seq_len qkv_dim"] | None = None,
        return_intermediates: bool = False,
    ) -> jt.Float[jt.Array, "... seq_len qkv_dim"] | tuple[jt.Array, dict]:
        # "Serial" implementation: y = x + MLP(LN(x + CausalSelfAttention(LN(x))))
        # A "parallel" implementation is also possible (https://arxiv.org/pdf/2204.02311):
        #  y = x + MLP(LN(x)) + CausalSelfAttention(LN(x))
        attended = self.causal_attn(
            layers.rms_norm(x), value_embedding, return_attention_weights=return_intermediates
        )
        if return_intermediates:
            attended, attn_weights = attended
        x = x + attended
        x = x + self.mlp(layers.rms_norm(x))
        if return_intermediates:
            return x, {"attn_weights": attn_weights, "hidden": x}
        return x


class DecoderOnlyTransformer(module.Module):
    """A simple decoder-only transformer model.

    Takes in a sequence of tokens, embeds them using a learned embedding layer,
    applies causal self-attention with ROPE embeddings, and outputs the logits
    for the next token prediction.
    """

    key: jt.PRNGKeyArray
    config: ModelConfig

    def init(self) -> None:
        config = self.config

        # Generate keys - embedding, attention layers, output projection
        keys = jax.random.split(self.key, config.num_layers + 2)

        # Embedding layer
        self.embedding_layer = layers.Embedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.qkv_dim,
            key=keys[0],
            param_dtype=config.param_dtype,
            dtype=config.dtype,
        )
        if config.embedding_init_std != 1.0:
            self.embedding_layer.init()
            self.embedding_layer.weights = (
                self.embedding_layer.weights * config.embedding_init_std
            ).astype(config.param_dtype)

        windows = config.window_sizes()
        self.blocks = []
        for ix in range(config.num_layers):
            self.blocks.append(
                DecoderBlock(
                    qkv_dim=config.qkv_dim,
                    num_heads=config.num_heads,
                    seq_len=config.seq_len,
                    key=keys[ix + 1],
                    window=windows[ix],
                    value_embedding=config.has_value_embedding(ix),
                    qk_scale=config.qk_scale,
                    mlp_init_scale=config.mlp_init_scale,
                    rope_base=config.rope_base,
                    attention_impl=config.attention_impl,
                    param_dtype=config.param_dtype,
                    dtype=config.dtype,
                )
            )

        # nanochat's additions, created only when enabled so older checkpoints load.
        extra = jax.random.fold_in(self.key, config.num_layers + 2)
        n = config.num_layers
        if config.value_embeddings:
            # Initialized like the value projections: uniform, std 1/sqrt(dim).
            bound = math.sqrt(3 / config.qkv_dim)
            self.value_embeds = []
            for ix in range(n):
                if config.has_value_embedding(ix):
                    table = layers.Embedding(
                        vocab_size=config.vocab_size,
                        embedding_dim=config.qkv_dim,
                        key=jax.random.fold_in(extra, ix),
                        param_dtype=config.param_dtype,
                        dtype=config.dtype,
                    )
                    table.init()
                    table.weights = jax.random.uniform(
                        jax.random.fold_in(extra, ix),
                        table.weights.shape,
                        minval=-bound,
                        maxval=bound,
                    ).astype(config.param_dtype)
                    self.value_embeds.append(table)
        if config.residual_scalars:
            # Stronger residual early and weaker late; more of the embedding early.
            depth = jnp.arange(n) / max(n - 1, 1)
            self.resid_lambdas = (1.15 - 0.10 * depth).astype(jnp.float32)
            self.x0_lambdas = (0.20 - 0.15 * depth).astype(jnp.float32)
        if config.smear:
            channels = min(SMEAR_GATE_CHANNELS, config.qkv_dim)
            self.smear_gate = jax.random.uniform(
                jax.random.fold_in(extra, n), (channels, 1), minval=0.0, maxval=0.02
            ).astype(jnp.float32)
            self.smear_lambda = jnp.zeros((1,), jnp.float32)
        if config.backout:
            self.backout_lambda = jnp.full((1,), 0.2, jnp.float32)

        # Output projection — always stored and computed in float32 so logits
        # are full precision before the softcap and loss computation.
        self.output_layer = layers.Linear(
            in_features=config.qkv_dim,
            out_features=config.vocab_size,
            key=keys[-1],
            init_std=0.001,  # small std to prevent large initial logit values which can destabilize training
            param_dtype=config.param_dtype,
            dtype=jnp.float32,
        )

    def __call__(
        self,
        x: jt.Integer[jt.Array, "... seq_len"],
        return_intermediates: bool = False,
    ) -> jt.Float[jt.Array, "... seq_len vocab_size"] | tuple[jt.Array, dict]:
        """Forward pass for the decoder-only transformer model.

        Args:
            x: Input token ids of shape [..., seq_len]. The length may not exceed
                the configuration's `seq_len`.
            return_intermediates: Also return a dict with `embedding` and, for each
                block k, `blocks.k.hidden` and `blocks.k.attn_weights`
                ([..., num_heads, seq_len, seq_len]).

        Returns:
            Unnormalized logits over the vocabulary of shape [..., seq_len, vocab_size]
        """
        if return_intermediates:
            hidden, acts = self.hidden(x, return_intermediates=True)
            return self.logits(hidden), acts
        return self.logits(self.hidden(x))

    def hidden(
        self,
        x: jt.Integer[jt.Array, "... seq_len"],
        return_intermediates: bool = False,
    ) -> jt.Float[jt.Array, "... seq_len qkv_dim"] | tuple[jt.Array, dict]:
        """Run everything up to the output layer."""
        config = self.config
        tokens = x
        # Input projection to project the embeddings to the model dimension
        x = self.embedding_layer(tokens)
        x = layers.rms_norm(x)

        if config.smear:
            # Mix the previous token's embedding into each position, through a gate.
            channels = self.smear_gate.shape[0]
            gate = self.smear_lambda.astype(x.dtype) * jax.nn.sigmoid(
                x[..., 1:, :channels] @ self.smear_gate.astype(x.dtype)
            )
            x = jnp.concatenate([x[..., :1, :], x[..., 1:, :] + gate * x[..., :-1, :]], axis=-2)

        embedding = x
        value_tables = iter(getattr(self, "value_embeds", []))
        backout_layer = config.num_layers // 2
        acts = {"embedding": embedding}

        # Blocks differ (windows, value embeddings), so they run in a Python loop
        # rather than a scan over stacked weights.
        for ix, block in enumerate(self.blocks):
            if config.residual_scalars:
                x = (
                    self.resid_lambdas[ix].astype(x.dtype) * x
                    + self.x0_lambdas[ix].astype(x.dtype) * embedding
                )
            ve = next(value_tables)(tokens) if config.has_value_embedding(ix) else None
            if return_intermediates:
                x, block_acts = block(x, ve, return_intermediates=True)
                for name, values in block_acts.items():
                    acts[f"blocks.{ix}.{name}"] = values
            elif config.remat_policy is not None:
                # remat over a lambda function because Module can't be hashed
                x = jax.remat(lambda b, x, ve: b(x, ve), policy=config.remat_policy)(
                    block, x, ve
                )
            else:
                x = block(x, ve)
            if config.backout and ix == backout_layer:
                halfway = x

        if config.backout:
            x = x - self.backout_lambda.astype(x.dtype) * halfway
        if not return_intermediates:
            return x
        return x, acts

    def logits(
        self, x: jt.Float[jt.Array, "... qkv_dim"]
    ) -> jt.Float[jt.Array, "... vocab_size"]:
        """Project hidden states to the vocabulary.

        Sampling calls this on one position only, which skips most of the cost of
        a forward pass when the vocabulary is large.
        """
        x = layers.rms_norm(x)

        # Output projection with softcap to prevent large logit values
        x = self.output_layer(x)
        x = self.config.logit_softcap * jnp.tanh(x / self.config.logit_softcap)
        return x


def loss_fn(
    logits: jt.Float[jt.Array, "batch_size seq_len vocab_size"],
    targets: jt.Int[jt.Array, "batch_size seq_len"],
    mask: jt.Int[jt.Array, "batch_size seq_len"],
) -> tuple[jt.Float[jt.Array, ""], jt.Float[jt.Array, "batch_size seq_len"]]:
    """Computes the cross-entropy loss between logits and targets.

    Args:
        logits: Logits of shape [batch_size, seq_len, vocab_size].
        targets: Target token ids of shape [batch_size, seq_len].
        mask: 1 for positions that count towards the loss.

    Returns:
        Scalar loss, and the unmasked loss of each position.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # [batch_size, seq_len, vocab_size]
    per_token_loss = -jnp.take_along_axis(
        log_probs, targets[..., None], axis=-1
    ).squeeze(-1)  # [batch_size, seq_len]

    # masked loss - [batch_size, seq_len]
    masked_loss = mask * per_token_loss

    # The total loss is averaged per valid token - this treats all tokens equally. A
    # fine-tuning batch can hold no trained tokens (only users' turns); its loss is 0.
    total_loss = jnp.sum(masked_loss) / jnp.maximum(jnp.sum(mask), 1)

    return total_loss, per_token_loss


def train_step(
    model: DecoderOnlyTransformer,
    batch: jt.PyTree,
    return_aux: bool = False,
) -> tuple[jt.Float[jt.Array, ""], dict]:
    """A single training step for the model.

    Args:
        model: The DecoderOnlyTransformer model.
        batch: PyTree containing 'inputs', 'targets', and 'mask', each of shape
            [batch_size, seq_len].
        return_aux: If True, also return auxiliary information like per-token loss.

    Returns:
        Loss value for the batch and a dictionary of auxiliary information, which
        is empty unless `return_aux` is True.
    """
    tokens, targets, mask = batch["inputs"], batch["targets"], batch["mask"]
    logits = model(tokens)  # doesn't use mask right now
    total_loss, per_token_loss = loss_fn(logits, targets, mask)
    if not return_aux:
        return total_loss, {}
    return total_loss, {"per_token_loss": per_token_loss}


def sample(
    model: DecoderOnlyTransformer,
    input_tokens: jax.Array,
    prompt_len: int,
    key: jt.PRNGKeyArray,
    *,
    max_output_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> jax.Array:
    """Generates samples from the model given input tokens.

    JIT-compatible: uses lax.scan over a fixed-size pre-padded token buffer.
    input_tokens must be pre-padded to the model's seq_len with zeros.
    prompt_len is the number of real (non-padding) tokens in input_tokens.
    Once the buffer is full, the oldest token is dropped for each new one.
    Use `jit_sample` so that repeated calls compile once.

    Args:
        model: The DecoderOnlyTransformer model.
        input_tokens: Token ids pre-padded to model seq_len, shape [seq_len].
        prompt_len: Number of real tokens in input_tokens.
        key: JAX PRNG key for sampling.
        max_output_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Higher values increase randomness,
            and 0 always picks the most likely token.
        top_k: If specified, only consider the top_k logits for sampling.

    Returns:
        Generated token ids of shape [max_output_tokens].
    """

    seq_len = input_tokens.shape[-1]

    def step(carry, _):
        tokens, pos, key = carry

        # A full buffer slides left. The stale last token can't affect the logits
        # we read, because attention is causal.
        full = pos == seq_len
        tokens = jnp.where(full, jnp.roll(tokens, -1), tokens)
        pos = pos - full

        hidden = model.hidden(tokens)  # [seq_len, qkv_dim]
        # Read logits at the last valid position
        logit = model.logits(hidden[pos - 1])

        key, subkey = jax.random.split(key)
        if top_k:
            top_k_logits, _ = jax.lax.top_k(logit, min(top_k, logit.shape[-1]))
            logit = jnp.where(logit < top_k_logits[-1], -jnp.inf, logit)
        # Temperature may be traced, so 0 is approached rather than branched on.
        next_token = jax.random.categorical(
            subkey, logits=logit / jnp.maximum(temperature, 1e-6)
        ).astype(tokens.dtype)
        tokens = jax.lax.dynamic_update_slice(tokens, next_token[None], (pos,))
        return (tokens, pos + 1, key), next_token

    _, generated = jax.lax.scan(
        step,
        (input_tokens, jnp.array(prompt_len), key),
        None,
        length=max_output_tokens,
    )
    return generated


jit_sample = jax.jit(sample, static_argnames=("max_output_tokens", "top_k"))
