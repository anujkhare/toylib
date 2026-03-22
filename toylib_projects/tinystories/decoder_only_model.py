import dataclasses
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


class MLP(module.Module):
    """A simple feedforward MLP with one hidden layer."""

    qkv_dim: int
    key: jt.PRNGKeyArray

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
            init_std=1 / math.sqrt(qkv_dim),
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


class CausalSelfAttention(module.Module):
    """Causal Self-Attention layer with Rotary Positional Embeddings (RoPE)."""

    qkv_dim: int
    num_heads: int
    seq_len: int
    key: jt.PRNGKeyArray

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
            base=10_000,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def _make_causal_mask(self, seq_len: int) -> jt.Float[jt.Array, "seq_len seq_len"]:
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(
        self, x: jt.Float[jt.Array, "... seq_len qkv_dim"]
    ) -> jt.Float[jt.Array, "... seq_len qkv_dim"]:
        x = self.mha(
            Q=x, K=x, V=x, mask=self._make_causal_mask(x.shape[-2]), rope=self.rope
        )
        return x


class DecoderBlock(module.Module):
    qkv_dim: int
    num_heads: int
    seq_len: int
    key: jt.PRNGKeyArray

    def init(self) -> None:
        keys = jax.random.split(self.key, 2)
        self.causal_attn = CausalSelfAttention(
            qkv_dim=self.qkv_dim,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            key=keys[0],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.mlp = MLP(
            qkv_dim=self.qkv_dim,
            key=keys[1],
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(
        self, x: jt.Float[jt.Array, "... seq_len qkv_dim"]
    ) -> jt.Float[jt.Array, "... seq_len qkv_dim"]:
        # "Serial" implementation: y = x + MLP(LN(x + CausalSelfAttention(LN(x))))
        # A "parallel" implementation is also possible (https://arxiv.org/pdf/2204.02311):
        #  y = x + MLP(LN(x)) + CausalSelfAttention(LN(x))
        x = x + self.causal_attn(layers.rms_norm(x))
        x = x + self.mlp(layers.rms_norm(x))
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

        self.blocks = []
        for ix in range(config.num_layers):
            self.blocks.append(
                DecoderBlock(
                    qkv_dim=config.qkv_dim,
                    num_heads=config.num_heads,
                    seq_len=config.seq_len,
                    key=keys[ix + 1],
                    param_dtype=config.param_dtype,
                    dtype=config.dtype,
                )
            )

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
        self, x: jt.Integer[jt.Array, "... seq_len"]
    ) -> jt.Float[jt.Array, "... seq_len vocab_size"]:
        """Forward pass for the decoder-only transformer model.

        Args:
            x: Input token ids of shape [seq_len]. Note that the sequence
                length should match the configuration `seq_len` as this is used for
                positional embeddings.

        Returns:
            Unnormalized logits over the vocabulary of shape [batch_size, seq_len, vocab_size]
        """
        # Input projection to project the embeddings to the model dimension
        x = self.embedding_layer(x)
        x = layers.rms_norm(x)

        # Apply each attention layer
        remat_policy = self.config.remat_policy

        # Self-attention layers
        def scan_body(block_inputs, block):
            if remat_policy is not None:
                # remat over a lambda function because Module can't be hashed
                block_outputs = jax.remat(lambda b, x: b(x), policy=remat_policy)(
                    block, block_inputs
                )
            else:
                block_outputs = block(block_inputs)

            return block_outputs, None

        stacked_blocks = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *self.blocks)
        x, _ = jax.lax.scan(scan_body, x, stacked_blocks)
        x = layers.rms_norm(x)

        # Output projection with softcap to prevent large logit values
        x = self.output_layer(x)
        x = self.config.logit_softcap * jnp.tanh(x / self.config.logit_softcap)
        return x


def loss_fn(
    logits: jt.Float[jt.Array, "batch_size seq_len vocab_size"],
    targets: jt.Int[jt.Array, "batch_size seq_len"],
    mask: jt.Int[jt.Array, "batch_size seq_len"],
) -> jt.Float[jt.Array, ""]:
    """Computes the cross-entropy loss between logits and targets.

    Args:
        logits: Logits of shape [batch_size, seq_len, vocab_size].
        targets: Target token ids of shape [batch_size, seq_len].

    Returns:
        Scalar loss value.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # [batch_size, seq_len, vocab_size]
    per_token_loss = -jnp.take_along_axis(
        log_probs, targets[..., None], axis=-1
    ).squeeze(-1)  # [batch_size, seq_len]

    # masked loss - [batch_size, seq_len]
    masked_loss = mask * per_token_loss

    # The total loss is averaged per valid token - this treats all tokens equally
    total_loss = jnp.sum(masked_loss) / jnp.sum(mask)

    return total_loss, per_token_loss


def train_step(
    model: DecoderOnlyTransformer,
    batch: jt.PyTree,
    return_aux: bool = False,
) -> jt.Float[jt.Array, ""]:
    """A single training step for the model.

    Args:
        model: The DecoderOnlyTransformer model.
        batch: PyTree containing 'inputs', 'targets', and 'mask', each of shape
            [batch_size, seq_len].
        return_aux: If True, also return auxiliary information like per-token loss.

    Returns:
        Loss value for the batch. If `return_aux` is True, also returns a dictionary
        with auxiliary information.
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

    Args:
        model: The DecoderOnlyTransformer model.
        input_tokens: Token ids pre-padded to model seq_len, shape [seq_len].
        prompt_len: Number of real tokens in input_tokens.
        key: JAX PRNG key for sampling.
        max_output_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Higher values increase randomness.
        top_k: If specified, only consider the top_k logits for sampling. Set
            to 1 for greedy sampling.

    Returns:
        Generated token ids of shape [max_output_tokens].
    """

    def step(carry, _):
        tokens, pos, key = carry

        logits = model(tokens)  # [seq_len, vocab_size]
        # Read logits at the last valid position
        logit = jax.lax.dynamic_index_in_dim(logits, pos - 1, axis=0, keepdims=False)
        logit = logit / temperature

        if top_k:
            top_k_logits, _ = jax.lax.top_k(logit, top_k)
            logit = jnp.where(logit < top_k_logits[-1], -jnp.inf, logit)

        key, subkey = jax.random.split(key)
        next_token = jax.random.categorical(subkey, logits=logit).astype(tokens.dtype)
        tokens = jax.lax.dynamic_update_slice(tokens, next_token[None], (pos,))
        return (tokens, pos + 1, key), next_token

    _, generated = jax.lax.scan(
        step,
        (input_tokens, jnp.array(prompt_len), key),
        None,
        length=max_output_tokens,
    )
    return generated
