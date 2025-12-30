import dataclasses
import jax
import jaxtyping as jt
import jax.numpy as jnp
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
            in_features=qkv_dim, out_features=4 * qkv_dim, key=keys[0]
        )
        self.fc2 = layers.Linear(
            in_features=4 * qkv_dim, out_features=qkv_dim, key=keys[1]
        )
        # Initialize weights to zero to stabilize training at the start
        self.fc2.weights = jnp.zeros_like(self.fc2.weights)

    def __call__(
        self, x: jt.Float[jt.Array, "... qkv_dim"]
    ) -> jt.Float[jt.Array, "... qkv_dim"]:
        x = self.fc1(x)
        # TODO: relu squared (https://arxiv.org/abs/2002.05202) or SwiGLU (https://arxiv.org/pdf/2204.02311)
        x = jax.nn.gelu(x)
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
        )
        # Initialize weights to zero to stabilize training at the start
        self.mha.linear.weights = jnp.zeros_like(self.mha.linear.weights)

        self.rope = attention.RotaryPositionalEmbedding(
            qkv_dim=self.qkv_dim // self.num_heads, seq_len=self.seq_len
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
        )
        self.mlp = MLP(qkv_dim=self.qkv_dim, key=keys[1])

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
            vocab_size=config.vocab_size, embedding_dim=config.qkv_dim, key=keys[0]
        )

        # Self-attention layers
        self.blocks = []
        for ix in range(config.num_layers):
            self.blocks.append(
                DecoderBlock(
                    qkv_dim=config.qkv_dim,
                    num_heads=config.num_heads,
                    seq_len=config.seq_len,
                    key=keys[ix + 1],
                )
            )

        # Output projection
        self.output_layer = layers.Linear(
            in_features=config.qkv_dim, out_features=config.vocab_size, key=keys[-1]
        )

    def __call__(
        self, x: jt.Float[jt.Array, "... seq_len"]
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
        for block in self.blocks:
            x = block(x)
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
    targets_one_hot = jax.nn.one_hot(
        targets, num_classes=logits.shape[-1]
    )  # [batch_size, seq_len, vocab_size]
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # [batch_size, seq_len, vocab_size]
    per_token_loss = -jnp.sum(
        targets_one_hot * log_probs, axis=-1
    )  # [batch_size, seq_len]

    # masked loss - [batch_size, seq_len]
    masked_loss = mask * per_token_loss

    # The total loss is averaged per valid token - this treats all tokens equally
    total_loss = jnp.sum(masked_loss) / jnp.sum(mask)

    return total_loss, per_token_loss


def train_step(
    model: DecoderOnlyTransformer,
    batch: jt.PyTree,
) -> jt.Float[jt.Array, ""]:
    """A single training step for the model.

    Args:
        model: The DecoderOnlyTransformer model.
        batch: PyTree containing 'inputs', 'targets', and 'mask', each of shape
            [batch_size, seq_len].

    Returns:
        Loss value for the batch.
    """
    tokens, targets, mask = batch["inputs"], batch["targets"], batch["mask"]
    logits = model(tokens)  # doesn't use mask right now
    total_loss, per_token_loss = loss_fn(logits, targets, mask)
    return total_loss, {"logits": logits, "per_token_loss": per_token_loss}


# TODO: jitted version??
def sample(
    model: DecoderOnlyTransformer,
    input_tokens: list[int],
    key: jt.PRNGKeyArray,
    *,
    max_output_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> jt.Int[jt.Array, "batch_size seq_len"]:
    """Generates samples from the model given input tokens.

    This is a simple implementation with no optimizations or batching support.

    Args:
        model: The DecoderOnlyTransformer model.
        input_tokens: Input token ids of shape [seq_len].
        key: JAX PRNG key for sampling.
        max_output_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Higher values increase randomness.
        top_k: If specified, only consider the top_k logits for sampling. Set
            to 1 for greedy sampling.

    Yields:
        Generated token ids one at a time.
    """
    if temperature < 0:
        raise ValueError("Temperature must be non-negative.")
    tokens = input_tokens.copy()

    # Generate up-to the max_output_tokens, no early stopping / stopping token
    for _ in range(max_output_tokens):
        # forward pass
        outputs = model(jnp.array(tokens))  # [seq_len, vocab_size]

        # Get the logits for the last token
        logits = outputs[-1, :]  # [vocab_size]
        logits /= temperature

        # Filter out to only top-k logits if specified
        # This is a method to reduce the sampling noise
        if top_k:
            top_k_logits, _ = jax.lax.top_k(logits, top_k)
            logits = jnp.where(logits < top_k_logits[-1], -jnp.inf, logits)

        # Sample the next token and update the inputs
        next_token = jax.random.categorical(key=key, logits=logits)
        tokens.append(next_token)

        yield next_token
