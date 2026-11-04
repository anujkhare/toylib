"""Language-model metrics. Generic ones (``Metric``, ``Loss``) live in ``toylib.harness.metrics``."""

import dataclasses

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np


# bfloat16 peaks, in the order to match against a device's name, as in nanochat.
PEAK_FLOPS = (("H100 NVL", 835e12), ("H100 PCIe", 756e12), ("H100", 989e12), ("A100", 312e12))


def flops_per_token(config) -> float:
    """Count matmul FLOPs for forward and backward, as nanochat's `estimate_flops` does.

    The embeddings are lookups, so they cost nothing here. A layer with a sliding
    window attends to at most that many tokens.
    """
    dim, depth, heads = config.qkv_dim, config.num_layers, config.num_heads
    matmul_params = 12 * dim**2 * depth + config.vocab_size * dim
    matmul_params += sum(
        min(12, dim) * heads for i in range(depth) if config.has_value_embedding(i)
    )
    matmul_params += min(24, dim) if config.smear else 0
    seen = sum(min(window, config.seq_len) for window in config.window_sizes())
    return 6 * matmul_params + 12 * heads * (dim // heads) * seen


def peak_flops() -> float | None:
    """Return the first device's bfloat16 peak, or None for hardware without one."""
    kind = jax.devices()[0].device_kind
    return next((peak for name, peak in PEAK_FLOPS if name in kind), None)


@dataclasses.dataclass
class BitsPerByte:
    """Metric that computes bits per byte from per-token loss.

    Total bits over total bytes, as in nanochat's `evaluate_bpb`:

        bits per byte = sum(loss in bits) / sum(bytes of the predicted tokens)

    Bytes are those of the targets, since the loss at a position is the cost of
    predicting its target. Tokens marked -1 (special tokens) count in neither sum.

    The bytes per token mapping is loaded from an .npy file at initialization.
    """

    bytes_per_token_path: str
    _bytes_per_token: jt.Array = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        """Load the bytes per token array from disk."""
        self._bytes_per_token = jnp.array(np.load(self.bytes_per_token_path))

    def __call__(
        self,
        loss: jt.ArrayLike,
        aux: jt.PyTree,
        batch: jt.PyTree,
    ) -> dict[str, jt.ArrayLike]:
        """Compute bits per byte metric.

        Args:
            loss: The loss value (unused)
            aux: Must contain 'per_token_loss' with shape [batch_size, seq_len]
            batch: Must contain 'targets' and 'mask' of shape [batch_size, seq_len]

        Returns:
            Dictionary with 'bits_per_byte' metric
        """
        del loss

        # Get per-token loss (in nats) and token ids
        per_token_loss = aux["per_token_loss"]  # [batch_size, seq_len]
        token_ids = batch["targets"]  # [batch_size, seq_len]
        mask = batch["mask"]  # [batch_size, seq_len]

        # Look up bytes per token for each token in the batch
        bytes_per_token = self._bytes_per_token[token_ids]  # [batch_size, seq_len]
        token_valid = jnp.where(bytes_per_token == -1, 0, 1) * mask

        # Convert loss from nats to bits (multiply by log2(e))
        total_bits = (per_token_loss * jnp.log2(jnp.e) * token_valid).sum()
        total_bytes = (bytes_per_token * token_valid).sum()

        return {"bits_per_byte": total_bits / (total_bytes + 1e-19)}
