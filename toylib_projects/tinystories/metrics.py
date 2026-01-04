import dataclasses
import typing
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np


class Metric(typing.Protocol):
    """Protocol for computing and accumulating metrics."""

    def __call__(
        self,
        loss: float,
        aux: jt.PyTree,
        batch: jt.PyTree,
    ) -> dict[str, jt.Array]:
        """Compute final metric value(s) for the given inputs.

        Args:
            loss: The loss value returned by forward_fn
            aux: The auxiliary jt.PyTree returned by forward_fn
            batch: The input batch
        """
        pass


@dataclasses.dataclass
class Loss:
    """Pass-through metric that returns the loss value."""

    def __call__(
        self,
        loss: float,
        aux: jt.PyTree,
        batch: jt.PyTree,
    ) -> dict[str, jt.Array]:
        """Return the loss value.

        Args:
            loss: The loss value returned by forward_fn
            aux: The auxiliary PyTree (unused)
            batch: The input batch (unused)

        Returns:
            Dictionary with 'loss' metric
        """
        del aux, batch
        return {"loss": loss}


@dataclasses.dataclass
class BitsPerByte:
    """Metric that computes bits per byte from per-token loss.

    This metric converts the per-token loss (in nats) to bits per byte by:
    1. Converting nats to bits (multiply by log2(e))
    2. Dividing by the number of bytes per token for each token

    The bytes per token mapping is loaded from an .npy file at initialization.
    """

    bytes_per_token_path: str
    _bytes_per_token: jt.Array = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        """Load the bytes per token array from disk."""
        self._bytes_per_token = jnp.array(np.load(self.bytes_per_token_path))

    def __call__(
        self,
        loss: float,
        aux: jt.PyTree,
        batch: jt.PyTree,
    ) -> dict[str, jt.Array]:
        """Compute bits per byte metric.

        Args:
            loss: The loss value (unused)
            aux: Must contain 'per_token_loss' with shape [batch_size, seq_len]
            batch: Must contain 'inputs' with token ids of shape [batch_size, seq_len]

        Returns:
            Dictionary with 'bits_per_byte' metric
        """
        del loss

        # Get per-token loss (in nats) and token ids
        per_token_loss = aux["per_token_loss"]  # [batch_size, seq_len]
        token_ids = batch["inputs"]  # [batch_size, seq_len]
        mask = batch["mask"]  # [batch_size, seq_len]

        # Look up bytes per token for each token in the batch
        bytes_per_token = self._bytes_per_token[token_ids]  # [batch_size, seq_len]

        # Convert loss from nats to bits (multiply by log2(e))
        # Then divide by bytes per token to get bits per byte
        bits_per_token = per_token_loss * jnp.log2(jnp.e)
        bits_per_byte = bits_per_token / bytes_per_token  # [batch_size, seq_len]
        token_valid = jnp.where(bytes_per_token == -1, 0, 1) * mask

        # Average over all positions
        mean_bits_per_byte = (bits_per_byte * token_valid).sum() / (
            token_valid.sum() + 1e-19
        )

        return {"bits_per_byte": mean_bits_per_byte}
