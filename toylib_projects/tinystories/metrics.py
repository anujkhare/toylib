import dataclasses
import typing
import jax.numpy as jnp
import jaxtyping as jt


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
