"""Classification metrics for the harness (``Task(metrics=[...])``)."""

import dataclasses

import jax.numpy as jnp
import jaxtyping as jt


@dataclasses.dataclass
class Accuracy:
    """Fraction of the batch whose arg-max logit is the label.

    Reads ``aux["logits"]`` (as returned by ``model.forward``) and ``batch["label"]``.
    """

    def __call__(
        self, loss: jt.ArrayLike, aux: jt.PyTree, batch: jt.PyTree
    ) -> dict[str, jt.ArrayLike]:
        del loss
        predictions = jnp.argmax(aux["logits"], axis=-1)
        return {"accuracy": jnp.mean(predictions == batch["label"])}
