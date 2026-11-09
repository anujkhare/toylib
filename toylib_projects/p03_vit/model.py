"""The Vision Transformer for post 3. The model and the loss are yours to write.

``train.py`` and the notebook need four things from this module:

* ``ModelConfig``: the model's hyperparameters. Add fields as you need them, and
  give each a default so ``train.RUNS`` keeps working.
* ``build_model(config, key) -> ViT``: the harness's ``model_factory``.
* ``ViT.__call__(images, return_intermediates=False)``: standardized float32
  images ``(B, H, W, C)`` to logits ``(B, num_classes)``. With
  ``return_intermediates=True``, return ``(logits, acts)``, where ``acts`` maps
  names to arrays for the post's figures, including ``"blocks.0.attn_weights"``,
  ``"blocks.1.attn_weights"``, ... of shape ``(B, num_heads, T, T)``.
* ``loss_fn(logits, labels) -> scalar``.

``forward`` wires them into the harness and shouldn't need changes.
``model_test.py`` checks this contract, and its tests skip until the model exists.

Pieces already in ``toylib.nn``: ``layers.Linear``, ``layers.Embedding``,
``layers.Conv2D``, ``mlp.MLP`` and ``attention.MultiHeadAttention``. The attention
was built for the GPT: leave ``mask`` and ``rope`` as ``None`` for full attention
without rotary embeddings, and note that it normalizes queries and keys and starts
its output projection at zero. There's no LayerNorm yet.
"""

import dataclasses

import jax
import jaxtyping as jt

from toylib.nn import module


@dataclasses.dataclass
class ModelConfig:
    image_shape: tuple[int, int, int] = (160, 160, 3)  # (H, W, C)
    num_classes: int = 10
    patch_size: int = 16  # 160x160 images give 10x10 = 100 patches.
    dim: int = 192  # The width of every token. These defaults are ViT-Ti/16's shape.
    depth: int = 12  # Transformer blocks.
    num_heads: int = 3
    mlp_dim: int = 768  # The hidden width of each block's MLP, usually 4 * dim.


class ViT(module.Module):
    image_shape: tuple[int, int, int]
    num_classes: int
    patch_size: int
    dim: int
    depth: int
    num_heads: int
    mlp_dim: int
    key: jt.PRNGKeyArray

    def init(self) -> None:
        raise NotImplementedError("Yours to write: create the ViT's layers.")

    def __call__(
        self,
        images: jt.Float[jt.Array, "B H W C"],
        return_intermediates: bool = False,
    ):
        """Logits for a batch of images.

        With ``return_intermediates=True``, returns ``(logits, acts)``. Under
        ``jit``, mark the flag static.
        """
        raise NotImplementedError("Yours to write: images -> logits.")


def build_model(config: ModelConfig, key: jax.Array) -> ViT:
    """The harness's ``model_factory``: construct and initialize the model."""
    model = ViT(**dataclasses.asdict(config), key=key)
    model.init()
    return model


def loss_fn(
    logits: jt.Float[jt.Array, "batch classes"], labels: jt.Int[jt.Array, " batch"]
) -> jt.Float[jt.Array, ""]:
    """The training loss, averaged over the batch."""
    raise NotImplementedError("Yours to write: logits and labels -> loss.")


def forward(model: module.Module, batch: dict) -> tuple[jax.Array, dict]:
    """The harness's ``forward_fn``: ``(model, batch) -> (loss, aux)``.

    ``aux`` carries the logits, so ``metrics.Accuracy`` can read them.
    """
    logits = model(batch["image"])
    loss = loss_fn(logits, batch["label"])
    return loss, {"logits": logits}
