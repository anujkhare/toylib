"""Export the trained ResNet and everything the post's interactive views show.

Reads the checkpoints in the bucket and writes ``data.js`` and ``weights.js`` next to
this file. ``weights.js`` loads only when a reader asks for a live forward pass::

    GOOGLE_CLOUD_PROJECT=practicalml-db49f uv run python -m toylib_projects.p01_cnn_mnist.post.explorer.build_data
"""

import base64
import json
import pathlib

import jax
import numpy as np

from toylib_projects.p01_cnn_mnist import imagenette
from toylib_projects.p01_cnn_mnist.post import figures

HERE = pathlib.Path(__file__).parent
# Channels kept per stage: the strongest on each photo, scaled to uint8.
CHANNELS = {"stem": 8, "block1": 8, "block3": 12, "block5": 16, "block7": 16}
CORRECT_PER_CLASS = 2
MISTAKES = 10


def b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.asarray(array).astype(dtype).tobytes()).decode()


def pick_photos(labels: np.ndarray, logits: np.ndarray) -> list[int]:
    """Clear photos of every class, then the gas pump mix-ups and other mistakes."""
    predictions = logits.argmax(-1)
    top = logits.max(-1)
    picks = []
    for cls in range(10):
        right = np.flatnonzero((labels == cls) & (predictions == cls))
        ranked = right[np.argsort(-top[right])]
        picks += [int(ranked[0]), int(ranked[len(ranked) // 2])]
    wrong = np.flatnonzero(predictions != labels)
    mixups = wrong[
        np.isin(labels[wrong], [figures.GAS_PUMP, figures.CASSETTE_PLAYER])
        & np.isin(predictions[wrong], [figures.GAS_PUMP, figures.CASSETTE_PLAYER])
    ]
    picks += mixups[np.argsort(-top[mixups])][:4].tolist()
    others = np.setdiff1d(wrong, mixups)
    picks += others[np.argsort(-top[others])][: MISTAKES - 4].tolist()
    return picks


def export_weights(net) -> None:
    """All weights as one Float32 blob, with each array's name, offset and length."""
    arrays = {
        "stem": net.stem.weights,
        "stem_norm.scale": net.stem_norm.scale,
        "stem_norm.bias": net.stem_norm.bias,
        "head.weights": net.head.weights,
        "head.bias": net.head.bias,
    }
    for i, block in enumerate(net.blocks):
        arrays |= {
            f"block{i}.conv1": block.conv1.weights,
            f"block{i}.norm1.scale": block.norm1.scale,
            f"block{i}.norm1.bias": block.norm1.bias,
            f"block{i}.conv2": block.conv2.weights,
            f"block{i}.norm2.scale": block.norm2.scale,
            f"block{i}.norm2.bias": block.norm2.bias,
        }
        if block.project is not None:
            arrays[f"block{i}.project"] = block.project.weights
    flat = [np.asarray(a, np.float32).ravel() for a in arrays.values()]
    offsets = np.cumsum([0] + [len(a) for a in flat[:-1]])
    bundle = {
        "data": b64(np.concatenate(flat), "<f4"),
        "index": [[n, int(o), len(a)] for n, o, a in zip(arrays, offsets, flat)],
    }
    text = "self.EXPLORER_WEIGHTS = " + json.dumps(bundle, separators=(",", ":")) + ";\n"
    (HERE / "weights.js").write_text(text)
    print(f"wrote weights.js, {len(text) / 1e6:.1f} MB")


def export_activations(acts: dict[str, np.ndarray]) -> dict:
    """The strongest channels of each stage, per photo, each with its own scale."""
    out = {}
    for stage, count in CHANNELS.items():
        maps = acts[stage]  # (N, H, W, C)
        order = np.argsort(-maps.mean((1, 2)), axis=-1)[:, :count]
        top = np.take_along_axis(maps, order[:, None, None, :], axis=-1)
        scale = top.max((1, 2, 3), keepdims=True) + 1e-6
        out[stage] = {
            "side": maps.shape[1],
            "channels": maps.shape[-1],
            "count": count,
            "maps": b64(top / scale * 255, "u1"),
        }
    return out


def main() -> None:
    net = figures.trained_resnet()
    _, raw, labels = figures.test_set()
    logits = figures.test_logits()
    predictions = logits.argmax(-1)
    picks = pick_photos(labels, logits)
    print(f"{len(picks)} photos")

    acts = {}
    for start in range(0, len(picks), 8):
        batch = figures.activations(net, raw[picks[start : start + 8]])
        for name, value in batch.items():
            acts.setdefault(name, []).append(value)
    acts = {name: np.concatenate(value) for name, value in acts.items()}
    head = np.asarray(net.head.weights)
    cams = acts["block7"] @ head  # (N, 10, 10, classes)

    probabilities = np.asarray(jax.nn.softmax(logits))
    confusion = np.zeros((10, 10), int)
    np.add.at(confusion, (labels, predictions), 1)
    right = predictions == labels
    payload = {
        "classNames": list(imagenette.CLASS_NAMES),
        "size": figures.IMAGE_SHAPE[0],
        "mean": list(figures.INFO.mean),
        "std": list(figures.INFO.std),
        "images": b64(raw[picks], "u1"),
        "labels": labels[picks].tolist(),
        "predictions": predictions[picks].tolist(),
        "probabilities": probabilities[picks].round(4).tolist(),
        "logits": logits[picks].round(3).tolist(),
        "cams": b64(np.moveaxis(cams, -1, 1), "<f4"),
        "activations": export_activations(acts),
        "confusion": confusion.tolist(),
        "testAccuracy": round(float(right.mean()), 4),
        "medianConfidence": {
            "right": round(float(np.median(probabilities.max(-1)[right])), 4),
            "wrong": round(float(np.median(probabilities.max(-1)[~right])), 4),
        },
        # JAX's logits for the first photos, which cnn.js must reproduce.
        "parity": logits[picks[:4]].astype(float).round(5).tolist(),
    }
    text = "window.EXPLORER_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    (HERE / "data.js").write_text(text)
    print(f"wrote data.js, {len(text) / 1e6:.1f} MB")
    export_weights(net)


if __name__ == "__main__":
    main()
