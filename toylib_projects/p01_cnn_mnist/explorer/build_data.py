"""Export the trained post 1 CNN and everything the explorer page shows.

Reads the checkpoints written by ``01_data_and_pipeline_cifar10.ipynb`` and the
ResNet runs under ``outputs/``, and writes ``data.js`` next to this file::

    uv run --with scikit-learn python -m toylib_projects.p01_cnn_mnist.explorer.build_data
"""

import base64
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging as absl_logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from toylib import viz
from toylib.harness import analyze, inference
from toylib.nn import layers
from toylib_projects.p01_cnn_mnist import data, model, resnet

HERE = pathlib.Path(__file__).parent
OUTPUTS = HERE.parent / "outputs"
RUN_DIR = OUTPUTS / "notebook_cifar10"
NOAUG_DIR = OUTPUTS / "cifar10_no_augmentation"
SNAPSHOT_EPOCHS = [0, 1, 2, 5, 10, 20, 40]
STEPS_PER_EPOCH = 390
IMAGES_PER_CLASS = 60
NUM_PATCHES = 6
MAX_SHIFT = 8
CONFIG = model.ModelConfig(image_shape=(32, 32, 3))
INFO = data.DATASETS["cifar10"]

# Display name, directory under outputs/, depth, and config for each ResNet run.
RESNET_RUNS = [
    ("ResNet-8", "resnet_resnet8_e40", 8, resnet.ResNetConfig(blocks_per_stage=1)),
    ("Plain-20", "resnet_plain20_e40", 20, resnet.ResNetConfig(residual=False)),
    ("ResNet-20", "resnet_resnet20_e40", 20, resnet.ResNetConfig()),
    ("Wide ResNet-20", "resnet_resnet20w32_e40", 20, resnet.ResNetConfig(width=32)),
    ("Plain-56", "resnet_plain56_e15", 56, resnet.ResNetConfig(residual=False, blocks_per_stage=9)),
    ("ResNet-56", "resnet_resnet56_e15", 56, resnet.ResNetConfig(blocks_per_stage=9)),
]


def b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.asarray(array).astype(dtype).tobytes()).decode()


def layer_outputs(net: model.CNN, x: jax.Array) -> dict[str, jax.Array]:
    """The CNN's activations after each operation, as ``explorer/cnn.js`` computes them."""
    conv1 = jax.nn.relu(net.conv1.conv_layer(x))
    pool1 = layers.max_pool(conv1, factor=2)
    conv2_raw = net.conv2.conv_layer(pool1)
    conv2 = jax.nn.relu(conv2_raw)
    pool2 = layers.max_pool(conv2, factor=2)
    conv3_raw = net.conv3.conv_layer(pool2)
    conv3 = jax.nn.relu(conv3_raw)
    return {
        "conv1": conv1,
        "pool1": pool1,
        "conv2_raw": conv2_raw,
        "conv2": conv2,
        "pool2": pool2,
        "conv3_raw": conv3_raw,
        "conv3": conv3,
    }


def predict(net, images: np.ndarray) -> np.ndarray:
    apply = jax.jit(lambda m, b: m(b["image"]))
    return inference.batched_apply(apply, net, {"image": images}, 500)


def pick_images(labels: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """A class-balanced sample of the test set, plus the most confident mistakes."""
    picked = [np.flatnonzero(labels == c)[:IMAGES_PER_CLASS] for c in range(10)]
    wrong = np.flatnonzero(logits.argmax(-1) != labels)
    confidence = jax.nn.softmax(logits[wrong]).max(-1)
    picked.append(wrong[np.argsort(-np.asarray(confidence))[:24]])
    return np.unique(np.concatenate(picked))


def export_weights(net: model.CNN) -> dict[str, str]:
    return {
        "conv1_w": b64(net.conv1.conv_layer.weights, "<f4"),
        "conv1_b": b64(net.conv1.conv_layer.bias, "<f4"),
        "conv2_w": b64(net.conv2.conv_layer.weights, "<f4"),
        "conv2_b": b64(net.conv2.conv_layer.bias, "<f4"),
        "conv3_w": b64(net.conv3.conv_layer.weights, "<f4"),
        "conv3_b": b64(net.conv3.conv_layer.bias, "<f4"),
        "head_w": b64(net.head.weights, "<f4"),
        "head_b": b64(net.head.bias, "<f4"),
    }


def receptive_box(layer: str, i: int) -> tuple[int, int]:
    """First and last input pixel seen along one axis by unit ``i`` of ``layer``."""
    return {
        "conv2": (2 * i - 3, 2 * i + 4),
        "conv3": (4 * i - 7, 4 * i + 10),
    }[layer]


def crop_with_fill(image: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    """A ``size`` square of a uint8 image; pixels outside it are mid gray."""
    out = np.full((size, size, 3), 128, np.uint8)
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + size, 32), min(left + size, 32)
    out[y0 - top : y1 - top, x0 - left : x1 - left] = image[y0:y1, x0:x1]
    return out


def top_patches(net, images, raw, layer: str, size: int) -> np.ndarray:
    """For each channel, the crops of the test images that excite it most."""
    fn = jax.jit(lambda b: layer_outputs(net, b)[layer])
    best_scores, best_places = [], []
    for start in range(0, len(images), 500):
        acts = np.asarray(fn(images[start : start + 500]))
        b, h, w, c = acts.shape
        flat = acts.transpose(0, 3, 1, 2).reshape(b, c, h * w)
        place = flat.argmax(-1)
        best_scores.append(flat.max(-1))
        best_places.append(place)
    scores = np.concatenate(best_scores)  # (N, C)
    places = np.concatenate(best_places)
    side = {"conv2": 16, "conv3": 8}[layer]
    out = []
    for c in range(scores.shape[1]):
        for image_index in np.argsort(-scores[:, c])[:NUM_PATCHES]:
            row, col = np.unravel_index(places[image_index, c], (side, side))
            top, left = receptive_box(layer, int(row))[0], receptive_box(layer, int(col))[0]
            out.append(crop_with_fill(raw[image_index], top, left, size))
    return np.stack(out)


def channel_activity(net, images: np.ndarray, layer: str) -> list[float]:
    """For each channel, the fraction of images on which it fires anywhere."""
    fn = jax.jit(lambda b: (layer_outputs(net, b)[layer] > 0).any(axis=(1, 2)).mean(0))
    means = [fn(images[s : s + 500]) for s in range(0, len(images), 500)]
    return np.mean(np.stack(means), axis=0).round(4).tolist()


def feature_visualizations(objective, count: int, key: jax.Array, steps: int = 256) -> np.ndarray:
    """Optimize one input per objective, one image built from three resolutions.

    Pixels alone drift to high-frequency noise, so coarse grids are upsampled and
    summed. With jitter and a smoothness penalty this keeps the result readable.
    """
    optimizer = optax.adam(0.05)
    sizes = (8, 16, 32)

    def compose(grids):
        return sum(jax.image.resize(g, (32, 32, 3), "bilinear") for g in grids)

    def loss_fn(grids, index, k):
        x = compose(grids)
        dy, dx = jax.random.randint(k, (2,), -2, 3)
        moved = jnp.roll(x, (dy, dx), axis=(0, 1))
        roughness = jnp.mean(jnp.abs(jnp.diff(x, axis=0))) + jnp.mean(jnp.abs(jnp.diff(x, axis=1)))
        return -objective(moved[None])[index] + 0.5 * roughness + 0.01 * jnp.mean(x**2)

    def one(index, key):
        keys = jax.random.split(key, len(sizes) + 1)
        grids = [0.1 * jax.random.normal(k, (n, n, 3)) for k, n in zip(keys[1:], sizes)]
        state = optimizer.init(grids)

        def step(carry, k):
            grids, state = carry
            grads = jax.grad(loss_fn)(grids, index, k)
            updates, state = optimizer.update(grads, state, grids)
            return (optax.apply_updates(grids, updates), state), None

        (grids, _), _ = jax.lax.scan(step, (grids, state), jax.random.split(keys[0], steps))
        return compose(grids)

    compiled = jax.jit(one)
    keys = jax.random.split(key, count)
    images = np.stack([np.asarray(compiled(i, keys[i])) for i in range(count)])
    low = np.percentile(images, 1, axis=(1, 2, 3), keepdims=True)
    high = np.percentile(images, 99, axis=(1, 2, 3), keepdims=True)
    return (np.clip((images - low) / (high - low + 1e-6), 0, 1) * 255).astype(np.uint8)


def effective_fields(net, images: np.ndarray) -> dict[str, list[float]]:
    """Mean absolute gradient of a central unit of each layer, against the input."""
    centers = {"conv1": 16, "pool1": 8, "conv2": 8, "pool2": 4, "conv3": 4}
    out = {}
    for layer, c in centers.items():

        def unit(x):
            return layer_outputs(net, x)[layer][:, c, c, :].sum()

        field = np.abs(np.asarray(jax.jit(jax.grad(unit))(jnp.asarray(images)))).mean((0, 3))
        out[layer] = (field / field.max()).round(4).ravel().tolist()
    return out


def shift_images(images: np.ndarray, dx: int) -> np.ndarray:
    """Shift horizontally, filling with zeros (the mean colour after standardizing)."""
    out = np.zeros_like(images)
    if dx >= 0:
        out[:, :, dx:] = images[:, :, : images.shape[2] - dx]
    else:
        out[:, :, :dx] = images[:, :, -dx:]
    return out


def shift_curve(net, images: np.ndarray, labels: np.ndarray) -> list[float]:
    return [
        round(float((predict(net, shift_images(images, dx)).argmax(-1) == labels).mean()), 4)
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1)
    ]


def map_2d(features: np.ndarray) -> np.ndarray:
    reduced = PCA(n_components=50, random_state=0).fit_transform(features)
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(reduced)
    return ((xy - xy.min(0)) / (xy.max(0) - xy.min(0))).round(4)


def resnet_results(images: np.ndarray, labels: np.ndarray) -> tuple[list[dict], dict]:
    rows, shifts = [], {}
    for name, directory, depth, config in RESNET_RUNS:
        log = OUTPUTS / directory / "logs_train.txt"
        if not log.exists():
            continue
        curves = viz.read_log(log)
        planned_epochs = int(directory.rsplit("_e", 1)[1])
        if len(curves.get("val/accuracy", ((), ()))[1]) <= planned_epochs:
            print(f"skipping {name}: still training")
            continue
        net = resnet.build_resnet(config, jax.random.key(0))
        rows.append({
            "name": name,
            "depth": depth,
            "residual": config.residual,
            "params": analyze.print_param_sizes(net, depth=1)[1],
            "test": round(float(curves["val/accuracy"][1][-1]), 4),
            "train": round(float(curves["train/accuracy"][1][-20:].mean()), 4),
            "curve": curves["val/accuracy"][1][1:].round(4).tolist(),
            "train_loss_curve": curves["train/loss"][1].round(4).tolist(),
        })
        if name in ("ResNet-20", "Wide ResNet-20"):
            trained = inference.load_model(net, OUTPUTS / directory / "checkpoints")
            shifts[name] = shift_curve(trained, images, labels)
    return rows, shifts


def main() -> None:
    absl_logging.set_verbosity(absl_logging.ERROR)
    template = model.build_model(CONFIG, jax.random.key(0))
    checkpoints = RUN_DIR / "checkpoints"
    nets = {
        e: inference.load_model(template, checkpoints, e * STEPS_PER_EPOCH)
        for e in SNAPSHOT_EPOCHS
    }
    final = nets[SNAPSHOT_EPOCHS[-1]]

    raw, labels = INFO.load("test")
    images = data.preprocess(raw, INFO)
    logits = predict(final, images)
    picked = pick_images(labels, logits)
    predictions = logits.argmax(-1)
    print(f"final test accuracy {(predictions == labels).mean():.4f}; {len(picked)} images")

    confusion = np.zeros((10, 10), int)
    np.add.at(confusion, (labels, predictions), 1)

    apply_conv3 = jax.jit(lambda m, b: layer_outputs(m, b["image"])["conv3"])
    features = inference.batched_apply(apply_conv3, final, {"image": images[picked]}, 500)
    xy = map_2d(features.reshape(len(picked), -1))

    probabilities = np.asarray(jax.nn.softmax(logits[picked]))
    print("patches and feature visualizations")
    patches = {"conv2": top_patches(final, images, raw, "conv2", 8), "conv3": top_patches(final, images, raw, "conv3", 18)}
    key = jax.random.key(0)
    center = {"conv2": 8, "conv3": 4}
    visuals = {
        layer: feature_visualizations(
            # Before ReLU, so channels that are silent at the start still get a gradient.
            lambda x, layer=layer: layer_outputs(final, x)[layer + "_raw"][0, center[layer], center[layer], :],
            count,
            jax.random.fold_in(key, n),
        )
        for n, (layer, count) in enumerate([("conv2", 64), ("conv3", 64)])
    }
    visuals["class"] = feature_visualizations(lambda x: final(x)[0], 10, jax.random.fold_in(key, 2))

    curves = viz.read_log(RUN_DIR / "logs_train.txt")
    print("receptive fields, shifts, resnets")
    resnets, resnet_shifts = resnet_results(images, labels)
    shifts = {"CNN with augmentation": shift_curve(final, images, labels)}
    if (NOAUG_DIR / "checkpoints").exists():
        no_augmentation = inference.load_model(template, NOAUG_DIR / "checkpoints")
        shifts = {"CNN, no augmentation": shift_curve(no_augmentation, images, labels), **shifts}
    shifts.update(resnet_shifts)

    payload = {
        "classNames": INFO.class_names,
        "mean": list(INFO.mean),
        "std": list(INFO.std),
        "snapshots": [
            {
                "epoch": e,
                "testAccuracy": round(float((predict(nets[e], images).argmax(-1) == labels).mean()), 4),
                "weights": export_weights(nets[e]),
            }
            for e in SNAPSHOT_EPOCHS
        ],
        "images": b64(raw[picked], "u1"),
        "labels": labels[picked].tolist(),
        "testIndex": picked.tolist(),
        "predictions": predictions[picked].tolist(),
        "confidence": probabilities.max(-1).round(4).tolist(),
        "map": xy.tolist(),
        "confusion": confusion.tolist(),
        "testAccuracy": round(float((predictions == labels).mean()), 4),
        "patches": {k: b64(v, "u1") for k, v in patches.items()},
        "activity": {k: channel_activity(final, images, k) for k in patches},
        "visuals": {k: b64(v, "u1") for k, v in visuals.items()},
        "effectiveFields": effective_fields(final, images[:256]),
        "shifts": shifts,
        "maxShift": MAX_SHIFT,
        "training": {
            "epochs": (curves["val/accuracy"][0] / STEPS_PER_EPOCH).round(2).tolist(),
            "testAccuracy": curves["val/accuracy"][1].round(4).tolist(),
            "testLoss": curves["val/loss"][1].round(4).tolist(),
        },
        "resnets": resnets,
        # JAX's logits for the first images, which cnn.js must reproduce.
        "parity": {
            "epoch": SNAPSHOT_EPOCHS[-1],
            "count": 8,
            "logits": logits[picked[:8]].astype(float).round(5).tolist(),
        },
    }
    text = "window.EXPLORER_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    (HERE / "data.js").write_text(text)
    print(f"wrote data.js, {len(text) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
