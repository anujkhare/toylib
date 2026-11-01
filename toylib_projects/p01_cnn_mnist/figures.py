"""Every figure in the post, from checkpoints in the GCS bucket.

    GOOGLE_CLOUD_PROJECT=practicalml-db49f uv run python -m toylib_projects.p01_cnn_mnist.figures

Writes PNGs to ``figures/``. The ResNet passes run on the CPU, so this takes minutes.
Name figures after the command to rebuild only those.
"""

import functools
import pathlib
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from toylib import viz
from toylib.harness import inference
from toylib.nn import layers
from toylib_projects.p01_cnn_mnist import data, imagenette, model, resnet

CHECKPOINT_ROOT = "gs://tinystories-checkpoints/toylib-p01-imagenette"
IMAGE_SHAPE = (160, 160, 3)
OUTPUT_DIR = pathlib.Path(model.__file__).parent / "figures"
INFO = data.DATASETS["imagenette"]

SMALL_CNN = model.ModelConfig(IMAGE_SHAPE, channels=(32, 64, 128, 128), pools=(True,) * 4)
RESNET = resnet.ResNetConfig(
    image_shape=IMAGE_SHAPE,
    width=32,
    blocks_per_stage=2,
    num_stages=4,
    stem_kernel_size=7,
    stem_stride=2,
)

# The last layer of each stage, then its side in units.
STAGES = {"stem": 80, "block1": 80, "block3": 40, "block5": 20, "block7": 10}
SHIFTS = (-20, -15, -10, -5, 0, 5, 10, 15, 20)
GAS_PUMP = imagenette.CLASS_NAMES.index("gas pump")
CASSETTE_PLAYER = imagenette.CLASS_NAMES.index("cassette player")


def as_images(weights: np.ndarray) -> np.ndarray:
    """``(H * W * C, N)`` weights -> ``(N, H, W, C)`` images, each scaled about grey."""
    images = np.moveaxis(weights.reshape(*IMAGE_SHAPE, -1), -1, 0)
    scale = np.abs(images).max(axis=(1, 2, 3), keepdims=True)
    return images / (2 * scale) + 0.5


def evidence_map(activations: np.ndarray, head_weights: np.ndarray, cls: int) -> np.ndarray:
    """What each position adds to one class's score, for ``(H, W, C)`` activations.

    The head reads the mean over positions, so the mean of this map plus the
    head's bias is exactly the class's logit.
    """
    return activations @ head_weights[:, cls]


def receptive_fields(config: resnet.ResNetConfig) -> dict[str, int]:
    """How many input pixels across can reach a unit, after each block."""
    size, jump = config.stem_kernel_size, config.stem_stride
    sizes = {"stem": size}
    block = 0
    for stage in range(config.num_stages):
        for i in range(config.blocks_per_stage):
            size += 2 * jump  # The first 3x3 conv sees before any stride applies.
            jump *= 2 if stage > 0 and i == 0 else 1
            size += 2 * jump
            sizes[f"block{block}"] = size
            block += 1
    return sizes


def shifted_crops(images: np.ndarray, dx: int, size: int = 160) -> np.ndarray:
    """Central crops moved ``dx`` pixels sideways, so the shift uses real pixels."""
    top = (images.shape[1] - size) // 2
    left = (images.shape[2] - size) // 2 + dx
    return images[:, top : top + size, left : left + size]


@functools.cache
def test_set() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stored test photos, their central crops, and the labels."""
    stored, labels = imagenette.load("test")
    return stored, data.center_crop(stored, IMAGE_SHAPE[0]), labels


@functools.cache
def trained_resnet(step: int | None = None) -> resnet.ResNet:
    template = resnet.build_resnet(RESNET, jax.random.key(0))
    return inference.load_model(template, f"{CHECKPOINT_ROOT}/resnet18w32_e200", step)


@functools.cache
def trained_small_cnn() -> model.CNN:
    template = model.build_model(SMALL_CNN, jax.random.key(0))
    return inference.load_model(template, f"{CHECKPOINT_ROOT}/small_cnn_e200")


@functools.cache
def trained_mlp() -> model.MLPClassifier:
    config = model.MLPConfig(IMAGE_SHAPE, hidden_dims=(256,))
    template = model.build_mlp(config, jax.random.key(0))
    return inference.load_model(template, f"{CHECKPOINT_ROOT}/mlp_e200")


def logits_of(net, raw: np.ndarray) -> np.ndarray:
    return inference.batched_apply(
        lambda m, b: m(b["image"]), net, {"image": data.preprocess(raw, INFO)}, 200
    )


@functools.cache
def test_logits() -> np.ndarray:
    return logits_of(trained_resnet(), test_set()[1])


def activations(net: resnet.ResNet, raw: np.ndarray) -> dict[str, np.ndarray]:
    """The ResNet's activations for a batch of uint8 photos."""
    _, acts = net(jnp.asarray(data.preprocess(raw, INFO)), return_intermediates=True)
    return {name: np.asarray(acts[name]) for name in STAGES}


def linear_templates() -> plt.Figure:
    """The one image per class that the linear model learned."""
    template = model.build_mlp(model.MLPConfig(IMAGE_SHAPE), jax.random.key(0))
    net = inference.load_model(template, f"{CHECKPOINT_ROOT}/linear_e200")
    return viz.image_grid(
        as_images(np.asarray(net.mlp.output_layer.weights)),
        titles=imagenette.CLASS_NAMES,
        ncols=5,
        suptitle="Linear model: one template per class",
    )


def mlp_hidden_weights(count: int = 32) -> plt.Figure:
    """The hidden units with the largest weights, as images."""
    weights = np.asarray(trained_mlp().mlp.hidden_layers[0].weights)
    largest = np.argsort(-np.linalg.norm(weights, axis=0))[:count]
    return viz.image_grid(
        as_images(weights[:, largest]),
        suptitle="MLP: the 32 hidden units with the largest weights",
    )


def shift_curves(count: int = 1000) -> plt.Figure:
    """Accuracy as the photo moves sideways, and how often the answer stays the same."""
    stored, _, labels = test_set()
    nets = {
        "MLP": trained_mlp(),
        "Small CNN": trained_small_cnn(),
        "ResNet-18 shape": trained_resnet(),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for (name, net), color in zip(nets.items(), viz.CATEGORICAL):
        answers = np.stack(
            [logits_of(net, shifted_crops(stored[:count], dx)).argmax(-1) for dx in SHIFTS]
        )
        accuracy = (answers == labels[:count]).mean(1) * 100
        unchanged = (answers == answers[SHIFTS.index(0)]).mean(1) * 100
        print(name, "accuracy", accuracy.round(1).tolist(), "unchanged", unchanged.round(1).tolist())
        axes[0].plot(SHIFTS, accuracy, label=name, color=color)
        axes[1].plot(SHIFTS, unchanged, label=name, color=color)
    axes[0].set(title="Test accuracy (%)", xlabel="pixels moved sideways", ylim=(0, 100))
    axes[1].set(title="Same answer as before moving (%)", xlabel="pixels moved sideways", ylim=(0, 100))
    axes[0].legend(fontsize=8)
    return fig


def _pick_photos(classes: list[int]) -> list[int]:
    """A confidently correct photo of each class."""
    _, _, labels = test_set()
    logits = test_logits()
    right = np.flatnonzero(logits.argmax(-1) == labels)
    picks = []
    for cls in classes:
        candidates = right[labels[right] == cls]
        picks.append(int(candidates[np.argmax(logits[candidates, cls])]))
    return picks


def stage_activations() -> plt.Figure:
    """Photos through the stages, with the two strongest channels at each."""
    names = imagenette.CLASS_NAMES
    picks = _pick_photos([names.index(n) for n in ("church", "French horn", "tench")])
    raw = test_set()[1][picks]
    acts = activations(trained_resnet(), raw)
    fig, axes = plt.subplots(2 * len(picks), len(STAGES) + 1, figsize=(9, 8.6))
    for row, photo in enumerate(raw):
        for channel in (0, 1):
            ax_row = axes[2 * row + channel]
            ax_row[0].imshow(photo)
            for ax, (stage, side) in zip(ax_row[1:], STAGES.items()):
                maps = acts[stage][row]
                strongest = np.argsort(-maps.mean((0, 1)))[channel]
                ax.imshow(maps[..., strongest], cmap=viz.SEQUENTIAL, interpolation="nearest")
                if row == 0 and channel == 0:
                    ax.set_title(f"{stage}, {side}x{side}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("The same photo through the stages: from edges to coarse blobs")
    return fig


def effective_fields(count: int = 32) -> plt.Figure:
    """Which pixels a central unit depends on, inside the box that can reach it."""
    net = trained_resnet()
    images = jnp.asarray(data.preprocess(test_set()[1][:count], INFO))
    theory = receptive_fields(RESNET)
    fig, axes = plt.subplots(1, len(STAGES), figsize=(10.5, 2.5))
    for ax, (stage, side) in zip(axes, STAGES.items()):

        def unit(x, stage=stage, side=side):
            acts = net(x, return_intermediates=True)[1]
            return acts[stage][:, side // 2, side // 2, :].sum()

        field = np.abs(np.asarray(jax.grad(unit)(images))).mean((0, 3))
        ax.imshow(np.sqrt(field / field.max()), cmap=viz.SEQUENTIAL)
        size = theory[stage]
        ax.add_patch(
            plt.Rectangle(
                (80 - size / 2, 80 - size / 2), size, size,
                fill=False, edgecolor=viz.CRITICAL, linewidth=1.2,
            )
        )
        ax.set(xlim=(-0.5, 159.5), ylim=(159.5, -0.5), xticks=[], yticks=[])
        ax.set_title(f"{stage}: reach {size}", fontsize=9)
    fig.suptitle("A unit can reach the whole box, but it mostly uses the middle")
    return fig


def stem_filters() -> plt.Figure:
    """The 7x7 filters of the first layer, before and after training."""
    rows = [
        ("before training", np.asarray(trained_resnet(0).stem.weights)),
        ("after 200 epochs", np.asarray(trained_resnet().stem.weights)),
    ]
    fig, axes = plt.subplots(4, 16, figsize=(10, 3.3))
    for r, (label, weights) in enumerate(rows):
        for i in range(32):
            ax = axes[2 * r + i // 16, i % 16]
            kernel = weights[..., i]
            ax.imshow(kernel / (2 * np.abs(kernel).max()) + 0.5, interpolation="nearest")
        axes[2 * r, 0].set_title(label, fontsize=9, loc="left")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("First-layer filters: hard to read as pictures")
    return fig


def _depth_probe(images: jax.Array, labels: jax.Array, residual: bool, norm: bool):
    """Gradient and activation size per block, for an untrained 56-layer network."""
    config = resnet.ResNetConfig(blocks_per_stage=9, residual=residual)
    net = resnet.build_resnet(config, jax.random.key(0))

    def measure():
        batch = {"image": images, "label": labels}
        grads = jax.grad(lambda m: model.forward(m, batch)[0])(net)
        acts = net(images, return_intermediates=True)[1]
        return (
            [float(jnp.linalg.norm(b.conv1.weights)) for b in grads.blocks],
            [float(jnp.sqrt(jnp.mean(acts[f"block{i}"] ** 2))) for i in range(27)],
        )

    if norm:
        return measure()
    # Replacing the layer's call keeps the parameters, so only the operation differs.
    with mock.patch.object(layers.GroupNorm, "__call__", lambda self, x: x):
        return measure()


def depth_at_init(count: int = 64) -> plt.Figure:
    """What 56 layers do to the signal, before any training."""
    _, raw, labels = test_set()
    images = jax.image.resize(
        jnp.asarray(data.preprocess(raw[:count], INFO)), (count, 32, 32, 3), "bilinear"
    )
    runs = {
        "no norm, no skips": (False, False),
        "GroupNorm": (False, True),
        "skips": (True, False),
        "GroupNorm and skips": (True, True),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for (name, (residual, norm)), color in zip(runs.items(), viz.CATEGORICAL):
        grads, acts = _depth_probe(images, jnp.asarray(labels[:count]), residual, norm)
        print(f"{name}: grad {grads[0]:.2e} -> {grads[-1]:.2e}, act {acts[0]:.2e} -> {acts[-1]:.2e}")
        axes[0].semilogy(acts, label=name, color=color)
        axes[1].semilogy(np.maximum(grads, 1e-30), label=name, color=color)
    axes[0].set(title="Signal going forward", xlabel="block", ylabel="activation RMS")
    axes[1].set(title="Gradient coming back", xlabel="block", ylabel="gradient norm")
    axes[0].legend(fontsize=8)
    return fig


def evidence_maps() -> plt.Figure:
    """Where the score for the predicted class comes from, one photo per class."""
    picks = _pick_photos(list(range(10)))
    return _evidence_grid(picks, "Each position's share of the winning score")


def _evidence_grid(picks: list[int], title: str) -> plt.Figure:
    net = trained_resnet()
    _, raw, labels = test_set()
    predictions = test_logits().argmax(-1)
    head = np.asarray(net.head.weights)
    last = activations(net, raw[picks])["block7"]
    names = imagenette.CLASS_NAMES
    rows = len(picks) // 5 + (len(picks) % 5 > 0)
    fig, axes = plt.subplots(2 * rows, 5, figsize=(9, 3.7 * rows))
    axes = axes.reshape(2 * rows, 5)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (index, maps) in enumerate(zip(picks, last)):
        r, c = 2 * (k // 5), k % 5
        cam = evidence_map(maps, head, int(predictions[index]))
        axes[r, c].imshow(raw[index])
        title_text = names[predictions[index]]
        if predictions[index] != labels[index]:
            title_text = f"{names[labels[index]]}\nsaid {title_text}"
        axes[r, c].set_title(title_text, fontsize=8)
        axes[r + 1, c].imshow(raw[index])
        scale = np.abs(cam).max()
        axes[r + 1, c].imshow(
            cam, cmap=viz.DIVERGING, vmin=-scale, vmax=scale, alpha=0.65,
            extent=(0, 160, 160, 0), interpolation="bilinear",
        )
    fig.suptitle(title)
    return fig


def confusion_matrix() -> plt.Figure:
    """Mistakes on the whole test set; the diagonal is left out of the scale."""
    _, _, labels = test_set()
    counts = np.zeros((10, 10), int)
    np.add.at(counts, (labels, test_logits().argmax(-1)), 1)
    np.fill_diagonal(counts, 0)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    viz.heatmap(
        counts,
        row_labels=imagenette.CLASS_NAMES,
        col_labels=imagenette.CLASS_NAMES,
        annotate=True,
        fmt="{:.0f}",
        ax=ax,
    )
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_title("Mistakes: true class (rows) against prediction")
    return fig


def gas_pump_mistakes() -> plt.Figure:
    """Gas pumps and cassette players mistaken for each other, with their evidence."""
    _, _, labels = test_set()
    logits = test_logits()
    predictions = logits.argmax(-1)
    picks = []
    for true, wrong in ((GAS_PUMP, CASSETTE_PLAYER), (CASSETTE_PLAYER, GAS_PUMP)):
        found = np.flatnonzero((labels == true) & (predictions == wrong))
        picks += found[np.argsort(-logits[found, wrong])][:5].tolist()
    return _evidence_grid(picks, "Where the wrong answer's evidence came from")


def confidence() -> plt.Figure:
    """How sure the model is when it is right, and when it is wrong."""
    _, _, labels = test_set()
    logits = test_logits()
    top = np.asarray(jax.nn.softmax(logits)).max(-1)
    right = logits.argmax(-1) == labels
    print(f"median confidence: right {np.median(top[right]):.4f}, wrong {np.median(top[~right]):.4f}")
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(top[right], bins, color=viz.GOOD, alpha=0.7, label="right")
    ax.hist(top[~right], bins, color=viz.CRITICAL, alpha=0.8, label="wrong")
    ax.set_yscale("log")
    ax.set(xlabel="probability of the top class", ylabel="test photos")
    ax.legend()
    ax.set_title("Sure of itself, right or wrong")
    return fig


FIGURES = (
    linear_templates,
    mlp_hidden_weights,
    shift_curves,
    stage_activations,
    effective_fields,
    stem_filters,
    depth_at_init,
    evidence_maps,
    confusion_matrix,
    gas_pump_mistakes,
    confidence,
)


def main(names: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    viz.use_style()
    for figure in FIGURES:
        if names and figure.__name__ not in names:
            continue
        print(figure.__name__)
        viz.save(figure(), OUTPUT_DIR / figure.__name__)


if __name__ == "__main__":
    main(sys.argv[1:])
