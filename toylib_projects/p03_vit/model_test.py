"""The contract ``train.py`` relies on. Every test skips until the ViT exists.

These check shapes, the attention weights the figures will read, gradients and
the loss, not how well the model learns.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from toylib.harness import analyze
from toylib_projects.p03_vit import model

SMALL = model.ModelConfig(
    image_shape=(32, 32, 3),
    num_classes=10,
    patch_size=8,  # 4x4 = 16 patches.
    dim=32,
    depth=2,
    num_heads=2,
    mlp_dim=64,
)
NUM_PATCHES = 16


def _build(config=SMALL, seed=0):
    try:
        return model.build_model(config, jax.random.key(seed))
    except NotImplementedError as e:
        pytest.skip(f"The ViT isn't written yet: {e}")


def _loss(logits, labels):
    try:
        return model.loss_fn(logits, labels)
    except NotImplementedError as e:
        pytest.skip(f"The loss isn't written yet: {e}")


def _images(n=4, config=SMALL, seed=0):
    return jax.random.normal(jax.random.key(seed), (n, *config.image_shape))


def test_logits_shape():
    logits = _build()(_images())
    assert logits.shape == (4, SMALL.num_classes)
    assert jnp.isfinite(logits).all()


def test_attention_weights_for_the_figures():
    """Each block's attention: one distribution over tokens per head and query."""
    logits, acts = _build()(_images(), return_intermediates=True)
    assert logits.shape == (4, SMALL.num_classes)
    for i in range(SMALL.depth):
        weights = acts[f"blocks.{i}.attn_weights"]
        assert weights.shape[:2] == (4, SMALL.num_heads)
        tokens = weights.shape[-1]
        assert weights.shape[-2] == tokens
        assert tokens in (NUM_PATCHES, NUM_PATCHES + 1)  # With or without a class token.
        np.testing.assert_allclose(weights.sum(-1), 1.0, atol=1e-5)


def test_no_information_leaks_between_images():
    net = _build()
    images = _images(n=3)
    changed = images.at[1].set(-images[1])
    before, after = np.asarray(net(images)), np.asarray(net(changed))
    np.testing.assert_allclose(before[[0, 2]], after[[0, 2]], atol=1e-5)


def test_same_result_under_jit():
    net = _build()
    images = _images()
    np.testing.assert_allclose(jax.jit(lambda m, x: m(x))(net, images), net(images), atol=1e-5)


def test_loss_matches_cross_entropy():
    logits = jax.random.normal(jax.random.key(1), (5, 10))
    labels = jnp.array([0, 3, 9, 3, 1])
    expected = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    np.testing.assert_allclose(_loss(logits, labels), expected, rtol=1e-5)
    np.testing.assert_allclose(_loss(jnp.zeros((2, 10)), jnp.array([0, 1])), math.log(10), rtol=1e-6)


def test_every_parameter_learns():
    """Each parameter gets a gradient within the first three updates.

    A layer that starts at zero, such as a zero-initialized head or the attention's
    output projection, blocks gradients to everything before it until it has been
    updated, so a parameter only fails if it gets none for three steps.
    """
    net = _build()
    batch = {"image": _images(8), "label": jnp.arange(8) % SMALL.num_classes}
    _loss(jnp.zeros((1, 10)), jnp.zeros(1, jnp.int32))  # Skip if there's no loss yet.
    grad_fn = jax.jit(jax.grad(lambda m: model.forward(m, batch)[0]))

    learned = jax.tree.map(lambda p: False, net)
    for _ in range(3):
        grads = grad_fn(net)
        learned = jax.tree.map(lambda seen, g: seen | bool(jnp.any(g != 0)), learned, grads)
        net = jax.tree.map(lambda p, g: p - 0.1 * g, net, grads)
    paths = jax.tree_util.tree_flatten_with_path(learned)[0]
    stuck = [jax.tree_util.keystr(path) for path, ok in paths if not ok]
    assert not stuck, f"No gradient reached: {stuck}"


def test_overfits_one_batch():
    net = _build()
    batch = {"image": _images(8), "label": jnp.arange(8) % 4}
    optimizer = optax.adam(1e-3)
    state = optimizer.init(net)

    @jax.jit
    def step(net, state):
        loss, grads = jax.value_and_grad(lambda m: model.forward(m, batch)[0])(net)
        updates, state = optimizer.update(grads, state, net)
        return optax.apply_updates(net, updates), state, loss

    losses = []
    for _ in range(100):
        net, state, loss = step(net, state)
        losses.append(float(loss))
    assert losses[-1] < losses[0] / 2, losses[::10]


def test_vit_s16_has_about_22m_parameters():
    """ViT-S/16 on ImageNet has 22.05M parameters with a class token and learned
    position embeddings. Other common choices change that by under 1%."""
    config = dataclasses.replace(
        model.ModelConfig(),
        image_shape=(224, 224, 3),
        num_classes=1000,
        patch_size=16,
        dim=384,
        depth=12,
        num_heads=6,
        mlp_dim=1536,
    )
    params = analyze.print_param_sizes(_build(config), depth=1)[1]
    assert params == pytest.approx(22_050_664, rel=0.02)
