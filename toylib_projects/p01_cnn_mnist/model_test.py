"""The model and the loss, on synthetic data so no download is needed.

``pipeline_test.py`` checks the plumbing; these check that the loss is the right
number and that the model can learn.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from toylib_projects.p01_cnn_mnist import model


@pytest.mark.parametrize(
    "image_shape",
    [(28, 28, 1), (32, 32, 3), (30, 30, 1)],  # 30 doesn't divide by 4: pooling rounds down.
)
def test_logits_shape(image_shape):
    net = model.build_model(model.ModelConfig(image_shape=image_shape), jax.random.key(0))
    assert net(jnp.zeros((4, *image_shape))).shape == (4, 10)


def test_loss_of_uniform_logits_is_ln_num_classes():
    loss = model.loss_fn(jnp.zeros((4, 10)), jnp.array([0, 3, 7, 9]))
    assert float(loss) == pytest.approx(math.log(10))


def test_loss_matches_optax():
    # Negative logits on purpose: they are scores, not probabilities.
    logits = jnp.array([[2.0, -1.0, 0.5], [-3.0, 1.0, 0.0]])
    labels = jnp.array([0, 2])
    expected = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    assert float(model.loss_fn(logits, labels)) == pytest.approx(float(expected))


def test_overfits_one_batch():
    rng = np.random.default_rng(0)
    batch = {
        "image": rng.normal(size=(8, 28, 28, 1)).astype(np.float32),
        "label": np.arange(8, dtype=np.int32),
    }
    net = model.build_model(model.ModelConfig(), jax.random.key(0))
    optimizer = optax.adam(1e-3)
    state = optimizer.init(net)

    @jax.jit
    def step(net, state):
        (loss, _), grads = jax.value_and_grad(model.forward, has_aux=True)(net, batch)
        updates, state = optimizer.update(grads, state, net)
        return optax.apply_updates(net, updates), state, loss

    losses = []
    for _ in range(30):
        net, state, loss = step(net, state)
        losses.append(float(loss))

    assert losses[-1] < 0.1 * losses[0]
    np.testing.assert_array_equal(np.asarray(net(batch["image"])).argmax(-1), batch["label"])


def test_default_cnn_keeps_its_parameter_count():
    # The CIFAR-10 checkpoints and the explorer depend on this layout.
    net = model.build_model(model.ModelConfig(image_shape=(32, 32, 3)), jax.random.key(0))
    assert sum(leaf.size for leaf in jax.tree.leaves(net)) == 97_290


def test_cnn_stages_come_from_the_config():
    config = model.ModelConfig(
        image_shape=(160, 160, 3), channels=(32, 64, 128, 128), pools=(True,) * 4
    )
    net = model.build_model(config, jax.random.key(0))
    logits, acts = net(jnp.zeros((2, 160, 160, 3)), return_intermediates=True)

    assert logits.shape == (2, 10)
    assert acts["conv1"].shape == (2, 80, 80, 32)
    assert acts["conv4"].shape == (2, 10, 10, 128)
    assert net.head.weights.shape[0] == 10 * 10 * 128


@pytest.mark.parametrize("hidden_dims", [(), (16,), (16, 8)])
def test_mlp_shapes_and_loss(hidden_dims):
    config = model.MLPConfig(image_shape=(8, 8, 3), hidden_dims=hidden_dims)
    net = model.build_mlp(config, jax.random.key(0))
    batch = {"image": np.zeros((4, 8, 8, 3), np.float32), "label": np.arange(4)}

    assert net(batch["image"]).shape == (4, 10)
    assert len(net.mlp.hidden_layers) == len(hidden_dims)
    assert np.isfinite(float(model.forward(net, batch)[0]))


def test_mlp_weight_count():
    # 19.7M weights in the first layer is the argument for sharing them.
    config = model.MLPConfig(image_shape=(160, 160, 3), hidden_dims=(256,))
    net = model.build_mlp(config, jax.random.key(0))
    assert net.mlp.hidden_layers[0].weights.size == 160 * 160 * 3 * 256


@pytest.mark.parametrize("hidden_dims", [(), (64,)])
def test_mlp_overfits_one_batch(hidden_dims):
    rng = np.random.default_rng(0)
    batch = {
        "image": rng.normal(size=(8, 16, 16, 3)).astype(np.float32),
        "label": np.arange(8, dtype=np.int32),
    }
    config = model.MLPConfig(image_shape=(16, 16, 3), hidden_dims=hidden_dims)
    net = model.build_mlp(config, jax.random.key(0))
    optimizer = optax.adam(1e-3)
    state = optimizer.init(net)

    @jax.jit
    def step(net, state):
        (loss, _), grads = jax.value_and_grad(model.forward, has_aux=True)(net, batch)
        updates, state = optimizer.update(grads, state, net)
        return optax.apply_updates(net, updates), state, loss

    first = None
    for _ in range(200):
        net, state, loss = step(net, state)
        first = first if first is not None else float(loss)
    assert float(loss) < 0.1 * first
