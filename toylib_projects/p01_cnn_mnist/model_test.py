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
