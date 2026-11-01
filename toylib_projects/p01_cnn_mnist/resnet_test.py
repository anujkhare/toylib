"""The ResNet's structure and wiring, on synthetic data."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from toylib.harness import analyze
from toylib_projects.p01_cnn_mnist import model, resnet


@pytest.mark.parametrize("residual", [True, False])
def test_shapes_through_the_stages(residual):
    config = resnet.ResNetConfig(residual=residual)
    net = resnet.build_resnet(config, jax.random.key(0))
    logits, acts = net(jnp.zeros((4, 32, 32, 3)), return_intermediates=True)

    assert logits.shape == (4, 10)
    assert acts["stem"].shape == (4, 32, 32, 16)
    assert acts["block2"].shape == (4, 32, 32, 16)
    assert acts["block5"].shape == (4, 16, 16, 32)
    assert acts["block8"].shape == (4, 8, 8, 64)


def test_imagenette_layout_ends_at_10x10():
    config = resnet.ResNetConfig(
        image_shape=(160, 160, 3),
        width=32,
        blocks_per_stage=2,
        num_stages=4,
        stem_kernel_size=7,
        stem_stride=2,
    )
    net = resnet.build_resnet(config, jax.random.key(0))
    logits, acts = net(jnp.zeros((2, 160, 160, 3)), return_intermediates=True)

    assert logits.shape == (2, 10)
    assert net.stem.weights.shape == (7, 7, 3, 32)
    assert acts["stem"].shape == (2, 80, 80, 32)
    assert acts["block7"].shape == (2, 10, 10, 256)


def test_only_stage_changes_get_a_projection():
    net = resnet.build_resnet(resnet.ResNetConfig(), jax.random.key(0))
    assert [i for i, b in enumerate(net.blocks) if b.project is not None] == [3, 6]


def test_resnet20_parameter_count():
    net = resnet.build_resnet(resnet.ResNetConfig(), jax.random.key(0))
    _, n_params, _ = analyze.print_param_sizes(net, depth=1)
    assert n_params == 272_282  # 271,632 in the stem and blocks, 650 in the head.


def test_overfits_one_batch():
    rng = np.random.default_rng(0)
    batch = {
        "image": rng.normal(size=(8, 32, 32, 3)).astype(np.float32),
        "label": np.arange(8, dtype=np.int32),
    }
    net = resnet.build_resnet(resnet.ResNetConfig(blocks_per_stage=1), jax.random.key(0))
    optimizer = optax.adam(1e-3)
    state = optimizer.init(net)

    @jax.jit
    def step(net, state):
        (loss, _), grads = jax.value_and_grad(model.forward, has_aux=True)(net, batch)
        updates, state = optimizer.update(grads, state, net)
        return optax.apply_updates(net, updates), state, loss

    losses = []
    for _ in range(40):
        net, state, loss = step(net, state)
        losses.append(float(loss))
    assert losses[-1] < 0.1 * losses[0]
