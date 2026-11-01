"""Tests for the arithmetic behind the post's figures."""

from unittest import mock

import jax
import numpy as np

from toylib.nn import layers
from toylib_projects.p01_cnn_mnist import figures, resnet


def test_as_images_centers_on_grey():
    images = figures.as_images(np.zeros((160 * 160 * 3, 2)) + 1e-3)
    assert images.shape == (2, 160, 160, 3)
    assert images.max() == 1.0


def test_evidence_map_adds_up_to_the_logit():
    config = resnet.ResNetConfig(image_shape=(32, 32, 3), width=8, blocks_per_stage=1)
    net = resnet.build_resnet(config, jax.random.key(0))
    image = np.asarray(jax.random.normal(jax.random.key(1), (1, 32, 32, 3)))
    logits, acts = net(image, return_intermediates=True)
    head = np.asarray(net.head.weights)
    for cls in (0, 7):
        cam = figures.evidence_map(np.asarray(acts["block2"][0]), head, cls)
        total = cam.mean() + float(net.head.bias[cls])
        assert abs(total - float(logits[0, cls])) < 1e-3


def test_receptive_fields_match_a_hand_count():
    sizes = figures.receptive_fields(figures.RESNET)
    assert sizes["stem"] == 7
    assert sizes["block0"] == 15
    assert sizes["block2"] == 35
    assert sizes["block7"] == 219


def test_receptive_field_matches_the_network():
    # GroupNorm's statistics span the whole image, so every pixel has some tiny
    # effect on every unit. Without it the gradient stops at the convolutions' reach.
    config = resnet.ResNetConfig(image_shape=(64, 64, 3), width=8, blocks_per_stage=1)
    net = resnet.build_resnet(config, jax.random.key(0))
    image = np.asarray(jax.random.normal(jax.random.key(1), (1, 64, 64, 3)))

    def unit(x):
        return net(x, return_intermediates=True)[1]["block1"][0, 8, 8].sum()

    with mock.patch.object(layers.GroupNorm, "__call__", lambda self, x: x):
        field = np.abs(np.asarray(jax.grad(unit)(image))).sum((0, 3))
    rows = np.flatnonzero(field.sum(1))
    assert rows[-1] - rows[0] + 1 == figures.receptive_fields(config)["block1"]


def test_shifted_crops_move_real_pixels():
    stored = np.arange(200)[None, None, :, None] * np.ones((1, 200, 200, 3), np.uint8)
    assert figures.shifted_crops(stored, 0)[0, 0, 0, 0] == 20
    assert figures.shifted_crops(stored, 15)[0, 0, 0, 0] == 35
    assert figures.shifted_crops(stored, -20).shape == (1, 160, 160, 3)
