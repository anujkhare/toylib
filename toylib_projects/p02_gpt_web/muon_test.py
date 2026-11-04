"""Tests for muon.py."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from toylib_projects.p02_gpt_web import muon


def nanochat_step(grad, param, momentum, second, mu, lr, wd, beta2):
    """nanochat's `muon_step_fused`, transcribed into NumPy in its [out, in] layout,
    in float32 throughout."""
    momentum = momentum + (1 - mu) * (grad - momentum)  # lerp_
    g = grad + mu * (momentum - grad)
    X = g.copy()
    target = np.linalg.norm(X) / X.shape[-2] ** 0.5
    row_norm = np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-6)
    X = X * (target / row_norm)
    X = X / (np.linalg.norm(X) * 1.01 + 1e-6)
    for a, b, c in muon.POLAR_EXPRESS:
        if g.shape[-2] > g.shape[-1]:
            A = X.T @ X
            X = a * X + X @ (b * A + c * (A @ A))
        else:
            A = X @ X.T
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    g = g * (min(g.shape) ** 0.5 / max(np.linalg.norm(g), 1e-6))
    red_dim = -1 if g.shape[-2] >= g.shape[-1] else -2
    v_mean = np.mean(g**2, axis=red_dim, keepdims=True)
    size = g.shape[red_dim]
    v_norm = np.sqrt(v_mean.sum() * size)
    second = second + (1 - beta2) * (v_mean - second)
    step_size = 1 / np.sqrt(np.maximum(second, 1e-10))
    v_norm_new = np.sqrt((v_mean * size * step_size**2).sum())
    g = g * step_size * (v_norm / max(v_norm_new, 1e-10))
    lr = lr * max(1.0, g.shape[-2] / g.shape[-1]) ** 0.5
    mask = (g * param) >= 0
    new_param = param - (lr * g + lr * wd * param * mask)
    return new_param, momentum, second


@pytest.mark.parametrize("shape", [(6, 4), (4, 6), (5, 5)])  # toylib's [in, out]
def test_matches_nanochat_line_for_line(shape):
    keys = jax.random.split(jax.random.key(0), 3)
    param = jax.random.normal(keys[0], shape)
    grads = [jax.random.normal(k, shape) for k in jax.random.split(keys[1], 3)]
    opt = muon.muon(learning_rate=0.02, momentum=0.9, weight_decay=0.3, dtype=jnp.float32)
    state = opt.init({"w": param})
    params = {"w": param}

    ref_param = np.asarray(param).T
    ref_m = np.zeros_like(ref_param)
    rows, cols = ref_param.shape
    ref_v = np.zeros((rows, 1) if rows >= cols else (1, cols), np.float32)
    for grad in grads:
        updates, state = opt.update({"w": grad}, state, params)
        params = optax.apply_updates(params, updates)
        ref_param, ref_m, ref_v = nanochat_step(
            np.asarray(grad).T, ref_param, ref_m, ref_v, mu=0.9, lr=0.02, wd=0.3, beta2=0.9
        )
    np.testing.assert_allclose(params["w"], ref_param.T, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(state.momentum["w"], ref_m.T, rtol=1e-5, atol=1e-6)
    assert int(state.count) == 3


@pytest.mark.parametrize("shape", [(64, 16), (16, 64), (32, 32)])
def test_orthogonalize_gives_singular_values_near_one(shape):
    x = jax.random.normal(jax.random.key(1), shape)
    s = jnp.linalg.svd(muon.orthogonalize(x, dtype=jnp.float32), compute_uv=False)
    # Five iterations stop short of exact, most of all for the tiny singular values
    # of a random square matrix; Muon+ fixes the overall norm.
    assert float(s.max()) < 1.3
    assert float(jnp.mean((s > 0.7) & (s < 1.3))) > 0.9
    np.testing.assert_allclose(jnp.sqrt(jnp.sum(s**2)), min(shape) ** 0.5, rtol=1e-4)


def test_bfloat16_iterations_stay_close_to_float32():
    x = jax.random.normal(jax.random.key(2), (48, 24))
    low, full = muon.orthogonalize(x), muon.orthogonalize(x, dtype=jnp.float32)
    # Rounding through five iterations moves entries a little, not the direction.
    cosine = jnp.sum(low * full) / (jnp.linalg.norm(low) * jnp.linalg.norm(full))
    assert float(cosine) > 0.99


def test_zero_gradient_only_decays():
    param = jnp.ones((3, 2))
    opt = muon.muon(learning_rate=0.1, weight_decay=0.5)
    state = opt.init(param)
    updates, _ = opt.update(jnp.zeros_like(param), state, param)
    # A zero update agrees with every sign, so every weight decays by lr * wd.
    np.testing.assert_allclose(updates, -0.05 * param, atol=1e-6)


def test_schedules_read_the_step_count():
    seen = []
    opt = muon.muon(learning_rate=lambda c: 0.01 * (c + 1), momentum=lambda c: 0.9)
    param = jax.random.normal(jax.random.key(3), (4, 4))
    state = opt.init(param)
    for _ in range(2):
        updates, state = opt.update(jax.random.normal(jax.random.key(4), (4, 4)), state, param)
        seen.append(float(jnp.linalg.norm(updates)))
    # The same gradient direction twice, with the learning rate doubled.
    assert seen[1] > 1.5 * seen[0]


def test_rejects_non_matrices():
    with pytest.raises(ValueError, match="matrices"):
        muon.muon(0.1).init({"b": jnp.zeros(3)})


def test_needs_params():
    opt = muon.muon(0.1)
    state = opt.init(jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="parameters"):
        opt.update(jnp.zeros((2, 2)), state)


def test_works_inside_multi_transform_and_jit():
    params = {"w": jax.random.normal(jax.random.key(5), (8, 4)), "b": jnp.zeros(4)}
    opt = optax.multi_transform(
        # Muon's steps have a fixed size, so the rate decays to land on the target.
        {"muon": muon.muon(lambda c: 0.2 * (1 - c / 200)), "adam": optax.adam(1e-3)},
        {"w": "muon", "b": "adam"},
    )
    state = opt.init(params)
    target = jax.random.normal(jax.random.key(6), (8, 4))

    @jax.jit
    def step(params, state):
        loss, grads = jax.value_and_grad(lambda p: jnp.sum((p["w"] - target) ** 2))(params)
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    losses = []
    for _ in range(200):
        params, state, loss = step(params, state)
        losses.append(float(loss))
    assert losses[-1] < 0.1 * losses[0]
