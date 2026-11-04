"""nanochat's Muon (its `optim.py`, July 2026), as an optax transformation.

Muon replaces each matrix's update with the nearest orthogonal matrix, so every
direction moves by the same amount. nanochat's version adds, in order:

1. Nesterov momentum, with a momentum that can follow a schedule.
2. MuonEq: rescale each row to the mean row norm before orthogonalizing.
3. Polar Express instead of Newton-Schulz, with 5 tuned iterations
   (https://arxiv.org/abs/2505.16932).
4. Muon+: snap the result's norm to sqrt(min(rows, cols)), its value if exactly
   orthogonal (https://arxiv.org/abs/2602.21545).
5. NorMuon: a second moment per output neuron (or per input, for wide matrices),
   which evens out update sizes across neurons (https://arxiv.org/abs/2510.05491).
6. Cautious weight decay: decay only where the update agrees with the weight's sign.

The learning rate is scaled by sqrt(max(1, rows / cols)), as nanochat does.

nanochat stores weights as [out, in] and toylib as [in, out]. The math below works
on nanochat's layout, so transposes at either end keep it line for line.
"""

import typing

import jax
import jax.numpy as jnp
import optax

# Polar Express coefficients for 5 iterations (safety factor 2e-2, cushion 2).
POLAR_EXPRESS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)

ScalarOrSchedule = float | optax.Schedule


class MuonState(typing.NamedTuple):
    count: jax.Array
    momentum: optax.Params
    second_moment: optax.Params


def _at(value: ScalarOrSchedule, count: jax.Array) -> jax.Array:
    return value(count) if callable(value) else jnp.asarray(value)


def _second_moment_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    rows, cols = shape[1], shape[0]  # nanochat's [out, in]
    return (rows, 1) if rows >= cols else (1, cols)


def orthogonalize(x: jax.Array, ns_steps: int = 5, dtype=jnp.bfloat16) -> jax.Array:
    """MuonEq, then Polar Express, then Muon+, on a matrix in [out, in] layout."""
    rows, cols = x.shape
    x = x.astype(dtype)
    target = jnp.linalg.norm(x.astype(jnp.float32)) / rows**0.5
    row_norm = jnp.maximum(jnp.linalg.norm(x.astype(jnp.float32), axis=-1, keepdims=True), 1e-6)
    x = x * (target / row_norm).astype(dtype)

    x = x / (jnp.linalg.norm(x.astype(jnp.float32)).astype(dtype) * 1.01 + 1e-6)
    for a, b, c in POLAR_EXPRESS[:ns_steps]:
        if rows > cols:  # Tall: iterate on the smaller cols x cols Gram matrix.
            gram = x.T @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
        else:
            gram = x @ x.T
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    g = x.astype(jnp.float32)

    current = jnp.maximum(jnp.linalg.norm(g), 1e-6)
    return g * (min(rows, cols) ** 0.5 / current)


def _update_matrix(grad, param, momentum, second_moment, *, lr, mu, wd, beta2, ns_steps, dtype):
    """One matrix's step, in toylib's [in, out] layout; returns (update, momentum, v)."""
    grad, param = grad.T.astype(jnp.float32), param.T.astype(jnp.float32)
    momentum = momentum.T
    rows, cols = grad.shape

    # Nesterov momentum.
    momentum = momentum + (1 - mu) * (grad - momentum)
    g = grad + mu * (momentum - grad)

    g = orthogonalize(g, ns_steps, dtype)

    # Variance reduction: a second moment per row (or column), then rescale so the
    # update keeps its overall norm.
    axis = -1 if rows >= cols else -2
    size = g.shape[axis]
    v_mean = jnp.mean(jnp.square(g), axis=axis, keepdims=True)
    v_norm = jnp.sqrt(jnp.sum(v_mean) * size)
    second_moment = second_moment + (1 - beta2) * (v_mean - second_moment)
    step_size = jax.lax.rsqrt(jnp.maximum(second_moment, 1e-10))
    v_norm_new = jnp.sqrt(jnp.sum(v_mean * size * jnp.square(step_size)))
    g = g * step_size * (v_norm / jnp.maximum(v_norm_new, 1e-10))

    # Cautious weight decay, and the learning rate scaled for tall matrices.
    lr = lr * max(1.0, rows / cols) ** 0.5
    agrees = (g * param) >= 0
    update = -(lr * g + lr * wd * param * agrees)
    return update.T, momentum.T, second_moment


def muon(
    learning_rate: ScalarOrSchedule,
    momentum: ScalarOrSchedule = 0.95,
    weight_decay: ScalarOrSchedule = 0.0,
    beta2: float = 0.9,
    ns_steps: int = 5,
    dtype=jnp.bfloat16,
) -> optax.GradientTransformation:
    """nanochat's Muon, for 2-D parameters. Schedules take the step count."""

    def init(params):
        for p in jax.tree.leaves(params):
            if p.ndim != 2:
                raise ValueError(f"Muon only takes matrices, not shape {p.shape}")
        return MuonState(
            count=jnp.zeros([], jnp.int32),
            momentum=jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params),
            second_moment=jax.tree.map(
                lambda p: jnp.zeros(_second_moment_shape(p.shape), jnp.float32), params
            ),
        )

    def update(grads, state, params=None):
        if params is None:
            raise ValueError("Muon's weight decay needs the parameters")
        options = dict(
            lr=_at(learning_rate, state.count),
            mu=_at(momentum, state.count),
            wd=_at(weight_decay, state.count),
            beta2=beta2,
            ns_steps=ns_steps,
            dtype=dtype,
        )
        results = jax.tree.map(
            lambda g, p, m, v: _update_matrix(g, p, m, v, **options),
            grads,
            params,
            state.momentum,
            state.second_moment,
        )
        # Exactly a tuple: optax's MaskedNode, for other optimizers' leaves, is a
        # NamedTuple and must pass through.
        updates, new_momentum, new_second = (
            jax.tree.map(lambda r: r[i], results, is_leaf=lambda r: type(r) is tuple)
            for i in range(3)
        )
        updates = jax.tree.map(lambda u, g: u.astype(g.dtype), updates, grads)
        return updates, MuonState(state.count + 1, new_momentum, new_second)

    return optax.GradientTransformation(init, update)
