"""Load trained models and run them over data, outside the training loop.

These are the pieces a post's figure scripts need: list the checkpoints a run
saved, restore the model at any of them, and push a whole dataset through a
function (a forward pass, a probe, a gradient) in fixed-size batches.

A checkpoint sweep, e.g. for "first-layer filters over training"::

    from toylib.harness import inference

    template = build_model(config, jax.random.key(0))
    for step in inference.checkpoint_steps(ckpt_dir):
        model = inference.load_model(template, ckpt_dir, step)
        logits = inference.batched_apply(forward, model, {"image": x_test}, 500)
"""

import pathlib
import typing

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp


def _manager(checkpoint_dir: str | pathlib.Path) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        str(checkpoint_dir), options=ocp.CheckpointManagerOptions(read_only=True)
    )


def checkpoint_steps(checkpoint_dir: str | pathlib.Path) -> list[int]:
    """Return the steps saved under ``checkpoint_dir``, in increasing order."""
    manager = _manager(checkpoint_dir)
    try:
        return sorted(int(s) for s in manager.all_steps())
    finally:
        manager.close()


def latest_step(checkpoint_dir: str | pathlib.Path) -> int:
    """Return the most recent saved step. Raises ``ValueError`` if there is none."""
    steps = checkpoint_steps(checkpoint_dir)
    if not steps:
        raise ValueError(f"No checkpoints found under {str(checkpoint_dir)!r}")
    return steps[-1]


def load_model(
    template: typing.Any,
    checkpoint_dir: str | pathlib.Path,
    step: int | None = None,
) -> typing.Any:
    """Restore the model saved by ``Experiment`` at ``step`` (default: latest).

    ``template`` is a freshly initialized model with the same configuration as
    the trained one; only its structure, shapes and dtypes are used. The
    optimizer state and dataset iterator in the checkpoint are ignored.
    """
    if step is None:
        step = latest_step(checkpoint_dir)
    host_template = jax.tree.map(np.asarray, template)
    manager = _manager(checkpoint_dir)
    try:
        restored = manager.restore(
            step,
            args=ocp.args.Composite(model=ocp.args.StandardRestore(host_template)),
        )
    finally:
        manager.close()
    return jax.tree.map(jnp.asarray, restored["model"])


def batched_apply(
    fn: typing.Callable[[typing.Any, typing.Any], typing.Any],
    model: typing.Any,
    inputs: typing.Any,
    batch_size: int,
) -> typing.Any:
    """Compute ``fn(model, batch)`` over ``inputs`` in batches; concatenate results.

    ``inputs`` is a pytree of arrays sharing a leading axis of length N (for
    example ``{"image": x_test}``). ``fn`` must return a pytree of arrays whose
    leading axis is the batch. The last batch is padded up to ``batch_size`` by
    repeating its final row, so ``fn`` is compiled once; padded rows are dropped
    from the output. Results come back as NumPy arrays of leading length N.

    Pass the model as an argument rather than closing over it: the compiled
    function is then reused across a checkpoint sweep.
    """
    leaves = jax.tree.leaves(inputs)
    if not leaves:
        raise ValueError("inputs has no arrays")
    n = leaves[0].shape[0]
    jitted = jax.jit(fn)

    outputs = []
    for start in range(0, n, batch_size):
        count = min(batch_size, n - start)
        batch = jax.tree.map(lambda x: np.asarray(x[start : start + count]), inputs)
        if count < batch_size:
            pad = batch_size - count
            batch = jax.tree.map(
                lambda x: np.concatenate([x, np.repeat(x[-1:], pad, axis=0)]), batch
            )
        out = jitted(model, batch)
        outputs.append(jax.tree.map(lambda y: np.asarray(y)[:count], out))
    return jax.tree.map(lambda *ys: np.concatenate(ys, axis=0), *outputs)
