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

from toylib.nn import module


def _key_name(key: typing.Any) -> str:
    if isinstance(key, jax.tree_util.GetAttrKey):
        return key.name
    if isinstance(key, jax.tree_util.SequenceKey):
        return str(key.idx)
    if isinstance(key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)):
        return str(key.key)
    return str(key)


def restore_tree(
    manager: ocp.CheckpointManager,
    step: int,
    name: str,
    template: typing.Any,
    fillable: typing.Any = None,
) -> typing.Any:
    """Restore item `name` at `step` into the structure of `template`, leaf by leaf.

    Leaves are matched by path, so empty nodes in the template, such as an
    optimizer group with no parameters, don't need to be on disk. A leaf the
    checkpoint lacks is taken from `template` where `fillable` (a matching tree
    of bools) is True, and is an error otherwise, as is a leaf on disk that the
    template doesn't have or one whose shape differs.
    """
    saved = manager.restore(
        step, args=ocp.args.Composite(**{name: ocp.args.StandardRestore()})
    )[name]
    saved_flat = ocp.tree.to_flat_dict(saved)
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(template)
    if fillable is None:
        can_fill = [False] * len(leaves_with_path)
    else:
        can_fill = jax.tree.leaves(fillable)

    leaves, filled = [], []
    for (path, leaf), fill in zip(leaves_with_path, can_fill, strict=True):
        key = tuple(_key_name(k) for k in path)
        where = f"{name} leaf {'/'.join(key)} at step {step}"
        if key not in saved_flat:
            if not fill:
                raise ValueError(f"The checkpoint has no {where}")
            leaves.append(leaf)
            filled.append("/".join(key))
            continue
        value = saved_flat.pop(key)
        if np.shape(value) != np.shape(leaf):
            raise ValueError(
                f"The {where} has shape {np.shape(value)}, not {np.shape(leaf)}"
            )
        leaves.append(np.asarray(value, dtype=np.asarray(leaf).dtype))
    if saved_flat:
        unexpected = ", ".join("/".join(k) for k in saved_flat)
        raise ValueError(f"Step {step}'s {name} has leaves the template lacks: {unexpected}")
    if filled:
        print(f"Step {step}'s {name} lacks frozen leaves, kept from the template: {', '.join(filled)}")
    return jax.tree_util.tree_unflatten(treedef, leaves)


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
    the trained one; only its structure, shapes and dtypes are used, except for
    frozen parameters the checkpoint predates, which are kept from it. The
    optimizer state and dataset iterator in the checkpoint are ignored.
    """
    if step is None:
        step = latest_step(checkpoint_dir)
    host_template = jax.tree.map(np.asarray, template)
    frozen = jax.tree.map(lambda m: not m, module.trainable_mask(host_template))
    manager = _manager(checkpoint_dir)
    try:
        restored = restore_tree(manager, step, "model", host_template, frozen)
    finally:
        manager.close()
    return jax.tree.map(jnp.asarray, restored)


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
