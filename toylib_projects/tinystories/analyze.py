import jax
import jaxtyping as jt
import math
import pandas as pd
import typing


def get_tree_stats(model: jt.PyTree) -> pd.DataFrame:
    """
    Groups parameter counts and MiB sizes at a specified depth.
    """
    results = []
    leaf_stats = [
        (k, v.shape, v.dtype) for k, v in jax.tree_util.tree_leaves_with_path(model)
    ]

    for path, shape, dtype in leaf_stats:
        path = [str(p) for p in path]
        count = math.prod(shape)
        nbytes = count * dtype.itemsize
        results.append(
            {
                "params": count,
                "n_bytes": nbytes,
                "path": "/".join(path),
            }
        )
        for i, p in enumerate(path):
            results[-1][f"level_{i}"] = p
    return pd.DataFrame(results)


def print_param_sizes(
    model: jt.PyTree, depth: int = 1, size_denom: int = 1
) -> tuple[pd.DataFrame, int, int]:
    """
    Analyzes parameters and the compiled XLA HLO for peak memory usage.
    """
    df_stats = get_tree_stats(model)
    if len(df_stats) == 0:
        print("Model has no parameters.")
        return pd.DataFrame()
    df_stats.loc[:, "n_bytes_divided"] = df_stats["n_bytes"] / size_denom
    total_params = df_stats.params.sum()
    total_bytes = df_stats.n_bytes_divided.sum()
    print(f"Total Parameters: {total_params}. Bytes: ({total_bytes:.2f})")
    return (
        df_stats.fillna("")
        .groupby([f"level_{i}" for i in range(depth)])
        .sum()[["params", "n_bytes_divided"]]
        .reset_index(),
        total_params,
        total_bytes,
    )


def print_xla_memory_analysis(
    train_step_fn: typing.Callable,
    params: jt.PyTree,
    batch: typing.Mapping[str, jt.Array],
):
    # NOTE: train_step_fn should usually be your grad function or update function
    lowered = jax.jit(train_step_fn).lower(params, batch)
    compiled = lowered.compile()
    analysis = compiled.memory_analysis()

    def _to_mib(b: int) -> float:
        return b / (1024**2)

    print("\n--- XLA Compilation Estimate ---")
    print(
        f"Arguments (Params + Batch):\t{_to_mib(analysis.argument_size_in_bytes):.2f} MiB"
    )
    print(f"Output (Grads + Loss):\t{_to_mib(analysis.output_size_in_bytes):.2f} MiB")
    print(f"Temp/Activations (Peak):\t{_to_mib(analysis.temp_size_in_bytes):.2f} MiB")
    print(
        f"Total Peak Memory:\t{_to_mib(analysis.temp_size_in_bytes + analysis.argument_size_in_bytes):.2f} MiB"
    )


def print_estimated_tokens(exp) -> int:
    """Estimate total number of tokens processed during training."""
    total_tokens = (
        exp.training_config.max_steps
        * exp.train_task.dataset.batch_size
        * exp.train_task.dataset.seq_len
    )
    print("------------------------------")
    print("Token Analysis:")
    print("------------------------------")
    print("Batch size:", exp.train_task.dataset.batch_size)
    print("Seq len:", exp.train_task.dataset.seq_len)
    print("Max steps:", exp.training_config.max_steps)
    print(
        "Num microbatches (split from within batch_size):",
        exp.training_config.num_microbatches,
    )
    print("Total training tokens:", total_tokens)
    print("------------------------------")


def print_chinchilla_estimate(model: jt.PyTree):
    _, model_params, _ = print_param_sizes(model, depth=2, size_denom=1)
    print("------------------------------")
    print("Chinchilla Analysis:")
    print("------------------------------")
    print("Model parameters:", model_params)
    print("Chinchilla estimate: (20 * model_params):", 20 * model_params, "tokens")
    print("------------------------------")
