"""Language-model sizing: training-token budget and the Chinchilla estimate."""

import jaxtyping as jt

from toylib.harness.analyze import print_param_sizes


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
    tokens_per_batch = (
        exp.train_task.dataset.batch_size * exp.train_task.dataset.seq_len
    )
    print(f"Seq len: {exp.train_task.dataset.seq_len:,}")
    print(f"Batch size: {exp.train_task.dataset.batch_size:,}")
    print(f"Tokens per batch: {tokens_per_batch:,}")
    print(f"Max steps: {exp.training_config.max_steps:,}")
    print(
        f"Num microbatches (split from within batch_size): {exp.training_config.num_microbatches:,}"
    )
    print(f"Total training tokens: {total_tokens:,}")
    print("------------------------------")


def print_chinchilla_estimate(model: jt.PyTree):
    _, model_params, _ = print_param_sizes(model, depth=2, size_denom=1)
    print("------------------------------")
    print("Chinchilla Analysis:")
    print("------------------------------")
    print(f"Model parameters: {model_params:,}")
    print(f"Chinchilla estimate: (20 * model_params): {20 * model_params:,} tokens")
    print("------------------------------")
