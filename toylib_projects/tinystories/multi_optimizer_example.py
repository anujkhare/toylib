"""Example demonstrating how to use different optimizers for different model parts.

This example shows two approaches:
1. Declarative: Define optimizers in TrainingConfig.multi_optimizer_config (recommended)
2. Dynamic: Build the multi-optimizer config at runtime (more flexible)
"""

import optax
from toylib_projects.tinystories import experiment


# Approach 1: Declarative configuration (Recommended)
# ====================================================


def create_multi_optimizer_experiment_declarative(train_task, eval_task):
    """Create an experiment with declarative optimizer configuration."""

    # Define the optimizer configurations
    optimizer_configs = [
        experiment.OptimizerConfig(
            name="embedding_opt", optimizer=optax.adam(learning_rate=1e-4)
        ),
        experiment.OptimizerConfig(
            name="attention_opt", optimizer=optax.adam(learning_rate=5e-4)
        ),
        experiment.OptimizerConfig(
            name="mlp_opt", optimizer=optax.adamw(learning_rate=1e-3, weight_decay=0.01)
        ),
        experiment.OptimizerConfig(
            name="default", optimizer=optax.adam(learning_rate=1e-3)
        ),
    ]

    def optimizer_for_param(key_path: tuple) -> str:
        path_strs = [str(k.key) for k in key_path if hasattr(k, "key")]

        if "embedding_layer" in path_strs:
            return "embedding_opt"
        elif "causal_attn" in path_strs:
            return "attention_opt"
        elif "mlp" in path_strs:
            return "mlp_opt"
        else:
            return "default"

    multi_optimizer_config = experiment.MultiOptimizerConfig(
        optimizer_configs=optimizer_configs,
        optimizer_for_param=optimizer_for_param,
    )

    # Create the experiment with optimizer configs
    exp = experiment.Experiment(
        train_task=train_task,
        eval_task=eval_task,
        training_config=experiment.TrainingConfig(
            learning_rate=1e-3,
            max_steps=100_000,
            max_grad_norm=1.0,
            optimizer_config=multi_optimizer_config,
        ),
    )

    return exp


# Approach 2: Dynamic config (More flexible, but more verbose)
# ===============================================================


def create_multi_optimizer_experiment_dynamic(train_task, eval_task, use_weight_decay):
    """Create an experiment with runtime-dependent optimizer configuration."""

    def optimizer_for_param(key_path: tuple) -> str:
        path_strs = [str(k.key) for k in key_path if hasattr(k, "key")]

        if "embedding_layer" in path_strs:
            return "embedding_opt"
        if "causal_attn" in path_strs:
            return "attention_opt"
        if "mlp" in path_strs:
            return "mlp_opt"
        return "default"

    mlp_optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.01)
    if not use_weight_decay:
        mlp_optimizer = optax.adam(learning_rate=1e-3)

    optimizer_configs = [
        experiment.OptimizerConfig(
            name="embedding_opt", optimizer=optax.adam(learning_rate=1e-4)
        ),
        experiment.OptimizerConfig(
            name="attention_opt", optimizer=optax.adam(learning_rate=5e-4)
        ),
        experiment.OptimizerConfig(name="mlp_opt", optimizer=mlp_optimizer),
        experiment.OptimizerConfig(
            name="default", optimizer=optax.adam(learning_rate=1e-3)
        ),
    ]

    multi_optimizer_config = experiment.MultiOptimizerConfig(
        optimizer_configs=optimizer_configs,
        optimizer_for_param=optimizer_for_param,
    )

    exp = experiment.Experiment(
        train_task=train_task,
        eval_task=eval_task,
        training_config=experiment.TrainingConfig(
            learning_rate=1e-3,
            max_steps=100_000,
            max_grad_norm=1.0,
            optimizer_config=multi_optimizer_config,
        ),
    )

    return exp


# Example usage:
# ==============

# # Approach 1 (Declarative):
# exp = create_multi_optimizer_experiment_declarative(train_task, eval_task)
# exp.init_state()
# exp.print_optimizer_assignment(exp.model)  # Optional debugging
# exp.outer_loop()

# # Approach 2 (Dynamic config):
# exp = create_multi_optimizer_experiment_dynamic(
#     train_task=train_task,
#     eval_task=eval_task,
#     use_weight_decay=True,
# )
# exp.init_state()
# exp.print_optimizer_assignment(exp.model)  # Optional debugging
# exp.outer_loop()
