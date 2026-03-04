"""Training script for TinyStories transformer model.

Usage (Colab/Interactive):
    from toylib_projects.tinystories import train

    # For training:
    exp = train.create_experiment(
        dataset_path="/path/to/data",
        checkpoint_dir="/path/to/checkpoints",
    )
    exp.outer_loop()

    # For compilation analysis:
    exp = train.create_experiment(use_dummy_data=True, depth=4)
    compile_utils.run_compilation_analysis(exp, output_dir="/tmp/compile_analysis")
"""

import jax

from toylib_projects.tinystories import analyze
from toylib_projects.tinystories import data
from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import experiment
from toylib_projects.tinystories import metrics as metrics_module
from toylib_projects.tinystories.scripts import compile as compile_utils


def create_muon_adam_multi_optimizer_config(
    muon_lr: float = 1e-4,
    adamw_embed_lr: float = 1e-4,
    adamw_output_lr: float = 1e-4,
    weight_decay: float = 0.0,
) -> experiment.MultiOptimizerConfig:
    """Create multi-optimizer config with Muon for blocks, Adam for embeddings/output.

    Optimizer routing:
    - embedding_layer -> Adam (embeddings typically need different treatment)
    - output_layer -> Adam (output projection)
    - everything else (blocks with causal_attn and mlp) -> Muon

    Args:
        muon_lr: Learning rate for Muon optimizer
        adamw_embed_lr: Learning rate for Adam optimizer (embeddings)
        adamw_output_lr: Learning rate for Adam optimizer (output)

    Returns:
        MultiOptimizerConfig ready for TrainingConfig
    """
    import optax

    def optimizer_for_param(key_path: tuple) -> str:
        """Route parameters to optimizers based on their path in the model tree."""
        # Extract string keys from the path (handles GetAttrKey and other types)
        path_strs = []
        for k in key_path:
            if hasattr(k, "key"):
                path_strs.append(k.key if isinstance(k.key, str) else str(k.key))
            else:
                path_strs.append(str(k))

        # Use Adam for embedding and output layers
        if "embedding_layer" in path_strs:
            return "adamw_embed"
        if "output_layer" in path_strs:
            return "adamw_output"
        # Use Muon for transformer blocks (attention and MLP)
        return "muon"

    optimizer_configs = [
        experiment.OptimizerConfig(
            name="muon",
            optimizer=optax.contrib.muon(learning_rate=muon_lr),
        ),
        experiment.OptimizerConfig(
            name="adamw_embed",
            optimizer=optax.adamw(
                learning_rate=adamw_embed_lr,
                b1=0.8,
                b2=0.95,
                eps=1e-10,
                weight_decay=weight_decay,
            ),
        ),
        experiment.OptimizerConfig(
            name="adamw_output",
            optimizer=optax.adamw(
                learning_rate=adamw_output_lr,
                b1=0.8,
                b2=0.95,
                eps=1e-10,
                weight_decay=weight_decay,
            ),
        ),
    ]

    return experiment.MultiOptimizerConfig(
        optimizer_configs=optimizer_configs,
        optimizer_for_param=optimizer_for_param,
    )


def get_model_config(
    depth: int, seq_len: int = 1024, vocab_size: int = 50257
) -> decoder_only_model.ModelConfig:
    # Rules of thumbs copied over from the nanochat repo
    num_layers = depth
    model_dim = (
        depth * 64
    )  # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
    num_heads = max(
        1, (model_dim + 127) // 128
    )  # head dim 128 (the division here is ceil div)
    num_kv_heads = num_heads  # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
    print(f"num_layers: {num_layers}")
    print(f"model_dim: {model_dim}")
    print(f"num_heads: {num_heads}")
    print(f"num_kv_heads: {num_kv_heads}")
    return decoder_only_model.ModelConfig(
        num_layers=depth,
        num_heads=num_heads,
        qkv_dim=model_dim,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )


def make_dataset(
    dataset_path: str,
    split: str,
    batch_size: int,
    seq_len: int,
    vocab_size: int = 50257,
    tokenizer_batch_size: int = 8,
    use_dummy: bool = False,
) -> data.BatchedTokenizedDataset | compile_utils.DummyDataset:
    """Create a dataset for training or compilation analysis.

    Args:
        dataset_path: Path to the dataset
        split: Dataset split (e.g., "train", "val")
        batch_size: Total batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        tokenizer_batch_size: Batch size for tokenizer
        use_dummy: If True, return a DummyDataset for compilation analysis

    Returns:
        Dataset instance
    """
    if use_dummy:
        return compile_utils.DummyDataset(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
        )

    return data.BatchedTokenizedDatasetGrain(
        dataset_path=dataset_path,
        split=split,
        batch_size=batch_size,
        seq_len=seq_len,
        tokenizer_batch_size=tokenizer_batch_size,
    )


def create_experiment(
    batch_size_per_device: int = 18,
    seq_len: int = 2048,
    max_steps: int = 12000,
    num_microbatches: int = 2,
    depth: int = 12,
    vocab_size: int = 50257,
    checkpoint_dir: str = "/tmp/checkpoints",
    dataset_path: str = "/tmp/",
    dataset_train_split: str = "train",
    dataset_val_split: str | None = "val",
    bpt_path: str = "/tmp/bpt_gpt2.npy",
    muon_lr: float = 2e-2,
    adamw_embed_lr: float = 2e-1,
    adamw_output_lr: float = 4e-3,
    use_dummy_data: bool = False,
) -> experiment.Experiment:
    """Create an experiment for training or compilation analysis.

    Args:
        batch_size_per_device: Batch size per device
        seq_len: Sequence length
        max_steps: Maximum training steps
        num_microbatches: Number of microbatches for gradient accumulation
        depth: Model depth (number of transformer layers)
        vocab_size: Vocabulary size
        checkpoint_dir: Directory for checkpoints
        dataset_path: Path to the dataset
        dataset_train_split: Training split name
        dataset_val_split: Validation split name (None to skip)
        bpt_path: Path to bytes-per-token file for BitsPerByte metric
        muon_lr: Learning rate for Muon optimizer
        adamw_embed_lr: Learning rate for Adam optimizer (embeddings)
        adamw_output_lr: Learning rate for Adam optimizer (output)
        use_dummy_data: If True, use DummyDataset for compilation analysis

    Returns:
        Initialized Experiment
    """
    # The batch is sharded across devices and then split into microbatches
    batch_size = batch_size_per_device * jax.local_device_count() * num_microbatches

    # Create datasets
    train_dataset = make_dataset(
        dataset_path=dataset_path,
        split=dataset_train_split,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        use_dummy=use_dummy_data,
    )
    train_task = experiment.Task(name="train", dataset=train_dataset)

    val_task = None
    if dataset_val_split is not None and not use_dummy_data:
        val_dataset = make_dataset(
            dataset_path=dataset_path,
            split=dataset_val_split,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            use_dummy=False,
        )
        val_task = experiment.Task(
            name="val",
            dataset=val_dataset,
            metrics=[metrics_module.Loss(), metrics_module.BitsPerByte(bpt_path)],
        )
    elif use_dummy_data:
        # For compilation analysis, create a dummy eval task
        val_dataset = make_dataset(
            dataset_path=dataset_path,
            split="val",
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            use_dummy=True,
        )
        val_task = experiment.Task(name="val", dataset=val_dataset)

    model_config = get_model_config(depth=depth, seq_len=seq_len, vocab_size=vocab_size)

    # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
    model_dim = model_config.qkv_dim
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print(
        f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
    )

    optimizer_config = create_muon_adam_multi_optimizer_config(
        muon_lr=muon_lr,
        adamw_embed_lr=adamw_embed_lr * dmodel_lr_scale,
        adamw_output_lr=adamw_output_lr * dmodel_lr_scale,
    )

    # Experiment
    exp = experiment.Experiment(
        model_config=model_config,
        training_config=experiment.TrainingConfig(
            max_steps=max_steps,
            num_microbatches=num_microbatches,
            max_grad_norm=0.0,  # no gradient clipping enabled
            optimizer_config=optimizer_config,
        ),
        checkpoint_config=experiment.CheckpointConfig(
            save_interval_steps=2500,
            max_to_keep=10,
            checkpoint_dir=checkpoint_dir,
            checkpoint_dataset_iterator=False,
        ),
        logger_config=experiment.LoggerConfig(
            log_dir=checkpoint_dir,
        ),
        train_task=train_task,
        eval_task=val_task,
    )

    # Initialize model and optimizer state
    exp.init_state()

    # Print stats
    analyze.print_estimated_tokens(exp)
    analyze.print_chinchilla_estimate(exp.model)
    print(analyze.print_param_sizes(exp.model, depth=3)[0])
    return exp


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train or compile TinyStories model")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Run compilation analysis instead of training",
    )
    args = parser.parse_args()
    use_dummy_data = args.compile  # Use dummy data for compilation analysis

    exp = create_experiment(
        batch_size_per_device=18,
        seq_len=2048,
        max_steps=12000,
        num_microbatches=2,
        depth=12,
        vocab_size=50257,
        checkpoint_dir="/tmp/checkpoints",
        muon_lr=1e-4,
        adamw_embed_lr=1e-4,
        adamw_output_lr=1e-4,
        use_dummy_data=use_dummy_data,
    )

    if args.compile:
        compile_utils.run_compilation_analysis(exp, output_dir="/tmp/compile_analysis")
    else:
        exp.outer_loop()

    exp.cleanup()


if __name__ == "__main__":
    main()
