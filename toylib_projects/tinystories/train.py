import jax

# import local modules
from toylib_projects.tinystories import analyze
from toylib_projects.tinystories import data
from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import experiment
from toylib_projects.tinystories import metrics as metrics_module


def create_muon_adam_multi_optimizer_config(
    muon_lr: float = 1e-4,
    adam_lr: float = 1e-4,
) -> experiment.MultiOptimizerConfig:
    """Create multi-optimizer config with Muon for blocks, Adam for embeddings/output.

    Optimizer routing:
    - embedding_layer -> Adam (embeddings typically need different treatment)
    - output_layer -> Adam (output projection)
    - everything else (blocks with causal_attn and mlp) -> Muon

    Args:
        muon_lr: Learning rate for Muon optimizer
        adam_lr: Learning rate for Adam optimizer (embeddings/output)

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
        if "embedding_layer" in path_strs or "output_layer" in path_strs:
            return "adam_opt"
        # Use Muon for transformer blocks (attention and MLP)
        else:
            return "muon_opt"

    optimizer_configs = [
        experiment.OptimizerConfig(
            name="muon_opt",
            optimizer=optax.contrib.muon(learning_rate=muon_lr),
            learning_rate=muon_lr,
        ),
        experiment.OptimizerConfig(
            name="adam_opt",
            optimizer=optax.adam(learning_rate=adam_lr),
            learning_rate=adam_lr,
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
    # Multi-optimizer parameters
    use_multi_optimizer: bool = True,
    muon_lr: float = 1e-4,
    adam_lr: float = 1e-4,
    single_lr: float = 1e-3,
) -> experiment.Experiment:
    # The batch is sharded across devices and then split into microbatches
    batch_size = batch_size_per_device * jax.local_device_count() * num_microbatches

    # Dataloader
    train_task = experiment.Task(
        name="train",
        dataset=data.BatchedTokenizedDatasetGrain(
            dataset_path=dataset_path,
            split=dataset_train_split,
            batch_size=batch_size,
            seq_len=seq_len,
            tokenizer_batch_size=8,
        ),
    )
    val_task = None
    if dataset_val_split is not None:
        val_task = experiment.Task(
            name="val",
            dataset=data.BatchedTokenizedDatasetGrain(
                dataset_path=dataset_path,
                split=dataset_val_split,
                batch_size=batch_size,
                seq_len=seq_len,
                tokenizer_batch_size=8,
            ),
            metrics=[metrics_module.Loss(), metrics_module.BitsPerByte(bpt_path)],
        )

    # Configure optimizer based on mode
    multi_optimizer_config = None
    learning_rate = single_lr

    if use_multi_optimizer:
        multi_optimizer_config = create_muon_adam_multi_optimizer_config(
            muon_lr=muon_lr,
            adam_lr=adam_lr,
        )
        learning_rate = muon_lr  # Set for consistency/logging

    # Experiment
    exp = experiment.Experiment(
        model_config=get_model_config(
            depth=depth, seq_len=seq_len, vocab_size=vocab_size
        ),
        training_config=experiment.TrainingConfig(
            learning_rate=learning_rate,
            max_steps=max_steps,
            num_microbatches=num_microbatches,
            max_grad_norm=1.0,
            optimizer_config=multi_optimizer_config,
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
    analyze.print_chinchilla_estimate(exp)
    print(analyze.print_param_sizes(exp.model, depth=3)[0])
    return exp


if __name__ == "__main__":
    exp = create_experiment(
        batch_size_per_device=18,
        seq_len=2048,
        max_steps=12000,
        num_microbatches=2,
        depth=12,
        vocab_size=50257,
        checkpoint_dir="/tmp/checkpoints",
        # Multi-optimizer configuration
        use_multi_optimizer=True,
        muon_lr=1e-4,
        adam_lr=1e-4,
    )
    exp.outer_loop()
