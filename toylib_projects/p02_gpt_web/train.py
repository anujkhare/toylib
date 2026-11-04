"""Training script for the decoder-only transformer model.

On ClimbMix, from token files written by webtext.py:
    uv run python -m toylib_projects.p02_gpt_web.train --run local

The d20 run goes through notebooks/01_d20_climbmix_colab.ipynb.

The older nanochat-style setup, tokenizing parquet shards as it reads them:
    from toylib_projects.p02_gpt_web import train

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

import dataclasses
import datetime
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pathlib

from toylib.harness import analyze as harness_analyze
from toylib.harness import metrics as harness_metrics
from toylib_projects.p02_gpt_web import analyze
from toylib_projects.p02_gpt_web import data
from toylib_projects.p02_gpt_web import decoder_only_model
from toylib_projects.p02_gpt_web import experiment
from toylib_projects.p02_gpt_web import metrics as metrics_module
from toylib_projects.p02_gpt_web import webtext
from toylib_projects.p02_gpt_web.scripts import compile as compile_utils
from toylib_projects.p02_gpt_web.tokenizer import bytes_per_token


OUTPUT_DIR = pathlib.Path(__file__).parent / "outputs"


def warmup_cosine_schedule(
    peak_lr: float, max_steps: int, warmup_steps: int
) -> optax.Schedule:
    """Ramp up to `peak_lr`, then decay to a tenth of it (nanoGPT's `get_lr`)."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=max_steps,
        end_value=peak_lr / 10,
    )


def create_adamw_optimizer_config(
    learning_rate: float | optax.Schedule,
    weight_decay: float = 0.1,
) -> experiment.MultiOptimizerConfig:
    """Create one AdamW for every parameter, with nanoGPT's betas and weight decay."""
    return experiment.MultiOptimizerConfig(
        optimizer_configs=[
            experiment.OptimizerConfig(
                name="adamw",
                optimizer=optax.adamw(
                    learning_rate=learning_rate,
                    b1=0.9,
                    b2=0.95,
                    weight_decay=weight_decay,
                ),
            )
        ],
        optimizer_for_param=lambda key_path: "adamw",
    )


def create_muon_adam_multi_optimizer_config(
    muon_lr: float | optax.Schedule = 1e-4,
    adamw_embed_lr: float | optax.Schedule = 1e-4,
    adamw_output_lr: float | optax.Schedule = 1e-4,
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


def find_num_heads(model_dim: int, target_head_dim: int = 128) -> int:
    """Pick the valid head count closest to `model_dim / target_head_dim`.

    Valid means the dimension splits evenly across heads, and each head's dimension
    is even, because RoPE rotates pairs of dimensions.
    """
    ideal = max(1, (model_dim + target_head_dim - 1) // target_head_dim)  # ceil div
    valid = [
        h
        for h in range(1, model_dim + 1)
        if model_dim % h == 0 and (model_dim // h) % 2 == 0
    ]
    return min(valid, key=lambda h: (abs(h - ideal), h))


def get_model_config(
    depth: int,
    seq_len: int = 1024,
    vocab_size: int = 50257,
    dtype: object = jnp.bfloat16,
) -> decoder_only_model.ModelConfig:
    # Rules of thumbs copied over from the nanochat repo
    num_layers = depth
    model_dim = (
        depth * 64
    )  # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
    num_heads = find_num_heads(model_dim)  # head dim about 128
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
        remat_policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        dtype=dtype,  # bf16 for accelerators (optimizer state stays float32 in optax); CPUs emulate it slowly
    )


def make_dataset(
    dataset_path: str,
    split: str,
    batch_size: int,
    seq_len: int,
    vocab_size: int = 50257,
    tokenizer_batch_size: int = 8,
    use_dummy: bool = False,
    repeatable: bool = False,
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
        repeatable: If True, every `iter()` starts over (for validation)

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
        repeatable=repeatable,
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
    muon_lr: float = 2e-2,
    adamw_embed_lr: float = 2e-1,
    adamw_output_lr: float = 4e-3,
    wandb_project_name: str = "tinystories",
    wandb_username: str = "",
    use_dummy_data: bool = False,
    run_id: str | None = None,
    bpt_path: str = "/tmp/bpt_gpt2.npy",
    warmup_steps: int | None = None,
    max_grad_norm: float = 0.0,
    num_eval_steps: int = 1,
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
        muon_lr: Learning rate for Muon optimizer
        adamw_embed_lr: Learning rate for Adam optimizer (embeddings)
        adamw_output_lr: Learning rate for Adam optimizer (output)
        wandb_project_name: Weights & Biases project name for logging
        wandb_username: Weights & Biases entity; blank for the account's default
        use_dummy_data: If True, use DummyDataset for compilation analysis
        warmup_steps: If set, use `warmup_cosine_schedule`; otherwise constant LRs
        max_grad_norm: Global gradient clipping threshold (0.0 disables)
        num_eval_steps: Validation batches per eval

    Returns:
        Initialized Experiment
    """
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

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

    # Generate the bytes per token counts
    bpt_arr = bytes_per_token.compute_bytes_per_token(tokenizer_name="gpt2")
    np.save(bpt_path, bpt_arr)

    val_task = None
    if dataset_val_split is not None and not use_dummy_data:
        val_dataset = make_dataset(
            dataset_path=dataset_path,
            split=dataset_val_split,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            use_dummy=False,
            repeatable=True,
        )
        val_task = experiment.Task(
            name="val",
            dataset=val_dataset,
            metrics=[harness_metrics.Loss(), metrics_module.BitsPerByte(bpt_path)],
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

    def schedule(peak_lr: float) -> float | optax.Schedule:
        if warmup_steps is None:
            return peak_lr
        return warmup_cosine_schedule(peak_lr, max_steps, warmup_steps)

    optimizer_config = create_muon_adam_multi_optimizer_config(
        muon_lr=schedule(muon_lr),
        adamw_embed_lr=schedule(adamw_embed_lr * dmodel_lr_scale),
        adamw_output_lr=schedule(adamw_output_lr * dmodel_lr_scale),
    )

    # Experiment
    exp = experiment.Experiment(
        model_config=model_config,
        training_config=experiment.TrainingConfig(
            max_steps=max_steps,
            num_microbatches=num_microbatches,
            max_grad_norm=max_grad_norm,
            optimizer_config=optimizer_config,
        ),
        eval_config=experiment.EvalConfig(num_eval_steps=num_eval_steps),
        checkpoint_config=experiment.CheckpointConfig(
            save_interval_steps=2500,
            max_to_keep=10,
            checkpoint_dir=f"{checkpoint_dir}/{run_id}",
            checkpoint_dataset_iterator=False,
        ),
        logger_config=experiment.WandBLoggerConfig(
            project_name=wandb_project_name,
            user_name=wandb_username,
            run_id=run_id,
        ),
        train_task=train_task,
        eval_task=val_task,
    )

    # Initialize model and optimizer state
    exp.init_state()

    # Print stats
    analyze.print_estimated_tokens(exp)
    analyze.print_chinchilla_estimate(exp.model)
    print(harness_analyze.print_param_sizes(exp.model, depth=3)[0])
    return exp


@dataclasses.dataclass(frozen=True)
class Run:
    """One training configuration, on token files written by webtext.py."""

    num_layers: int
    num_heads: int
    qkv_dim: int
    batch_size: int
    max_steps: int
    seq_len: int = 256
    vocab_size: int = 32_768
    num_train_shards: int = 8  # About 54M tokens each.
    num_tokenizer_shards: int = 8
    optimizer: str = "adamw"  # Or "muon", nanochat's split with its learning rates.
    peak_lr: float = 1e-3  # For AdamW.
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    num_microbatches: int = 1
    dtype: object = jnp.float32
    remat: bool = False  # Trades speed for memory.
    eval_interval_steps: int = 250
    num_eval_steps: int = 20
    save_interval_steps: int = 500


RUNS = {
    # Checks the pipeline in about a minute.
    "smoke": Run(
        num_layers=2, num_heads=4, qkv_dim=64, batch_size=8, max_steps=100,
        seq_len=64, vocab_size=1_000, num_train_shards=2, num_tokenizer_shards=2,
        warmup_steps=10, eval_interval_steps=50, num_eval_steps=2,
        save_interval_steps=50,
    ),
    "local": Run(
        num_layers=4, num_heads=8, qkv_dim=256, batch_size=32, max_steps=5_000,
        optimizer="muon",
    ),
    # 12 tokens per parameter, nanochat's default, at 131,072 tokens a step.
    "d20": Run(
        num_layers=20, num_heads=10, qkv_dim=1_280, batch_size=64, max_steps=44_000,
        seq_len=2_048, num_train_shards=120, optimizer="muon", warmup_steps=500,
        num_microbatches=4, dtype=jnp.bfloat16, remat=True, eval_interval_steps=500,
        save_interval_steps=2_500,
    ),
}


def model_config(run: Run) -> decoder_only_model.ModelConfig:
    remat_policy = None
    if run.remat:
        remat_policy = jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    return decoder_only_model.ModelConfig(
        num_layers=run.num_layers,
        num_heads=run.num_heads,
        qkv_dim=run.qkv_dim,
        vocab_size=run.vocab_size,
        seq_len=run.seq_len,
        dtype=run.dtype,
        remat_policy=remat_policy,
    )


def create_run_experiment(
    run: Run,
    run_id: str = "local",
    data_dir: pathlib.Path | None = None,
    output_dir: pathlib.Path = OUTPUT_DIR,
    checkpoint_dir: str | None = None,
    logger_config: experiment.LoggerConfig | None = None,
) -> experiment.Experiment:
    """Create an experiment that trains on the token files from webtext.py.

    Logs and checkpoints go under `output_dir` unless `checkpoint_dir`, which can be a
    gs:// path, or `logger_config` say otherwise.
    """
    run_dir = pathlib.Path(output_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = data_dir or webtext.data_dir_for(run.num_train_shards)

    tokenizer = webtext.load_tokenizer(run.vocab_size, data_dir)
    bpt_path = run_dir / "bytes_per_token.npy"
    np.save(bpt_path, tokenizer.bytes_per_token())

    def dataset(split: str, seed: int, repeatable: bool) -> data.TokenBinDataset:
        return data.TokenBinDataset(
            path=str(webtext.bin_path(split, run.vocab_size, data_dir)),
            batch_size=run.batch_size,
            seq_len=run.seq_len,
            seed=seed,
            repeatable=repeatable,
            tokenizer=tokenizer,
        )

    def schedule(peak_lr: float) -> optax.Schedule:
        return warmup_cosine_schedule(peak_lr, run.max_steps, run.warmup_steps)

    if run.optimizer == "muon":
        dmodel_lr_scale = (run.qkv_dim / 768) ** -0.5
        optimizer_config = create_muon_adam_multi_optimizer_config(
            muon_lr=schedule(2e-2),
            adamw_embed_lr=schedule(2e-1 * dmodel_lr_scale),
            adamw_output_lr=schedule(4e-3 * dmodel_lr_scale),
        )
    else:
        optimizer_config = create_adamw_optimizer_config(
            schedule(run.peak_lr), run.weight_decay
        )

    exp = experiment.Experiment(
        model_config=model_config(run),
        training_config=experiment.TrainingConfig(
            max_steps=run.max_steps,
            num_microbatches=run.num_microbatches,
            max_grad_norm=run.max_grad_norm,
            optimizer_config=optimizer_config,
        ),
        eval_config=experiment.EvalConfig(
            eval_interval_steps=run.eval_interval_steps,
            num_eval_steps=run.num_eval_steps,
        ),
        checkpoint_config=experiment.CheckpointConfig(
            save_interval_steps=run.save_interval_steps,
            max_to_keep=3,
            checkpoint_dir=checkpoint_dir or str(run_dir / "checkpoints"),
            checkpoint_dataset_iterator=True,
        ),
        logger_config=logger_config
        or experiment.LoggerConfig(
            log_dir=str(run_dir), run_id=run_id, train_log_interval_steps=10
        ),
        train_task=experiment.Task(
            name="train", dataset=dataset("train", seed=0, repeatable=False)
        ),
        eval_task=experiment.Task(
            name="val",
            dataset=dataset("val", seed=1, repeatable=True),
            metrics=[harness_metrics.Loss(), metrics_module.BitsPerByte(str(bpt_path))],
        ),
        sampling_max_tokens=40,
    )
    exp.init_state()
    analyze.print_estimated_tokens(exp)
    analyze.print_chinchilla_estimate(exp.model)
    return exp


def main(use_dummy_data: bool = False, compile: bool = False) -> None:
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

    if compile:
        compile_utils.run_compilation_analysis(exp, output_dir="/tmp/compile_analysis")
    else:
        exp.outer_loop()

    return exp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or compile the model")
    parser.add_argument(
        "--run",
        choices=sorted(RUNS),
        help="Train this configuration",
    )
    parser.add_argument("--run-id", help="Output folder name (default: the --run name)")
    parser.add_argument(
        "--resume", action="store_true", help="Continue from the newest checkpoint"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Run compilation analysis instead of training",
    )
    args = parser.parse_args()

    if args.run:
        exp = create_run_experiment(RUNS[args.run], run_id=args.run_id or args.run)
        if args.resume:
            exp.resume()
        exp.outer_loop()
        exp.cleanup()
    else:
        _ = main(compile=args.compile, use_dummy_data=args.compile)
