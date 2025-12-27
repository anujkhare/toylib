# import local modules
from toylib_projects.tinystories import data
from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import experiment


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
    batch_size: int = 8,
    seq_len: int = 2048,
    depth: int = 12,
    vocab_size: int = 50257,
    checkpoint_dir: str = "/tmp/checkpoints",
    dataset_path: str = "/tmp/",
    dataset_train_split: str = "train",
    dataset_val_split: str | None = "val",
) -> experiment.Experiment:
    # Dataloader
    train_task = experiment.Task(
        name="train",
        dataset=data.BatchedTokenizedDatasetParquet(
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
            dataset=data.BatchedTokenizedDatasetParquet(
                dataset_path=dataset_path,
                split=dataset_val_split,
                batch_size=batch_size,
                seq_len=seq_len,
                tokenizer_batch_size=8,
            ),
        )
    # Experiment
    exp = experiment.Experiment(
        model_config=get_model_config(
            depth=depth, seq_len=seq_len, vocab_size=vocab_size
        ),
        training_config=experiment.TrainingConfig(
            learning_rate=1e-3,
            max_steps=100_000,
        ),
        checkpoint_config=experiment.CheckpointConfig(
            save_interval_steps=2500,
            max_to_keep=10,
            checkpoint_dir=checkpoint_dir,
            save_dataset_iterator=False,
        ),
        train_task=train_task,
        eval_task=val_task,
    )
    return exp


if __name__ == "__main__":
    exp = create_experiment(
        batch_size=48,
    )
    exp.init_state()
    exp.outer_loop()
