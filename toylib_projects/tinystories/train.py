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


def main():
    batch_size = 8
    seq_len = 1024
    depth = 12
    vocab_size = 50257

    # Dataloader
    dataset = data.BatchedTokenizedDatasetParquet(
        dataset_path="/tmp/",
        split="train",
        batch_size=batch_size,
        seq_len=seq_len,
        tokenizer_batch_size=8,
    )
    train_task = experiment.Task(name="train", dataset=dataset)
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
            checkpoint_dir="/tmp/checkpoints",
        ),
        train_task=train_task,
    )
    exp.init_state()
    exp.outer_loop()


if __name__ == "__main__":
    main()
