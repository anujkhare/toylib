# import local modules
from toylib_projects.tinystories import data
from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import experiment


def get_model_config(depth: int, seq_len: int = 1024) -> decoder_only_model.ModelConfig:
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
        vocab_size=50257,  # GPT-2 tokenizer vocab size
        seq_len=seq_len,
        dropout_rate=0.1,
    )


def main():
    # Dataloader
    dataset = data.BatchedTokenizedDatasetParquet(
        dataset_path="/tmp/",
        split="train",
        batch_size=128,
        seq_len=512,
        tokenizer_batch_size=8,
    )
    train_task = experiment.Task(name="train", dataset=dataset)
    exp = experiment.Experiment(
        model_config=decoder_only_model.ModelConfig(
            vocab_size=50257,  # GPT-2 tokenizer vocab size
        ),
        training_config=experiment.TrainingConfig(
            batch_size=128,
            learning_rate=1e-3,
            max_steps=100_000,
        ),
        checkpoint_config=experiment.CheckpointConfig(
            save_interval_steps=5000,
            max_to_keep=10,
            checkpoint_dir="/tmp/checkpoints",
        ),
        train_task=train_task,
    )
    exp.init_state()
    exp.outer_loop()


if __name__ == "__main__":
    main()
