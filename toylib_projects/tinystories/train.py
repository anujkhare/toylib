# import local modules
from toylib_projects.tinystories import data
from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import experiment


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
